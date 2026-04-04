"""
Simulation runner for information diffusion over a social network.

Loop (per step):
  1. Deliver pending messages to each recipient agent
  2. Agent calls LLM to decide: forward or not, rewrite or not
  3. Forwarded messages are queued for neighbors in the next step
  4. All decisions are logged

Design notes:
  - One agent forwards at most once per cascade (prevents loops)
  - asyncio.gather is used for parallel agent decisions within a step
  - Full event log is written to SimulationLog
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from agents.base import DiffusionAgent
from network.graph_builder import get_neighbors
from simulation.message import Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

@dataclass
class SimulationEvent:
    """A single logged event during the simulation."""

    step: int
    event_type: str          # "seed" | "received" | "forwarded" | "dropped"
    agent_id: str
    message_id: str
    cascade_id: str
    content: str
    reasoning: Optional[str] = None
    parent_message_id: Optional[str] = None


@dataclass
class SimulationLog:
    events: list[SimulationEvent] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)  # all messages ever created

    def add_event(self, event: SimulationEvent) -> None:
        self.events.append(event)

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)

    def cascade_messages(self, cascade_id: str) -> list[Message]:
        return [m for m in self.messages if m.cascade_id == cascade_id]

    def to_dict(self) -> dict:
        return {
            "events": [e.__dict__ for e in self.events],
            "messages": [m.model_dump() for m in self.messages],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class DiffusionSimulation:
    """
    Orchestrates message diffusion over a social network graph.

    Parameters
    ----------
    agents:     dict mapping agent_id → DiffusionAgent
    graph:      NetworkX DiGraph with 'agent_id' node attributes
    max_steps:  maximum number of simulation steps
    """

    def __init__(
        self,
        agents: dict[str, DiffusionAgent],
        graph: nx.DiGraph,
        max_steps: int = 10,
    ) -> None:
        self.agents = agents
        self.graph = graph
        self.max_steps = max_steps
        self.log = SimulationLog()

        # step_queue[step] = list of messages to deliver at that step
        self._step_queue: dict[int, list[Message]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(
        self,
        origin_agent_id: str,
        content: str,
        step: int = 0,
        label: str = "",
        item_id: str = "",
    ) -> str:
        """
        Inject a seed message originating from *origin_agent_id*.
        Returns the cascade_id.
        """
        neighbors = get_neighbors(self.graph, origin_agent_id)
        if not neighbors:
            logger.warning("Seed agent %s has no outgoing neighbors.", origin_agent_id)
            return ""

        # Create one seed message per neighbor
        first_neighbor = neighbors[0]
        seed_msg = Message.create_seed(
            content=content,
            origin_agent_id=origin_agent_id,
            receiver_agent_id=first_neighbor,
            step=step,
        )
        cascade_id = seed_msg.cascade_id

        # Fan out seed to ALL neighbors
        for nb in neighbors:
            msg = Message.create_seed(
                content=content,
                origin_agent_id=origin_agent_id,
                receiver_agent_id=nb,
                step=step,
                label=label,
                item_id=item_id,
            )
            # Reuse cascade_id so they all belong to same cascade
            msg = msg.model_copy(update={"cascade_id": cascade_id})
            self._enqueue(step, msg)
            self.log.add_message(msg)
            self.log.add_event(
                SimulationEvent(
                    step=step,
                    event_type="seed",
                    agent_id=origin_agent_id,
                    message_id=msg.id,
                    cascade_id=cascade_id,
                    content=content,
                )
            )

        logger.info(
            "Seeded cascade %s from %s to %d neighbors.",
            cascade_id[:8],
            origin_agent_id,
            len(neighbors),
        )
        return cascade_id

    async def run(self) -> SimulationLog:
        """Run the full simulation for max_steps steps."""
        for step in range(self.max_steps):
            messages_this_step = self._step_queue.pop(step, [])
            if not messages_this_step and step > 0:
                logger.info("Step %d: no messages, simulation ends early.", step)
                break

            logger.info(
                "Step %d: delivering %d messages.", step, len(messages_this_step)
            )
            await self._process_step(step, messages_this_step)

        return self.log

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue(self, step: int, msg: Message) -> None:
        self._step_queue.setdefault(step, []).append(msg)

    async def _process_step(
        self, step: int, messages: list[Message]
    ) -> None:
        """Process all messages for one simulation step in parallel."""
        tasks = [
            self._handle_message(step, msg)
            for msg in messages
        ]
        await asyncio.gather(*tasks)

    async def _handle_message(self, step: int, msg: Message) -> None:
        """
        Single message handler:
          - Retrieve the receiving agent
          - Ask LLM for decision
          - If forwarding: create child messages and enqueue for next step
        """
        receiver_id = msg.receiver_agent_id
        agent = self.agents.get(receiver_id)
        if agent is None:
            logger.warning("Agent %s not found; dropping message.", receiver_id)
            return

        # Log receipt
        self.log.add_event(
            SimulationEvent(
                step=step,
                event_type="received",
                agent_id=receiver_id,
                message_id=msg.id,
                cascade_id=msg.cascade_id,
                content=msg.content,
                parent_message_id=msg.parent_message_id,
            )
        )

        # One forward per cascade per agent (prevents cycles)
        if agent.has_forwarded(msg.cascade_id):
            self.log.add_event(
                SimulationEvent(
                    step=step,
                    event_type="dropped",
                    agent_id=receiver_id,
                    message_id=msg.id,
                    cascade_id=msg.cascade_id,
                    content=msg.content,
                    reasoning="Already forwarded this cascade.",
                )
            )
            return

        # LLM decision
        decision = await agent.evaluate_message(msg, step)

        if not decision.forward:
            self.log.add_event(
                SimulationEvent(
                    step=step,
                    event_type="dropped",
                    agent_id=receiver_id,
                    message_id=msg.id,
                    cascade_id=msg.cascade_id,
                    content=msg.content,
                    reasoning=decision.reasoning,
                )
            )
            return

        # Use the agent's retelling when provided, fall back to original
        forwarded_content = decision.rewritten_content or msg.content

        # Fan out to all neighbors
        neighbors = get_neighbors(self.graph, receiver_id)
        if not neighbors:
            return

        agent.mark_forwarded(msg.cascade_id)

        for nb_id in neighbors:
            child = msg.forward(
                new_content=forwarded_content,
                new_sender_id=receiver_id,
                new_receiver_id=nb_id,
                step=step + 1,
            )
            self._enqueue(step + 1, child)
            self.log.add_message(child)

        self.log.add_event(
            SimulationEvent(
                step=step,
                event_type="forwarded",
                agent_id=receiver_id,
                message_id=msg.id,
                cascade_id=msg.cascade_id,
                content=forwarded_content,
                reasoning=decision.reasoning,
                parent_message_id=msg.parent_message_id,
            )
        )
        logger.debug(
            "Step %d: %s forwarded cascade %s to %d neighbors.",
            step,
            receiver_id,
            msg.cascade_id[:8],
            len(neighbors),
        )
