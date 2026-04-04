"""
Core agent model for information diffusion simulation.

Agents have:
  - A persona (background, epistemic profile, communication style)
  - A simple memory of received messages
  - LLM-based decision logic: forward or not, optionally rewrite

Design principle: keep agents stateless w.r.t. the network topology.
The simulation runner feeds them messages; agents only see content + their own memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persona model
# ---------------------------------------------------------------------------

@dataclass
class AgentPersona:
    """Static description of an agent's identity and epistemic profile."""

    agent_id: str
    name: str
    epistemic_type: str       # "open" | "closed" | "credulous" | "strategic"
    background: str           # biographical intro (age, occupation, location, context)
    epistemic_profile: str    # pre-formatted bullet-point rules (Cassam framework)
    communication_style: str  # how the agent speaks and presents information
    information_behavior: str # how the agent evaluates and shares information

    def to_text(self) -> str:
        first_name = self.name.split()[0]
        return (
            f"You are {self.name}, a {self.background}\n"
            f"\nEPISTEMIC PROFILE — follow these rules strictly:\n{self.epistemic_profile}\n"
            f"\nCOMMUNICATION STYLE: {self.communication_style}\n"
            f"\nINFORMATION BEHAVIOR: {self.information_behavior}\n"
            f"\nDo not break character. Respond as {first_name} would, in first person."
        )


# ---------------------------------------------------------------------------
# LLM decision schema
# ---------------------------------------------------------------------------

class ForwardDecision(BaseModel):
    """Structured output from the agent's LLM reasoning."""

    forward: bool
    reasoning: str
    rewritten_content: Optional[str] = None  # agent's retelling; None only if not forwarding


# ---------------------------------------------------------------------------
# Agent memory
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """One episode: a message received, the agent's reasoning, and what it decided."""
    step: int
    content: str       # truncated message text
    reasoning: str     # agent's own LLM reasoning
    forwarded: bool    # what was actually decided


@dataclass
class AgentMemory:
    """
    Structured episodic memory.

    Each entry stores the message text, the agent's own reasoning, and the
    forwarding decision together. This lets subsequent LLM calls see not just
    what the agent received, but how it interpreted each message — making the
    agent's epistemic state accumulate across steps.
    """

    max_size: int = 20
    _entries: list[MemoryEntry] = field(default_factory=list)

    def add(self, step: int, content: str, reasoning: str, forwarded: bool) -> None:
        entry = MemoryEntry(
            step=step,
            content=content[:500],
            reasoning=reasoning[:300],
            forwarded=forwarded,
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size :]

    def as_text(self) -> str:
        if not self._entries:
            return "(no prior messages)"
        lines = []
        for e in self._entries:
            action = "SHARED" if e.forwarded else "IGNORED"
            lines.append(
                f"- [step {e.step}] {action}: \"{e.content}\"\n"
                f"  Your reasoning then: {e.reasoning}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DiffusionAgent:
    """
    An LLM-backed agent that participates in information diffusion.

    The agent:
    1. Receives a message from a neighbor
    2. Decides (via LLM) whether to forward it
    3. Optionally rewrites it before forwarding
    4. Stores seen messages in memory
    """

    def __init__(
        self,
        persona: AgentPersona,
        llm,                  # LLMBackend – injected at construction
        memory_size: int = 20,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.persona = persona
        self.llm = llm
        self.memory = AgentMemory(max_size=memory_size)
        self._forwarded_cascade_ids: set[str] = set()
        self.prompt_template = prompt_template

    @property
    def id(self) -> str:
        return self.persona.agent_id

    @property
    def name(self) -> str:
        return self.persona.name

    # ------------------------------------------------------------------
    # Core decision
    # ------------------------------------------------------------------

    async def evaluate_message(
        self,
        message,          # simulation.message.Message
        step: int,
    ) -> ForwardDecision:
        """
        Ask the LLM whether to forward *message* and how.
        Returns a ForwardDecision with forward/rewrite flags.
        """
        prompt = _build_decision_prompt(self.persona, self.memory, message, self.prompt_template)
        try:
            decision = await self.llm.complete_json(
                prompt, ForwardDecision, temperature=0.4
            )
        except Exception as exc:
            logger.warning(
                "Agent %s LLM decision failed (%s); defaulting to no-forward.",
                self.id,
                exc,
            )
            decision = ForwardDecision(
                forward=False,
                reasoning=f"LLM error: {exc}",
            )

        # Store the message AND the agent's own reasoning so future decisions
        # are conditioned on this agent's accumulated epistemic history.
        self.memory.add(
            step=step,
            content=message.content,
            reasoning=decision.reasoning,
            forwarded=decision.forward,
        )
        return decision

    def has_forwarded(self, cascade_id: str) -> bool:
        """True if this agent already forwarded something in this cascade."""
        return cascade_id in self._forwarded_cascade_ids

    def mark_forwarded(self, cascade_id: str) -> None:
        self._forwarded_cascade_ids.add(cascade_id)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_TEMPLATE = """\
{persona}

=== YOUR RECENT ACTIVITY ===
{memory}

=== INCOMING MESSAGE ===
"{content}"

=== TASK ===
Decide whether you would share this with your followers.
Stay fully in character — let your epistemic profile and information behavior drive the decision.

If you decide to share (forward=true), you MUST retell it in your own words as you naturally would —
the way you would actually say it to someone, not copy-pasting. Apply your own framing, emphasis,
and voice. You may add commentary, express your reaction, or embed it in a broader point you want
to make. The result should sound like you, not like the original source.

Set rewritten_content to your retelling. If you decide not to share (forward=false),
leave rewritten_content null.
"""


def _build_decision_prompt(
    persona: AgentPersona,
    memory: AgentMemory,
    message,
    template: Optional[str] = None,
) -> str:
    """
    Build the LLM decision prompt.

    If *template* is provided it must contain the placeholders:
        {persona}   – agent persona description
        {memory}    – formatted recent memory
        {content}   – incoming message text
    """
    tmpl = template if template is not None else _DEFAULT_PROMPT_TEMPLATE
    return tmpl.format(
        persona=persona.to_text(),
        memory=memory.as_text(),
        content=message.content,
    )
