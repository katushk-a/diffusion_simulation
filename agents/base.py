"""
Core agent model for information diffusion simulation.

Agents have:
  - A persona (background, traits, behavioral disposition)
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
    """Static description of an agent's identity and behavioral disposition."""

    agent_id: str
    name: str
    background: str          # demographics, occupation, life context
    traits: str              # psychological traits / communication style
    disposition: str         # how they handle information:
                             # e.g. "accuracy-oriented", "persuasion-oriented",
                             #      "skeptical", "credulous", "neutral"

    def to_text(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Background: {self.background}\n"
            f"Traits: {self.traits}\n"
            f"Information behavior: {self.disposition}"
        )


# ---------------------------------------------------------------------------
# LLM decision schema
# ---------------------------------------------------------------------------

class ForwardDecision(BaseModel):
    """Structured output from the agent's LLM reasoning."""

    forward: bool
    reasoning: str
    rewrite: bool = False
    rewritten_content: Optional[str] = None


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
            content=content[:200],
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
You are simulating a social media user. Decide whether to share a message.

=== YOUR PROFILE ===
{persona}

=== YOUR RECENT MESSAGES ===
{memory}

=== INCOMING MESSAGE ===
"{content}"

=== TASK ===
Decide whether you would share/forward this message to your followers.
Consider your personality, past behavior, and the message content.

If you forward it, you MAY rewrite it (paraphrase, add commentary, change emphasis).
Set rewrite=true and provide rewritten_content only if you actually change the text.
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
