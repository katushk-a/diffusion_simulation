"""
Message model for information diffusion simulation.

Each message carries metadata that allows full reconstruction of the
diffusion cascade as a tree structure.
"""

import uuid
from typing import Optional
from pydantic import BaseModel, Field


def new_id() -> str:
    return str(uuid.uuid4())


class Message(BaseModel):
    """A single message propagating through the network."""

    id: str = Field(default_factory=new_id)
    content: str

    # Provenance
    origin_agent_id: str          # agent who first created this message
    sender_agent_id: str          # agent who sent THIS copy
    receiver_agent_id: str        # agent who received this copy

    # Cascade structure
    cascade_id: str               # shared across all messages in one diffusion event
    parent_message_id: Optional[str] = None  # None only for seed message

    # Content metadata — set on the seed, propagated unchanged through every forward
    label: str = ""               # e.g. "true", "fake", "misleading" — from the dataset
    item_id: str = ""             # original dataset row ID for traceability

    # Timing
    step: int                     # simulation step when this message was sent

    def is_seed(self) -> bool:
        return self.parent_message_id is None

    @classmethod
    def create_seed(
        cls,
        content: str,
        origin_agent_id: str,
        receiver_agent_id: str,
        step: int = 0,
        label: str = "",
        item_id: str = "",
    ) -> "Message":
        """Factory for the first message in a cascade."""
        cascade_id = new_id()
        return cls(
            content=content,
            origin_agent_id=origin_agent_id,
            sender_agent_id=origin_agent_id,
            receiver_agent_id=receiver_agent_id,
            cascade_id=cascade_id,
            parent_message_id=None,
            label=label,
            item_id=item_id,
            step=step,
        )

    def forward(
        self,
        new_content: str,
        new_sender_id: str,
        new_receiver_id: str,
        step: int,
    ) -> "Message":
        """Create a child message when an agent forwards (possibly rewriting) this one.
        label and item_id are inherited unchanged — they identify the original news item."""
        return Message(
            content=new_content,
            origin_agent_id=self.origin_agent_id,
            sender_agent_id=new_sender_id,
            receiver_agent_id=new_receiver_id,
            cascade_id=self.cascade_id,
            parent_message_id=self.id,
            label=self.label,
            item_id=self.item_id,
            step=step,
        )
