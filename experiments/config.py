"""
Experiment configuration model.

Every experiment is fully described by an ExperimentConfig.
Saving and loading configs ensures reproducibility.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from network.graph_builder import NetworkType
from agents.personas import DispositionType


class SeedMessage(BaseModel):
    """A message to inject at the start of the simulation."""
    content: str
    origin_agent_id: str  # "agent_000", "agent_001", etc.
    step: int = 0
    label: str = ""       # content type from dataset ("true", "fake", "misleading", ...)
    item_id: str = ""     # original dataset row ID


class ExperimentConfig(BaseModel):
    """Complete, serializable experiment specification."""

    # Identity
    name: str
    description: str = ""

    # Reproducibility
    seed: int = 42

    # Network
    n_agents: int = 20
    network_type: NetworkType = "scale_free"
    network_params: dict[str, Any] = Field(default_factory=dict)

    # Agent dispositions (fraction of each type, must sum to ~1.0)
    disposition_mix: dict[str, float] = Field(
        default_factory=lambda: {"neutral": 1.0}
    )

    # Simulation
    max_steps: int = 8
    seed_messages: list[SeedMessage] = Field(default_factory=list)

    # LLM backend
    llm_backend: Literal["openai", "ollama", "mock"] = "mock"
    llm_model: Optional[str] = None          # e.g. "gpt-4o-mini" or "llama3.2"
    llm_embedding_model: Optional[str] = None

    # Network from file (overrides n_agents + network_type when set)
    # Supported formats: edge list (.txt/.csv/.edgelist), GraphML (.graphml),
    #                    GML (.gml), GEXF (.gexf), adjacency list (.adjlist)
    network_file: Optional[str] = None

    # Custom LLM prompt template.
    # Must contain the placeholders {persona}, {memory}, {content}.
    # Leave as None to use the built-in default prompt.
    prompt_template: Optional[str] = None

    # Output
    output_dir: str = "results"
    compute_narrative_metrics: bool = True

    class Config:
        use_enum_values = True

    def to_json(self, path: str) -> None:
        import json, pathlib
        pathlib.Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        import json, pathlib
        return cls.model_validate_json(pathlib.Path(path).read_text())
