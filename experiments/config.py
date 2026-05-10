"""
Experiment configuration model.

Every experiment is fully described by an ExperimentConfig.
Saving and loading configs ensures reproducibility.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from network.graph_builder import NetworkType


class SeedMessage(BaseModel):
    """A message to inject at the start of the simulation."""
    content: str
    origin_agent_id: str  # "agent_000", "agent_001", etc.
    step: int = 0
    label: str = ""       # content type from ISOT dataset ("true" or "fake")
    item_id: str = ""     # original dataset row ID


class Intervention(BaseModel):
    """
    A single intervention applied during the simulation.

    Three types:
      block   — remove target agents from the network before the cascade starts.
                Models deplatforming or account suspension.
      label   — prepend a fact-check notice to messages before they reach target
                agents. Models platform warning labels.
      correct — inject a counter-message at a given step. Models a debunking
                article published after misinformation has already spread.

    Targeting (for block / label): specify any combination of:
      target_agent_ids       — specific agents by ID
      target_epistemic_types — all agents of these types ("open","closed",...)
      target_top_k_hubs      — top-k agents by out-degree centrality
    All non-empty targets are unioned.
    """

    type: Literal["block", "label", "correct"]

    # --- targeting (block / label) ---
    target_agent_ids: list[str] = Field(default_factory=list)
    target_epistemic_types: list[str] = Field(default_factory=list)
    target_top_k_hubs: int = 0

    # --- timing (label / correct) ---
    at_step: int = 0          # label: apply from this step onward; correct: inject at this step

    # --- label params ---
    label_prefix: str = "[FACT-CHECKERS: This claim is disputed] "

    # --- correct params ---
    correction_content: str = ""
    correction_origin_agent_id: str = "agent_000"


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

    # Agent population composition.
    # Maps epistemic_type → fraction of agents. Must sum to ~1.0.
    # Available types: "open", "closed", "credulous", "strategic"
    # When None, all 25 persona templates are cycled in order (no composition control).
    persona_mix: Optional[dict[str, float]] = None

    # Simulation
    max_steps: int = 8
    seed_messages: list[SeedMessage] = Field(default_factory=list)

    # Max concurrent LLM calls per simulation step.
    # Lower for local models (Ollama), higher for hosted APIs (OpenAI).
    max_concurrent_llm: int = 8

    # LLM backend
    llm_backend: Literal["openai", "ollama", "mock"] = "mock"
    llm_model: Optional[str] = None          # e.g. "gpt-4o-mini" or "llama3.2"
    llm_embedding_model: Optional[str] = None
    # When set to "sentence_transformers", narrative embeddings use a local ST model
    # (llm_embedding_model is the ST model name, default "all-MiniLM-L6-v2").
    # Useful when the completion API has no embedding endpoint (e.g. MetaCentrum).
    llm_embedding_backend: Optional[str] = None

    # Network from file (overrides n_agents + network_type when set)
    # Supported formats: edge list (.txt/.csv/.edgelist), GraphML (.graphml),
    #                    GML (.gml), GEXF (.gexf), adjacency list (.adjlist)
    network_file: Optional[str] = None

    # Custom LLM prompt template.
    # Must contain the placeholders {persona}, {memory}, {content}.
    # Leave as None to use the built-in default prompt.
    prompt_template: Optional[str] = None

    # Interventions applied during the simulation (empty = no intervention / baseline)
    interventions: list[Intervention] = Field(default_factory=list)

    # Ablation flags
    agent_memory_enabled: bool = True   # False = agents have no memory of prior messages
    # Controls how much agents rewrite when forwarding (0.0 = verbatim, 1.0 = full rewrite).
    # Injected into the agent prompt as a concrete instruction.
    rewrite_intensity: float = 0.5

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
