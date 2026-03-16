"""
Pre-built experiment configurations for the thesis.

These cover the core experimental conditions:
  1. Baseline diffusion across three network types
  2. Agent disposition effects (accuracy vs persuasion)
  3. Narrative drift comparison (true vs misleading message)
"""

from experiments.config import ExperimentConfig, SeedMessage

TRUE_NEWS = (
    "Scientists at Johns Hopkins published a peer-reviewed study confirming that "
    "regular moderate exercise (30 min/day) reduces cardiovascular disease risk by 35%."
)

FAKE_NEWS = (
    "BREAKING: Secret government documents leaked – 5G towers are being used to "
    "transmit mind-control frequencies. Share before this gets deleted!"
)

MISLEADING_NEWS = (
    "New study shows vaccines are linked to a 35% increase in autism risk – "
    "why is the mainstream media hiding this?"
)


# ---------------------------------------------------------------------------
# Experiment 1: Network topology comparison
# ---------------------------------------------------------------------------

def network_topology_experiments(
    llm_backend: str = "mock",
    n_agents: int = 30,
    max_steps: int = 6,
    seed: int = 42,
    compute_narrative_metrics: bool = True,
    llm_model: str | None = None,
    llm_embedding_model: str | None = None,
) -> list[ExperimentConfig]:
    """Three runs differing only in network type."""
    configs = []
    for net_type in ["random", "scale_free", "small_world"]:
        configs.append(
            ExperimentConfig(
                name=f"topology_{net_type}",
                description=f"Baseline diffusion on {net_type} network.",
                seed=seed,
                n_agents=n_agents,
                network_type=net_type,
                disposition_mix={"neutral": 1.0},
                max_steps=max_steps,
                seed_messages=[
                    SeedMessage(content=TRUE_NEWS, origin_agent_id="agent_000")
                ],
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_embedding_model=llm_embedding_model,
                compute_narrative_metrics=compute_narrative_metrics,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Experiment 2: Agent disposition effects
# ---------------------------------------------------------------------------

def disposition_experiments(
    llm_backend: str = "mock",
    n_agents: int = 30,
    max_steps: int = 6,
    seed: int = 42,
    compute_narrative_metrics: bool = True,
    llm_model: str | None = None,
    llm_embedding_model: str | None = None,
) -> list[ExperimentConfig]:
    """Same network, different agent disposition mixes."""
    mixes = {
        "all_neutral": {"neutral": 1.0},
        "all_accuracy": {"accuracy-oriented": 1.0},
        "all_persuasion": {"persuasion-oriented": 1.0},
        "mixed": {
            "accuracy-oriented": 0.25,
            "persuasion-oriented": 0.25,
            "skeptical": 0.25,
            "credulous": 0.25,
        },
    }
    configs = []
    for mix_name, mix in mixes.items():
        configs.append(
            ExperimentConfig(
                name=f"disposition_{mix_name}",
                description=f"Scale-free network with {mix_name} agent mix.",
                seed=seed,
                n_agents=n_agents,
                network_type="scale_free",
                disposition_mix=mix,
                max_steps=max_steps,
                seed_messages=[
                    SeedMessage(content=FAKE_NEWS, origin_agent_id="agent_000")
                ],
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_embedding_model=llm_embedding_model,
                compute_narrative_metrics=compute_narrative_metrics,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Experiment 3: True vs. misleading news narrative drift
# ---------------------------------------------------------------------------

def narrative_drift_experiments(
    llm_backend: str = "mock",
    n_agents: int = 30,
    max_steps: int = 6,
    seed: int = 42,
    compute_narrative_metrics: bool = True,
    llm_model: str | None = None,
    llm_embedding_model: str | None = None,
) -> list[ExperimentConfig]:
    """Compare narrative evolution for true vs. misleading content."""
    configs = []
    for label, content in [("true_news", TRUE_NEWS), ("fake_news", FAKE_NEWS), ("misleading", MISLEADING_NEWS)]:
        configs.append(
            ExperimentConfig(
                name=f"narrative_{label}",
                description=f"Narrative drift for {label} on scale-free network.",
                seed=seed,
                n_agents=n_agents,
                network_type="scale_free",
                disposition_mix={"neutral": 1.0},
                max_steps=max_steps,
                seed_messages=[
                    SeedMessage(content=content, origin_agent_id="agent_000")
                ],
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_embedding_model=llm_embedding_model,
                compute_narrative_metrics=compute_narrative_metrics,
            )
        )
    return configs


def all_presets(llm_backend: str = "mock", **kwargs) -> list[ExperimentConfig]:
    return (
        network_topology_experiments(llm_backend=llm_backend, **kwargs)
        + disposition_experiments(llm_backend=llm_backend, **kwargs)
        + narrative_drift_experiments(llm_backend=llm_backend, **kwargs)
    )
