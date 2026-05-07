"""
Pre-built experiment configurations for the thesis.

These cover the core experimental conditions:
  1. Baseline diffusion across three network types
  2. Narrative drift comparison (true vs misleading message)
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
    llm_embedding_backend: str | None = None,
    max_concurrent_llm: int = 8,
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
                max_steps=max_steps,
                seed_messages=[
                    SeedMessage(content=TRUE_NEWS, origin_agent_id="agent_000")
                ],
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_embedding_model=llm_embedding_model,
                llm_embedding_backend=llm_embedding_backend,
                compute_narrative_metrics=compute_narrative_metrics,
                max_concurrent_llm=max_concurrent_llm,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Experiment 2: True vs. misleading news narrative drift
# ---------------------------------------------------------------------------

def narrative_drift_experiments(
    llm_backend: str = "mock",
    n_agents: int = 30,
    max_steps: int = 6,
    seed: int = 42,
    compute_narrative_metrics: bool = True,
    llm_model: str | None = None,
    llm_embedding_model: str | None = None,
    llm_embedding_backend: str | None = None,
    max_concurrent_llm: int = 8,
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
                max_steps=max_steps,
                seed_messages=[
                    SeedMessage(content=content, origin_agent_id="agent_000")
                ],
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_embedding_model=llm_embedding_model,
                llm_embedding_backend=llm_embedding_backend,
                compute_narrative_metrics=compute_narrative_metrics,
                max_concurrent_llm=max_concurrent_llm,
            )
        )
    return configs


def community_experiment(
    llm_backend: str = "mock",
    n_agents: int = 30,
    max_steps: int = 6,
    seed: int = 42,
    compute_narrative_metrics: bool = True,
    llm_model: str | None = None,
    llm_embedding_model: str | None = None,
    llm_embedding_backend: str | None = None,
    max_concurrent_llm: int = 8,
) -> list[ExperimentConfig]:
    """
    Three dense cliques (communities) with sparse inter-clique bridges.
    Seed originates inside one clique — tests whether and how information
    crosses community boundaries, and whether narrative drifts at the bridge.

    Runs true, fake, and misleading variants so results are comparable
    against the flat scale-free narrative experiments.
    """
    configs = []
    for label, content in [("true_news", TRUE_NEWS), ("fake_news", FAKE_NEWS), ("misleading", MISLEADING_NEWS)]:
        configs.append(
            ExperimentConfig(
                name=f"community_{label}",
                description=(
                    f"Narrative drift for {label} on community network "
                    "(3 cliques, p_in=0.8, p_out=0.05)."
                ),
                seed=seed,
                n_agents=n_agents,
                network_type="community",
                network_params={"n_cliques": 3, "p_in": 0.8, "p_out": 0.05},
                max_steps=max_steps,
                seed_messages=[
                    SeedMessage(content=content, origin_agent_id="agent_000")
                ],
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_embedding_model=llm_embedding_model,
                llm_embedding_backend=llm_embedding_backend,
                compute_narrative_metrics=compute_narrative_metrics,
                max_concurrent_llm=max_concurrent_llm,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Ablation 1: Memory on vs. off
# ---------------------------------------------------------------------------

def memory_ablation_experiments(
    llm_backend: str = "mock",
    n_agents: int = 30,
    max_steps: int = 6,
    seed: int = 42,
    compute_narrative_metrics: bool = True,
    llm_model: str | None = None,
    llm_embedding_model: str | None = None,
    llm_embedding_backend: str | None = None,
    max_concurrent_llm: int = 8,
) -> list[ExperimentConfig]:
    """
    Memory ablation: compare agent behaviour with and without episodic memory.
    Runs true_news and fake_news on scale-free (same network as narrative preset)
    so results are directly comparable to the baseline narrative experiments.
    """
    configs = []
    for memory_enabled in [True, False]:
        suffix = "memory_on" if memory_enabled else "memory_off"
        for label, content in [("true_news", TRUE_NEWS), ("fake_news", FAKE_NEWS)]:
            configs.append(
                ExperimentConfig(
                    name=f"ablation_{suffix}_{label}",
                    description=(
                        f"Memory ablation ({suffix}): {label} on scale-free network."
                    ),
                    seed=seed,
                    n_agents=n_agents,
                    network_type="scale_free",
                    max_steps=max_steps,
                    seed_messages=[
                        SeedMessage(content=content, origin_agent_id="agent_000")
                    ],
                    agent_memory_enabled=memory_enabled,
                    llm_backend=llm_backend,
                    llm_model=llm_model,
                    llm_embedding_model=llm_embedding_model,
                llm_embedding_backend=llm_embedding_backend,
                    compute_narrative_metrics=compute_narrative_metrics,
                    max_concurrent_llm=max_concurrent_llm,
                )
            )
    return configs


# ---------------------------------------------------------------------------
# Experiment 3: Topology × Content type (3×3 factorial)
# ---------------------------------------------------------------------------

def topology_x_content_experiments(
    llm_backend: str = "mock",
    n_agents: int = 30,
    max_steps: int = 6,
    seed: int = 42,
    compute_narrative_metrics: bool = True,
    llm_model: str | None = None,
    llm_embedding_model: str | None = None,
    llm_embedding_backend: str | None = None,
    max_concurrent_llm: int = 8,
) -> list[ExperimentConfig]:
    """
    3×3 factorial: all three content types (true, fake, misleading)
    on all three network topologies (random, scale_free, small_world).
    Yields 9 configs so topology and content effects can be cleanly disentangled.
    """
    configs = []
    for net_type in ["random", "scale_free", "small_world"]:
        for label, content in [
            ("true_news", TRUE_NEWS),
            ("fake_news", FAKE_NEWS),
            ("misleading", MISLEADING_NEWS),
        ]:
            configs.append(
                ExperimentConfig(
                    name=f"topo_x_content_{net_type}_{label}",
                    description=(
                        f"Topology×Content factorial: {label} on {net_type} network."
                    ),
                    seed=seed,
                    n_agents=n_agents,
                    network_type=net_type,
                    max_steps=max_steps,
                    seed_messages=[
                        SeedMessage(content=content, origin_agent_id="agent_000")
                    ],
                    llm_backend=llm_backend,
                    llm_model=llm_model,
                    llm_embedding_model=llm_embedding_model,
                llm_embedding_backend=llm_embedding_backend,
                    compute_narrative_metrics=compute_narrative_metrics,
                    max_concurrent_llm=max_concurrent_llm,
                )
            )
    return configs


def all_presets(llm_backend: str = "mock", max_concurrent_llm: int = 8, **kwargs) -> list[ExperimentConfig]:
    return (
        network_topology_experiments(llm_backend=llm_backend, max_concurrent_llm=max_concurrent_llm, **kwargs)
        + narrative_drift_experiments(llm_backend=llm_backend, max_concurrent_llm=max_concurrent_llm, **kwargs)
        + community_experiment(llm_backend=llm_backend, max_concurrent_llm=max_concurrent_llm, **kwargs)
        + topology_x_content_experiments(llm_backend=llm_backend, max_concurrent_llm=max_concurrent_llm, **kwargs)
    )
