"""
Pre-built experiment configurations for the thesis.

All seed content is sampled from the ISOT Fake News Dataset (True.csv / Fake.csv).
Falls back to hardcoded strings if the files are not found.
"""

import pathlib
from functools import lru_cache

from data.dataset import NewsDataset
from experiments.config import ExperimentConfig, SeedMessage

# ---------------------------------------------------------------------------
# ISOT dataset — loaded once, cached for the process lifetime
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

# Hardcoded fallbacks used when CSV files are not present (e.g. mock/CI runs)
_FALLBACK = {
    "true": (
        "Scientists at Johns Hopkins published a peer-reviewed study confirming that "
        "regular moderate exercise (30 min/day) reduces cardiovascular disease risk by 35%."
    ),
    "fake": (
        "BREAKING: Secret government documents leaked – 5G towers are being used to "
        "transmit mind-control frequencies. Share before this gets deleted!"
    ),
}


@lru_cache(maxsize=1)
def _isot() -> NewsDataset | None:
    """Load and cache the ISOT dataset. Returns None if files are missing."""
    ds = NewsDataset.load_isot(data_dir=_DATA_DIR, max_chars=600)
    if len(ds) == 0:
        return None
    return ds


def _sample(label: str, seed: int, experiment: str = "") -> str:
    """
    Return one article from ISOT for the given label ('true' or 'fake').

    Mixing `experiment` into the seed means different preset types (topology,
    narrative, community, …) draw different articles even at the same base seed,
    while all conditions *within* the same preset always use the same article
    (so topology comparisons hold content constant).

    The same (label, seed, experiment) triple always returns the same article
    — fully reproducible.
    """
    ds = _isot()
    if ds is None:
        return _FALLBACK.get(label, "")
    try:
        mixed_seed = seed ^ (hash(experiment) & 0xFFFF_FFFF)
        return ds.sample(n=1, label=label, seed=mixed_seed)[0].text
    except Exception:
        return _FALLBACK.get(label, "")


# ---------------------------------------------------------------------------
# Shared content labels used across all presets
# ---------------------------------------------------------------------------

_CONTENT_TYPES = ["true", "fake"]


def _make_config(
    name: str,
    description: str,
    label: str,
    seed: int,
    n_agents: int,
    network_type: str,
    max_steps: int,
    llm_backend: str,
    llm_model: str | None,
    llm_embedding_model: str | None,
    llm_embedding_backend: str | None,
    compute_narrative_metrics: bool,
    max_concurrent_llm: int,
    network_params: dict | None = None,
    agent_memory_enabled: bool = True,
    experiment: str = "",
) -> ExperimentConfig:
    content = _sample(label, seed, experiment=experiment)
    return ExperimentConfig(
        name=name,
        description=description,
        seed=seed,
        n_agents=n_agents,
        network_type=network_type,
        network_params=network_params or {},
        max_steps=max_steps,
        seed_messages=[SeedMessage(content=content, origin_agent_id="agent_000", label=label)],
        agent_memory_enabled=agent_memory_enabled,
        llm_backend=llm_backend,
        llm_model=llm_model,
        llm_embedding_model=llm_embedding_model,
        llm_embedding_backend=llm_embedding_backend,
        compute_narrative_metrics=compute_narrative_metrics,
        max_concurrent_llm=max_concurrent_llm,
    )


# ---------------------------------------------------------------------------
# Experiment 1: Network topology comparison (3 topologies × 2 content types)
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
    """3×2 factorial: random / scale_free / small_world × true / fake (ISOT)."""
    configs = []
    for net_type in ["random", "scale_free", "small_world"]:
        for label in _CONTENT_TYPES:
            configs.append(_make_config(
                name=f"topology_{net_type}_{label}",
                description=f"{label} news diffusion on {net_type} network.",
                label=label,
                seed=seed,
                experiment="topology",
                n_agents=n_agents,
                network_type=net_type,
                max_steps=max_steps,
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_embedding_model=llm_embedding_model,
                llm_embedding_backend=llm_embedding_backend,
                compute_narrative_metrics=compute_narrative_metrics,
                max_concurrent_llm=max_concurrent_llm,
            ))
    return configs


# ---------------------------------------------------------------------------
# Experiment 2: Community network (true vs. fake)
# ---------------------------------------------------------------------------

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
    Three dense cliques with sparse inter-clique bridges.
    Tests whether information crosses community boundaries differently
    for true vs. fake news (ISOT).
    """
    configs = []
    for label in _CONTENT_TYPES:
        configs.append(_make_config(
            name=f"community_{label}",
            description=f"{label} news on community network (3 cliques, p_in=0.8, p_out=0.05).",
            label=label,
            seed=seed,
            experiment="community",
            n_agents=n_agents,
            network_type="community",
            network_params={"n_cliques": 3, "p_in": 0.8, "p_out": 0.05},
            max_steps=max_steps,
            llm_backend=llm_backend,
            llm_model=llm_model,
            llm_embedding_model=llm_embedding_model,
            llm_embedding_backend=llm_embedding_backend,
            compute_narrative_metrics=compute_narrative_metrics,
            max_concurrent_llm=max_concurrent_llm,
        ))
    return configs


# ---------------------------------------------------------------------------
# Ablation: Memory on vs. off
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
    Memory ablation: agent behaviour with and without episodic memory.
    True and fake news on scale-free network (ISOT).
    """
    configs = []
    for memory_enabled in [True, False]:
        suffix = "memory_on" if memory_enabled else "memory_off"
        for label in _CONTENT_TYPES:
            configs.append(_make_config(
                name=f"ablation_{suffix}_{label}",
                description=f"Memory ablation ({suffix}): {label} news on scale-free network.",
                label=label,
                seed=seed,
                experiment="ablation",
                n_agents=n_agents,
                network_type="scale_free",
                max_steps=max_steps,
                llm_backend=llm_backend,
                llm_model=llm_model,
                llm_embedding_model=llm_embedding_model,
                llm_embedding_backend=llm_embedding_backend,
                compute_narrative_metrics=compute_narrative_metrics,
                max_concurrent_llm=max_concurrent_llm,
                agent_memory_enabled=memory_enabled,
            ))
    return configs


def all_presets(llm_backend: str = "mock", max_concurrent_llm: int = 8, **kwargs) -> list[ExperimentConfig]:
    return (
        network_topology_experiments(llm_backend=llm_backend, max_concurrent_llm=max_concurrent_llm, **kwargs)
        + community_experiment(llm_backend=llm_backend, max_concurrent_llm=max_concurrent_llm, **kwargs)
        + memory_ablation_experiments(llm_backend=llm_backend, max_concurrent_llm=max_concurrent_llm, **kwargs)
    )
