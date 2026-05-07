"""
Experiment runner.

Takes an ExperimentConfig, wires up all components, runs the simulation,
computes metrics, and saves structured results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import networkx as nx

from agents.base import DiffusionAgent
from agents.personas import build_agent_personas
from experiments.config import ExperimentConfig, Intervention
from metrics.cascade import CascadeMetrics, compute_all_cascades
from metrics.narrative import CascadeNarrativeStats, NarrativeTracker
from network.graph_builder import (
    assign_agents_to_graph,
    build_graph,
    graph_summary,
    load_graph_from_file,
)
from simulation.runner import DiffusionSimulation, SimulationLog
from utils.llm import LLMBackend, create_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intervention helpers
# ---------------------------------------------------------------------------

def _resolve_targets(
    interventions: list[Intervention],
    personas: list,
    graph: nx.DiGraph,
) -> set[str]:
    """
    Union all target agent_ids across a list of interventions.
    Handles three targeting modes: explicit IDs, epistemic_type, top-k hubs.
    """
    targeted: set[str] = set()
    for iv in interventions:
        targeted.update(iv.target_agent_ids)
        if iv.target_epistemic_types:
            targeted.update(
                p.agent_id for p in personas
                if p.epistemic_type in iv.target_epistemic_types
            )
        if iv.target_top_k_hubs > 0:
            degree = dict(graph.out_degree())
            top_k = sorted(degree, key=degree.get, reverse=True)[: iv.target_top_k_hubs]
            targeted.update(top_k)
    return targeted


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    config: ExperimentConfig
    graph_stats: dict
    graph: nx.DiGraph
    cascade_metrics: list[CascadeMetrics]
    narrative_stats: list[CascadeNarrativeStats]
    simulation_log: SimulationLog
    elapsed_seconds: float

    def summary(self) -> dict:
        """Compact summary suitable for printing / CSV export."""
        cascade_sizes = [c.size for c in self.cascade_metrics]
        cascade_depths = [c.depth for c in self.cascade_metrics]
        sv_vals = [c.structural_virality for c in self.cascade_metrics]
        adoption_rates = [c.adoption_rate for c in self.cascade_metrics]

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        unique_forwarders = [c.unique_forwarders for c in self.cascade_metrics]
        unique_receivers = [c.unique_receivers for c in self.cascade_metrics]

        return {
            "name": self.config.name,
            "network_type": self.config.network_type,
            "n_agents": self.config.n_agents,
            "agent_memory_enabled": self.config.agent_memory_enabled,
            "n_cascades": len(self.cascade_metrics),
            # mean_cascade_size counts message-tree nodes (total forwarding actions),
            # NOT unique agents — use mean_unique_receivers for agents actually reached
            "mean_cascade_size": safe_mean(cascade_sizes),
            "max_cascade_size": max(cascade_sizes) if cascade_sizes else 0,
            "mean_unique_receivers": safe_mean(unique_receivers),
            "max_unique_receivers": max(unique_receivers) if unique_receivers else 0,
            "mean_unique_forwarders": safe_mean(unique_forwarders),
            "mean_cascade_depth": safe_mean(cascade_depths),
            "max_cascade_depth": max(cascade_depths) if cascade_depths else 0,
            "mean_structural_virality": safe_mean(sv_vals),
            "mean_adoption_rate": safe_mean(adoption_rates),
            "mean_similarity_to_seed": safe_mean(
                [n.mean_similarity_to_seed for n in self.narrative_stats]
            ),
            "mean_semantic_drift_per_step": safe_mean(
                [n.semantic_drift_per_step for n in self.narrative_stats]
            ),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def save(self) -> pathlib.Path:
        """Persist full results as JSON in output_dir / name_YYYYMMDD_HHMM /."""
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        out = pathlib.Path(self.config.output_dir) / f"{self.config.name}_{ts}"
        out.mkdir(parents=True, exist_ok=True)

        # Config
        self.config.to_json(str(out / "config.json"))

        # Simulation log
        (out / "log.json").write_text(
            json.dumps(self.simulation_log.to_dict(), indent=2)
        )

        # Cascade metrics
        (out / "cascade_metrics.json").write_text(
            json.dumps([asdict(c) for c in self.cascade_metrics], indent=2)
        )

        # Narrative stats (convert dataclasses to dicts)
        narrative_dicts = []
        for ns in self.narrative_stats:
            d = {
                "cascade_id": ns.cascade_id,
                "seed_content": ns.seed_content,
                "mean_similarity_to_seed": ns.mean_similarity_to_seed,
                "min_similarity_to_seed": ns.min_similarity_to_seed,
                "mean_edit_distance": ns.mean_edit_distance,
                "semantic_drift_per_step": ns.semantic_drift_per_step,
                "records": [asdict(r) for r in ns.records],
            }
            narrative_dicts.append(d)
        (out / "narrative_stats.json").write_text(
            json.dumps(narrative_dicts, indent=2)
        )

        # Summary
        (out / "summary.json").write_text(
            json.dumps(self.summary(), indent=2)
        )

        # Visualisations
        try:
            from analysis.visualise import save_all_plots
            save_all_plots(self, out)
        except Exception as exc:
            logger.warning("Could not save plots: %s", exc)

        logger.info("Results saved to %s", out)
        return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    async def run(self) -> ExperimentResult:
        cfg = self.config
        start = time.monotonic()

        # 1. Reproducibility
        random.seed(cfg.seed)

        # 2. Build LLM backend
        llm_kwargs = {}
        if cfg.llm_model:
            llm_kwargs["model"] = cfg.llm_model
        if cfg.llm_embedding_model:
            llm_kwargs["embedding_model"] = cfg.llm_embedding_model
        llm: LLMBackend = create_backend(cfg.llm_backend, **llm_kwargs)

        logger.info("LLM backend: %s", cfg.llm_backend)

        # 3. Build network
        if cfg.network_file:
            graph = load_graph_from_file(cfg.network_file)
            n_agents = graph.number_of_nodes()
            logger.info(
                "Loaded network from %s: %d nodes, %d edges.",
                cfg.network_file, n_agents, graph.number_of_edges(),
            )
        else:
            n_agents = cfg.n_agents
            graph = build_graph(
                n_agents,
                cfg.network_type,
                seed=cfg.seed,
                **cfg.network_params,
            )
        personas = build_agent_personas(
            n_agents,
            seed=cfg.seed,
            persona_mix=cfg.persona_mix,
        )
        agent_ids = [p.agent_id for p in personas]
        graph = assign_agents_to_graph(graph, agent_ids)

        # 3b. Apply block interventions — remove agents from graph before simulation
        block_interventions = [iv for iv in cfg.interventions if iv.type == "block"]
        if block_interventions:
            blocked = _resolve_targets(block_interventions, personas, graph)
            if blocked:
                graph = graph.copy()
                graph.remove_nodes_from(blocked)
                personas = [p for p in personas if p.agent_id not in blocked]
                logger.info("Blocked %d agents: %s", len(blocked), blocked)

        gstats = graph_summary(graph)
        logger.info("Graph: %s", gstats)

        # 4. Build agents
        agents = {
            p.agent_id: DiffusionAgent(
                p, llm,
                prompt_template=cfg.prompt_template,
                memory_enabled=cfg.agent_memory_enabled,
            )
            for p in personas
        }

        # 5. Build simulation (pass label + correct interventions for runtime use)
        sim = DiffusionSimulation(
            agents=agents,
            graph=graph,
            max_steps=cfg.max_steps,
            interventions=[iv for iv in cfg.interventions if iv.type != "block"],
            max_concurrent_llm=cfg.max_concurrent_llm,
        )

        # 6. Seed messages
        for sm in cfg.seed_messages:
            sim.seed(
                origin_agent_id=sm.origin_agent_id,
                content=sm.content,
                step=sm.step,
                label=sm.label,
                item_id=sm.item_id,
            )

        # 7. Run
        log = await sim.run()
        logger.info(
            "Simulation complete: %d messages, %d events.",
            len(log.messages),
            len(log.events),
        )

        # 8. Cascade metrics
        cascade_metrics = compute_all_cascades(log)

        # 9. Narrative metrics
        narrative_stats: list[CascadeNarrativeStats] = []
        if cfg.compute_narrative_metrics and log.messages:
            tracker = NarrativeTracker(llm)
            narrative_stats = await tracker.analyze_all(log)

        elapsed = time.monotonic() - start
        logger.info("Experiment %r finished in %.1fs.", cfg.name, elapsed)

        result = ExperimentResult(
            config=cfg,
            graph_stats=gstats,
            graph=graph,
            cascade_metrics=cascade_metrics,
            narrative_stats=narrative_stats,
            simulation_log=log,
            elapsed_seconds=elapsed,
        )
        result.save()
        return result
