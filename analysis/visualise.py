"""
Visualisation helpers for diffusion simulation results.

All functions return matplotlib Figure objects so they can be shown
interactively or saved to disk. ``save_all_plots`` and
``save_dataset_plots`` are the main entry points called automatically
by ``ExperimentResult.save()`` and ``DatasetExperimentResult.save()``.

Available plots
---------------
plot_cascade_tree          – directed cascade as a layered tree
plot_diffusion_curve       – cumulative reach per step (one line per cascade)
plot_narrative_similarity  – similarity-to-seed vs step for each cascade
plot_comparison_by_label   – grouped bars comparing metrics across content labels
save_all_plots             – save all relevant plots for an ExperimentResult
save_dataset_plots         – save comparison plots for a DatasetExperimentResult
"""

from __future__ import annotations

import logging
import pathlib
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on servers/headless systems
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from experiments.runner import ExperimentResult
    from experiments.dataset_experiment import DatasetExperimentResult
    from metrics.cascade import CascadeMetrics
    from metrics.narrative import CascadeNarrativeStats
    from simulation.runner import SimulationLog

logger = logging.getLogger(__name__)

# Consistent colour palette for content labels
_LABEL_COLOURS = {
    "true":       "#2ecc71",   # green
    "fake":       "#e74c3c",   # red
    "misleading": "#e67e22",   # orange
    "":           "#95a5a6",   # grey (unknown)
}

def _label_colour(label: str) -> str:
    return _LABEL_COLOURS.get(label, "#3498db")


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------

def _hierarchical_layout(g: nx.DiGraph, root=None) -> dict:
    """
    Top-down hierarchical layout for a (possibly disconnected) directed graph.
    Nodes at the same BFS depth share the same y level.
    """
    if g.number_of_nodes() == 0:
        return {}

    # Find roots (nodes with in-degree 0)
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    if not roots:
        roots = [list(g.nodes)[0]]
    if root is not None and root in g.nodes:
        roots = [root] + [r for r in roots if r != root]

    level_groups: dict[int, list] = defaultdict(list)
    visited: set = set()

    def _bfs(start, offset_level: int = 0):
        queue = [(start, offset_level)]
        while queue:
            node, lvl = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            level_groups[lvl].append(node)
            for child in g.successors(node):
                if child not in visited:
                    queue.append((child, lvl + 1))

    for r in roots:
        _bfs(r)

    # Any remaining nodes (disconnected)
    max_lvl = max(level_groups.keys(), default=0)
    for node in g.nodes:
        if node not in visited:
            max_lvl += 1
            level_groups[max_lvl].append(node)

    pos = {}
    for lvl, nodes in level_groups.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i + 1) / (n + 1), -lvl)
    return pos


# ---------------------------------------------------------------------------
# 1. Cascade tree
# ---------------------------------------------------------------------------

def plot_cascade_tree(
    log: SimulationLog,
    cascade_id: str,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Draw the diffusion tree for one cascade.

    Nodes represent agents; edges show forwarding direction.
    Node colour encodes simulation step (depth in the tree).
    The origin/seed node is drawn as a diamond.
    """
    msgs = log.cascade_messages(cascade_id)
    if not msgs:
        fig, ax_ = plt.subplots(figsize=(4, 3))
        ax_.text(0.5, 0.5, "No messages", ha="center", va="center")
        ax_.axis("off")
        return fig

    # Build agent-level forwarding graph
    g = nx.DiGraph()
    node_step: dict[str, int] = {}
    origin = msgs[0].origin_agent_id

    for msg in msgs:
        s, r = msg.sender_agent_id, msg.receiver_agent_id
        if not g.has_edge(s, r):
            g.add_edge(s, r)
        # Step of a node = the step at which they first received a message
        if r not in node_step:
            node_step[r] = msg.step
        if s not in node_step:
            node_step[s] = max(msg.step - 1, 0)

    pos = _hierarchical_layout(g, root=origin)

    steps = [node_step.get(n, 0) for n in g.nodes]
    max_step = max(steps, default=1) or 1
    colours = [cm.viridis(s / max_step) for s in steps]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(6, g.number_of_nodes() * 0.4), 5))
    else:
        fig = ax.figure

    # Separate origin from others for marker shape
    non_origin = [n for n in g.nodes if n != origin]
    origin_nodes = [origin] if origin in g.nodes else []

    origin_colours = [colours[list(g.nodes).index(n)] for n in origin_nodes]
    non_origin_colours = [colours[list(g.nodes).index(n)] for n in non_origin]

    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True,
                           arrowstyle="->", arrowsize=15,
                           edge_color="#aaaaaa", width=1.2)
    if non_origin:
        nx.draw_networkx_nodes(g, pos, nodelist=non_origin, node_color=non_origin_colours,
                               node_size=400, ax=ax)
    if origin_nodes:
        nx.draw_networkx_nodes(g, pos, nodelist=origin_nodes, node_color=origin_colours,
                               node_size=600, node_shape="D", ax=ax)

    labels = {n: n.replace("agent_", "") for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)

    label_str = next((m.label for m in msgs if m.label), "")
    ax.set_title(title or f"Cascade {cascade_id[:8]}  [{label_str}]", fontsize=10)
    ax.axis("off")

    sm = cm.ScalarMappable(cmap=cm.viridis,
                           norm=plt.Normalize(vmin=0, vmax=max_step))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Step", shrink=0.6)

    if own_fig:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Diffusion curve
# ---------------------------------------------------------------------------

def plot_diffusion_curve(
    log: SimulationLog,
    cascade_metrics: list[CascadeMetrics] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Cumulative number of unique agents reached per simulation step,
    one line per cascade.
    """
    cascade_ids = list({m.cascade_id for m in log.messages})

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    # Build label map
    label_map: dict[str, str] = {}
    if cascade_metrics:
        label_map = {c.cascade_id: c.label for c in cascade_metrics}

    for cid in cascade_ids:
        msgs = log.cascade_messages(cid)
        if not msgs:
            continue

        # Unique agents reached per step
        agents_by_step: dict[int, set] = defaultdict(set)
        for m in msgs:
            agents_by_step[m.step].add(m.receiver_agent_id)

        all_steps = sorted(agents_by_step.keys())
        cumulative, seen = [], set()
        for step in range(all_steps[-1] + 1):
            seen |= agents_by_step.get(step, set())
            cumulative.append((step, len(seen)))

        xs = [p[0] for p in cumulative]
        ys = [p[1] for p in cumulative]
        lbl = label_map.get(cid, "")
        colour = _label_colour(lbl)
        ax.plot(xs, ys, marker="o", color=colour, linewidth=1.8,
                label=f"{lbl or 'cascade'} {cid[:6]}", alpha=0.8)

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Cumulative agents reached")
    ax.set_title("Diffusion curves per cascade")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    if own_fig:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Narrative similarity
# ---------------------------------------------------------------------------

def plot_narrative_similarity(
    narrative_stats: list[CascadeNarrativeStats],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Scatter + trend line of cosine similarity-to-seed vs simulation step,
    one series per cascade.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    if not narrative_stats:
        ax.text(0.5, 0.5, "No narrative data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        if own_fig:
            fig.tight_layout()
        return fig

    colours = plt.cm.tab10(np.linspace(0, 1, len(narrative_stats)))

    for ns, colour in zip(narrative_stats, colours):
        steps = [r.step for r in ns.records if r.step > 0]
        sims  = [r.similarity_to_seed for r in ns.records if r.step > 0]
        if not steps:
            continue
        ax.scatter(steps, sims, color=colour, alpha=0.6, s=30)
        # Trend line
        if len(steps) >= 2:
            z = np.polyfit(steps, sims, 1)
            xs = np.linspace(min(steps), max(steps), 50)
            ax.plot(xs, np.polyval(z, xs), color=colour, linewidth=1.5,
                    label=f"{ns.cascade_id[:6]} (drift={ns.semantic_drift_per_step:.3f})")
        else:
            ax.plot(steps, sims, color=colour, linewidth=1.5,
                    label=ns.cascade_id[:6])

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Cosine similarity to seed")
    ax.set_title("Narrative drift: similarity to seed over time")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    if own_fig:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Comparison by label
# ---------------------------------------------------------------------------

def plot_comparison_by_label(
    comparison: dict,
    metrics: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing diffusion metrics across content labels
    (true / fake / misleading).

    Parameters
    ----------
    comparison : output of ``compare_by_label()``
    metrics    : which metrics to display (default: size, adoption_rate, structural_virality)
    """
    if metrics is None:
        metrics = ["mean_size", "mean_adoption_rate", "mean_structural_virality"]

    labels = list(comparison.keys())
    if not labels:
        fig, ax_ = plt.subplots(figsize=(6, 4))
        ax_.text(0.5, 0.5, "No comparison data", ha="center", va="center")
        ax_.axis("off")
        return fig

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 4))
    else:
        fig = ax.figure

    n_metrics = len(metrics)
    n_labels = len(labels)
    bar_width = 0.7 / n_metrics
    x = np.arange(n_labels)

    for i, metric in enumerate(metrics):
        values = [comparison[lbl].get(metric, 0.0) for lbl in labels]
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=metric.replace("mean_", ""),
                      color=plt.cm.Set2(i / max(n_metrics - 1, 1)),
                      edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl or "unknown" for lbl in labels], fontsize=10)
    # Colour x-tick labels by content type
    for tick, lbl in zip(ax.get_xticklabels(), labels):
        tick.set_color(_label_colour(lbl))

    ax.set_title("Diffusion metrics by content label")
    ax.set_ylabel("Mean value")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    if own_fig:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_all_plots(result: ExperimentResult, output_dir: pathlib.Path) -> None:
    """
    Generate and save all relevant plots for a single ExperimentResult.
    Called automatically by ``ExperimentResult.save()``.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log = result.simulation_log
    cascade_metrics = result.cascade_metrics

    # 1. One cascade tree per cascade (skip if too many)
    cascade_ids = list({m.cascade_id for m in log.messages})
    for cid in cascade_ids[:5]:   # cap at 5 to avoid thousands of files
        fig = plot_cascade_tree(log, cid)
        fig.savefig(out / f"cascade_tree_{cid[:8]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2. Diffusion curve
    if log.messages:
        fig = plot_diffusion_curve(log, cascade_metrics)
        fig.savefig(out / "diffusion_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Narrative similarity
    if result.narrative_stats:
        fig = plot_narrative_similarity(result.narrative_stats)
        fig.savefig(out / "narrative_similarity.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Plots saved to %s", out)


def save_dataset_plots(
    result: DatasetExperimentResult,
    output_dir: pathlib.Path,
) -> None:
    """
    Generate and save comparison plots for a DatasetExperimentResult.
    Called automatically by ``DatasetExperimentResult.save()``.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Comparison by label (main thesis figure)
    if result.comparison:
        fig = plot_comparison_by_label(result.comparison)
        fig.savefig(out / "comparison_by_label.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Aggregate narrative similarity across all items
    all_narrative = [ns for ir in result.item_results for ns in ir.narrative_stats]
    if all_narrative:
        fig = plot_narrative_similarity(all_narrative)
        fig.savefig(out / "narrative_similarity_all.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info("Dataset plots saved to %s", out)
