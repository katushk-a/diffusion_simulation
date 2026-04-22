"""
Comparison plots for the topology × content 3×3 factorial experiment.

Loads summary.json from every topo_x_content_* result directory and
produces four publication-ready figures saved to
results/topo_x_content_comparison/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "topo_x_content_comparison"

NETWORKS = ["random", "scale_free", "small_world"]
LABELS   = ["true_news", "fake_news", "misleading"]

NETWORK_DISPLAY = {"random": "Random", "scale_free": "Scale-free", "small_world": "Small-world"}
LABEL_DISPLAY   = {"true_news": "True news", "fake_news": "Fake news", "misleading": "Misleading"}

LABEL_COLOURS = {
    "true_news":  "#2ecc71",
    "fake_news":  "#e74c3c",
    "misleading": "#e67e22",
}
NETWORK_HATCHES = {"random": "", "scale_free": "//", "small_world": "xx"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> dict[tuple[str, str], dict]:
    """Return {(network, label): summary_dict} for all 9 conditions."""
    data: dict[tuple[str, str], dict] = {}
    for net in NETWORKS:
        for lbl in LABELS:
            # Match timestamped directory
            pattern = f"topo_x_content_{net}_{lbl}_*"
            matches = sorted(RESULTS_DIR.glob(pattern))
            if not matches:
                print(f"  WARNING: no results found for {net}/{lbl}", file=sys.stderr)
                continue
            summary_path = matches[-1] / "summary.json"
            with open(summary_path) as f:
                data[(net, lbl)] = json.load(f)
    return data


def get_metric(data, net, lbl, key, default=0.0):
    return data.get((net, lbl), {}).get(key, default)


# ---------------------------------------------------------------------------
# Plot 1: Grouped bar — cascade size by network, grouped by content
# ---------------------------------------------------------------------------

def plot_cascade_size(data: dict, out: pathlib.Path) -> None:
    metric = "mean_cascade_size"
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(NETWORKS))
    bar_width = 0.22
    offsets = [-bar_width, 0, bar_width]

    for i, lbl in enumerate(LABELS):
        values = [get_metric(data, net, lbl, metric) for net in NETWORKS]
        bars = ax.bar(x + offsets[i], values, bar_width,
                      label=LABEL_DISPLAY[lbl],
                      color=LABEL_COLOURS[lbl],
                      edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([NETWORK_DISPLAY[n] for n in NETWORKS], fontsize=11)
    ax.set_ylabel("Mean cascade size (agents reached)", fontsize=10)
    ax.set_title("Cascade reach by network topology & content type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    fig.savefig(out / "01_cascade_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 01_cascade_size.png")


# ---------------------------------------------------------------------------
# Plot 2: Grouped bar — semantic drift per step
# ---------------------------------------------------------------------------

def plot_semantic_drift(data: dict, out: pathlib.Path) -> None:
    metric = "mean_semantic_drift_per_step"
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(NETWORKS))
    bar_width = 0.22
    offsets = [-bar_width, 0, bar_width]

    for i, lbl in enumerate(LABELS):
        values = [get_metric(data, net, lbl, metric) for net in NETWORKS]
        bars = ax.bar(x + offsets[i], values, bar_width,
                      label=LABEL_DISPLAY[lbl],
                      color=LABEL_COLOURS[lbl],
                      edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([NETWORK_DISPLAY[n] for n in NETWORKS], fontsize=11)
    ax.set_ylabel("Mean semantic drift per step (Δ cosine similarity)", fontsize=10)
    ax.set_title("Narrative drift rate by network topology & content type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02_semantic_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 02_semantic_drift.png")


# ---------------------------------------------------------------------------
# Plot 3: Heatmap grid — 4 metrics as 3×3 heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(data: dict, out: pathlib.Path) -> None:
    metrics = [
        ("mean_cascade_size",          "Cascade size"),
        ("mean_cascade_depth",         "Cascade depth"),
        ("mean_adoption_rate",         "Adoption rate"),
        ("mean_semantic_drift_per_step", "Drift/step"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("3×3 Factorial: Topology × Content — Key Metrics", fontsize=13, fontweight="bold", y=1.01)

    for ax, (metric_key, metric_label) in zip(axes.flat, metrics):
        grid = np.array([
            [get_metric(data, net, lbl, metric_key) for lbl in LABELS]
            for net in NETWORKS
        ])

        # Diverging colormap for drift (can go negative), sequential for others
        cmap = "RdYlGn_r" if "drift" in metric_key else "YlOrRd"
        if "drift" in metric_key:
            vmax = max(abs(grid.min()), abs(grid.max())) or 0.01
            vmin, vmax = -vmax, vmax
        else:
            vmin, vmax = grid.min(), grid.max()

        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABELS], fontsize=9)
        ax.set_yticks(range(len(NETWORKS)))
        ax.set_yticklabels([NETWORK_DISPLAY[n] for n in NETWORKS], fontsize=9)
        ax.set_title(metric_label, fontsize=10, fontweight="bold")

        # Annotate cells
        for r in range(len(NETWORKS)):
            for c in range(len(LABELS)):
                val = grid[r, c]
                text = f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
                brightness = (grid[r, c] - vmin) / (vmax - vmin + 1e-9)
                txt_colour = "white" if (brightness < 0.35 or brightness > 0.75) else "black"
                ax.text(c, r, text, ha="center", va="center",
                        fontsize=9, color=txt_colour, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.tight_layout()
    fig.savefig(out / "03_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 03_heatmaps.png")


# ---------------------------------------------------------------------------
# Plot 4: Similarity to seed — line chart per topology
# ---------------------------------------------------------------------------

def plot_similarity_bars(data: dict, out: pathlib.Path) -> None:
    metric = "mean_similarity_to_seed"
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(NETWORKS))
    bar_width = 0.22
    offsets = [-bar_width, 0, bar_width]

    for i, lbl in enumerate(LABELS):
        values = [get_metric(data, net, lbl, metric) for net in NETWORKS]
        bars = ax.bar(x + offsets[i], values, bar_width,
                      label=LABEL_DISPLAY[lbl],
                      color=LABEL_COLOURS[lbl],
                      edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([NETWORK_DISPLAY[n] for n in NETWORKS], fontsize=11)
    ax.set_ylabel("Mean cosine similarity to seed message", fontsize=10)
    ax.set_title("Message fidelity by network topology & content type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5, label="perfect fidelity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04_similarity_to_seed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 04_similarity_to_seed.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading results from {RESULTS_DIR} ...")
    data = load_results()
    print(f"Loaded {len(data)} conditions. Generating plots → {OUT_DIR}\n")

    plot_cascade_size(data, OUT_DIR)
    plot_semantic_drift(data, OUT_DIR)
    plot_heatmaps(data, OUT_DIR)
    plot_similarity_bars(data, OUT_DIR)

    print(f"\nAll plots saved to {OUT_DIR}")
