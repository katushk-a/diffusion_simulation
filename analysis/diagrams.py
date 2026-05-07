"""
Thesis diagrams — simulation architecture and flow.

Generates publication-ready figures saved to results/diagrams/:
  architecture.png   – layered component diagram (what the system is)
  flow.png           – step-by-step diffusion flowchart (what happens)
  combined.png       – both panels side by side
  integrated.png     – flow traced through the architecture (single diagram)
"""

from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
C = {
    "input":    "#2c3e50",   # dark navy
    "agent":    "#2980b9",   # blue
    "llm":      "#8e44ad",   # purple
    "sim":      "#16a085",   # teal
    "output":   "#27ae60",   # green
    "decision": "#e67e22",   # orange
    "drop":     "#e74c3c",   # red
    "forward":  "#27ae60",   # green
    "bg":       "#f8f9fa",   # near-white
    "arrow":    "#555555",
    "text_light": "#ffffff",
    "text_dark":  "#2c3e50",
}

def _box(ax, x, y, w, h, label, sublabel=None,
         facecolor="#2980b9", textcolor="white",
         fontsize=9, radius=0.04, zorder=3):
    """Draw a rounded rectangle with label (and optional sublabel)."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor="white",
        linewidth=1.5, zorder=zorder,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + h * 0.12, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor, zorder=zorder+1)
        ax.text(x, y - h * 0.2, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color=textcolor, alpha=0.85, zorder=zorder+1,
                style="italic")
    else:
        ax.text(x, y, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor, zorder=zorder+1,
                multialignment="center")

def _arrow(ax, x0, y0, x1, y1, label=None, color="#555555",
           lw=1.5, style="-|>", zorder=2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color, lw=lw,
                    connectionstyle="arc3,rad=0.0",
                ), zorder=zorder)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.02, my, label, fontsize=7.5, color=color,
                ha="left", va="center", zorder=zorder+1,
                fontweight="bold")

def _section_bg(ax, x, y, w, h, color, alpha=0.07, label=None, zorder=1):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=color, edgecolor=color,
        linewidth=0, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + 0.015, y + h - 0.02, label,
                fontsize=7, color=color, alpha=0.6,
                fontweight="bold", va="top", zorder=zorder+1,
                style="italic")


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

def draw_architecture() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle("LLM-Backed Information Diffusion Simulation — System Architecture",
                 fontsize=13, fontweight="bold", color=C["text_dark"], y=0.97)

    # ── Section backgrounds ────────────────────────────────────────────────
    _section_bg(ax, 0.01, 0.60, 0.25, 0.34, C["input"],   label="INPUT LAYER")
    _section_bg(ax, 0.28, 0.10, 0.44, 0.84, C["sim"],     label="SIMULATION ENGINE")
    _section_bg(ax, 0.74, 0.10, 0.25, 0.84, C["output"],  label="OUTPUT LAYER")
    _section_bg(ax, 0.01, 0.05, 0.25, 0.52, C["llm"],     label="LLM BACKENDS")

    # ── INPUT: Network ─────────────────────────────────────────────────────
    _box(ax, 0.13, 0.85, 0.20, 0.09, "Network Graph",
         "Random · Scale-free\nSmall-world · Community",
         facecolor=C["input"])

    # ── INPUT: Agents ──────────────────────────────────────────────────────
    _box(ax, 0.13, 0.73, 0.20, 0.09, "Agent Personas",
         "Open · Closed\nCredulous · Strategic",
         facecolor=C["input"])

    # ── INPUT: Seed message ────────────────────────────────────────────────
    _box(ax, 0.13, 0.61, 0.20, 0.07, "Seed Message",
         "True / Fake / Misleading",
         facecolor=C["input"])

    # ── LLM: backends ──────────────────────────────────────────────────────
    _box(ax, 0.08, 0.44, 0.10, 0.08, "Ollama\nllama3.1:8b",  facecolor=C["llm"])
    _box(ax, 0.19, 0.44, 0.10, 0.08, "OpenAI\ngpt-4o-mini",  facecolor=C["llm"])
    _box(ax, 0.08, 0.33, 0.10, 0.08, "Mock\n(testing)",      facecolor="#7f8c8d")
    _box(ax, 0.19, 0.33, 0.10, 0.08, "Embeddings\nnomic-embed", facecolor=C["llm"])

    # ── SIMULATION: core boxes ─────────────────────────────────────────────
    _box(ax, 0.50, 0.88, 0.38, 0.08, "DiffusionSimulation",
         "asyncio · LLM semaphore · step queue",
         facecolor=C["sim"], fontsize=10)

    _box(ax, 0.38, 0.73, 0.14, 0.08, "Step Loop\n(t = 0 … T)",
         facecolor=C["sim"])

    _box(ax, 0.55, 0.73, 0.16, 0.08, "Message Queue\nper step",
         facecolor=C["sim"])

    _box(ax, 0.72, 0.73, 0.14, 0.08, "Intervention\nEngine",
         facecolor="#c0392b")

    # Agent decision block
    _box(ax, 0.50, 0.54, 0.38, 0.10,
         "Agent Decision (per message)",
         "Persona  ·  Memory  ·  Incoming content  →  LLM  →  ForwardDecision",
         facecolor=C["agent"], fontsize=9)

    _box(ax, 0.38, 0.38, 0.14, 0.08, "Forward\n+ Rewrite",
         facecolor=C["forward"])
    _box(ax, 0.55, 0.38, 0.14, 0.08, "Drop\n(logged)",
         facecolor=C["drop"])
    _box(ax, 0.72, 0.38, 0.14, 0.08, "Agent\nMemory",
         facecolor=C["agent"])

    _box(ax, 0.50, 0.22, 0.38, 0.08,
         "Fan out to neighbours  →  Enqueue for step t+1",
         facecolor=C["sim"])

    _box(ax, 0.50, 0.12, 0.38, 0.07,
         "Cascade ends  (no messages OR t = T_max)",
         facecolor="#7f8c8d")

    # ── OUTPUT ─────────────────────────────────────────────────────────────
    _box(ax, 0.865, 0.84, 0.20, 0.08, "summary.json",   facecolor=C["output"])
    _box(ax, 0.865, 0.73, 0.20, 0.08, "cascade_metrics\n.json", facecolor=C["output"])
    _box(ax, 0.865, 0.62, 0.20, 0.08, "narrative_stats\n.json", facecolor=C["output"])
    _box(ax, 0.865, 0.51, 0.20, 0.08, "log.json\n(all events)", facecolor=C["output"])
    _box(ax, 0.865, 0.38, 0.20, 0.14,
         "Plots\ncascade tree\ndiffusion curve\nnarrative drift",
         facecolor=C["output"])

    # ── ARROWS: inputs → engine ────────────────────────────────────────────
    _arrow(ax, 0.23, 0.85, 0.31, 0.85)
    _arrow(ax, 0.23, 0.73, 0.31, 0.73)
    _arrow(ax, 0.23, 0.61, 0.31, 0.61)

    # engine internals
    _arrow(ax, 0.50, 0.84, 0.50, 0.77)   # simulation → step loop area
    _arrow(ax, 0.38, 0.69, 0.38, 0.59)   # step loop → agent decision
    _arrow(ax, 0.55, 0.69, 0.55, 0.59)
    _arrow(ax, 0.72, 0.69, 0.72, 0.59)

    _arrow(ax, 0.42, 0.49, 0.38, 0.42)   # decision → forward
    _arrow(ax, 0.55, 0.49, 0.55, 0.42)   # decision → drop
    _arrow(ax, 0.68, 0.49, 0.72, 0.42)   # decision → memory

    _arrow(ax, 0.38, 0.34, 0.38, 0.26)   # forward → fan out
    _arrow(ax, 0.50, 0.18, 0.50, 0.155)  # fan out → end

    # LLM → agent decision
    _arrow(ax, 0.23, 0.42, 0.31, 0.54, color=C["llm"])
    _arrow(ax, 0.23, 0.31, 0.31, 0.50, color=C["llm"])

    # engine → outputs
    _arrow(ax, 0.69, 0.88, 0.765, 0.84)
    _arrow(ax, 0.69, 0.73, 0.765, 0.73)
    _arrow(ax, 0.69, 0.54, 0.765, 0.62)
    _arrow(ax, 0.69, 0.22, 0.765, 0.38)
    _arrow(ax, 0.69, 0.12, 0.765, 0.51)

    # memory feedback loop
    ax.annotate("", xy=(0.72, 0.59), xytext=(0.72, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=C["agent"], lw=1.5,
                                connectionstyle="arc3,rad=0.4"))
    ax.text(0.80, 0.51, "persists\nper agent", fontsize=7, color=C["agent"],
            ha="center", va="center", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — SIMULATION FLOW
# ═══════════════════════════════════════════════════════════════════════════

def draw_flow() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 14))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle("Simulation Step-by-Step Flow",
                 fontsize=13, fontweight="bold", color=C["text_dark"], y=0.985)

    BW, BH = 0.46, 0.055   # standard box width / height
    CX = 0.50              # centre x

    def y(step): return 0.96 - step * 0.082

    # ── Nodes ──────────────────────────────────────────────────────────────
    nodes = [
        # (step_idx, label, sublabel, color, shape)
        (0,  "START",                    None,                          C["input"],    "start"),
        (1,  "Inject Seed Message",      "origin agent → content + label", C["input"], "box"),
        (2,  "Fan out to all neighbours","one message copy per neighbour", C["sim"],   "box"),
        (3,  "Enqueue for Step t = 0",   None,                          C["sim"],      "box"),
        (4,  "┌─ STEP LOOP (t = 0…T) ─┐", None,                        C["sim"],     "section"),
        (5,  "Pull messages for step t", None,                          C["sim"],      "box"),
        (6,  "Any messages?",            None,                          C["decision"], "diamond"),
        (7,  "For each message:",        "parallel, semaphore-limited",  C["sim"],     "box"),
        (8,  "Agent already forwarded\nthis cascade?", None,            C["decision"], "diamond"),
        (9,  "Build LLM prompt",         "persona  +  memory  +  content", C["agent"],"box"),
        (10, "LLM call → ForwardDecision", "forward · reasoning · rewrite", C["llm"], "box"),
        (11, "Decision: forward?",       None,                          C["decision"], "diamond"),
        (12, "Rewrite in agent's voice", None,                          C["forward"],  "box"),
        (13, "Fan out to neighbours",    "one child message per neighbour", C["sim"],  "box"),
        (14, "Enqueue for step t + 1",   None,                          C["sim"],      "box"),
        (15, "Log FORWARDED event\nUpdate agent memory", None,          C["agent"],    "box"),
        (16, "t = t + 1",               None,                           C["sim"],      "box"),
        (17, "POST-SIMULATION",          None,                          "#7f8c8d",     "section"),
        (18, "Compute cascade metrics",  "size · depth · virality · adoption", C["output"], "box"),
        (19, "Embed all messages",       "nomic-embed-text (cosine similarity)", C["output"], "box"),
        (20, "Compute narrative drift",  "similarity to seed · drift/step", C["output"], "box"),
        (21, "Save results + plots",     "summary.json · cascade_metrics · log", C["output"], "box"),
        (22, "END",                      None,                          C["output"],   "start"),
    ]

    positions = {}
    for step_idx, label, sublabel, color, shape in nodes:
        yi = y(step_idx)
        positions[step_idx] = (CX, yi)

        if shape == "start":
            circle = plt.Circle((CX, yi), 0.035, color=color, zorder=3)
            ax.add_patch(circle)
            ax.text(CX, yi, label, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white", zorder=4)

        elif shape == "diamond":
            dx, dy = 0.13, 0.032
            diamond = plt.Polygon(
                [[CX, yi+dy], [CX+dx, yi], [CX, yi-dy], [CX-dx, yi]],
                closed=True, facecolor=color, edgecolor="white", lw=1.5, zorder=3
            )
            ax.add_patch(diamond)
            ax.text(CX, yi, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white", zorder=4,
                    multialignment="center")

        elif shape == "section":
            rect = FancyBboxPatch((CX - BW/2 - 0.01, yi - BH/2 - 0.005),
                                   BW + 0.02, BH + 0.01,
                                   boxstyle="round,pad=0,rounding_size=0.03",
                                   facecolor=color, edgecolor=color,
                                   linewidth=0, alpha=0.15, zorder=2)
            ax.add_patch(rect)
            ax.text(CX, yi, label, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color, zorder=4,
                    alpha=0.8)

        else:  # box
            _box(ax, CX, yi, BW, BH, label, sublabel, facecolor=color, fontsize=8.5)

    # ── Main vertical arrows ───────────────────────────────────────────────
    straight = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    for i in straight:
        if i+1 > 21: break
        y0 = y(i) - (0.035 if i in (0,22) else BH/2)
        y1 = y(i+1) + (0.035 if i+1 in (0,22) else BH/2)
        ax.annotate("", xy=(CX, y1), xytext=(CX, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                   lw=1.5), zorder=2)

    # ── Drop branch from diamond 8 (already forwarded?) ───────────────────
    # "YES" → drop, exits right
    yd8 = y(8)
    ax.annotate("", xy=(0.88, yd8), xytext=(CX+0.13, yd8),
                arrowprops=dict(arrowstyle="-|>", color=C["drop"], lw=1.5), zorder=2)
    _box(ax, 0.88, yd8, 0.18, BH, "LOG DROPPED\n(skip)", facecolor=C["drop"], fontsize=8)
    ax.text(CX+0.14, yd8+0.018, "YES", fontsize=7.5,
            color=C["drop"], fontweight="bold")
    ax.text(CX+0.02, yd8-0.032, "NO", fontsize=7.5,
            color=C["sim"], fontweight="bold")

    # ── Drop branch from diamond 11 (forward?) ────────────────────────────
    yd11 = y(11)
    ax.annotate("", xy=(0.88, yd11), xytext=(CX+0.13, yd11),
                arrowprops=dict(arrowstyle="-|>", color=C["drop"], lw=1.5), zorder=2)
    _box(ax, 0.88, yd11, 0.18, BH, "LOG DROPPED\n+ update memory", facecolor=C["drop"], fontsize=8)
    ax.text(CX+0.14, yd11+0.018, "NO", fontsize=7.5,
            color=C["drop"], fontweight="bold")
    ax.text(CX+0.02, yd11-0.032, "YES", fontsize=7.5,
            color=C["forward"], fontweight="bold")

    # ── No messages branch from diamond 6 ─────────────────────────────────
    yd6 = y(6)
    ypost = y(17) + BH/2
    ax.annotate("", xy=(0.12, ypost), xytext=(0.12, yd6),
                arrowprops=dict(arrowstyle="-|>", color=C["drop"], lw=1.5,
                                connectionstyle="arc3,rad=0.0"), zorder=2)
    ax.plot([CX - 0.13, 0.12], [yd6, yd6], color=C["drop"], lw=1.5, zorder=2)
    ax.plot([0.12, 0.12], [yd6, ypost], color=C["drop"], lw=1.5, zorder=2)
    ax.text(0.05, yd6 + 0.012, "NO\n(end cascade)", fontsize=7.5,
            color=C["drop"], fontweight="bold", ha="center")
    ax.text(CX + 0.02, yd6 - 0.028, "YES", fontsize=7.5,
            color=C["sim"], fontweight="bold")

    # ── Loop back arrow (step t+1) ─────────────────────────────────────────
    y16 = y(16) - BH/2
    y5  = y(5)  + BH/2
    ax.plot([CX + BW/2, 0.96, 0.96, CX + BW/2],
            [y16, y16, y5, y5],
            color=C["sim"], lw=1.5, linestyle="--", zorder=2)
    ax.annotate("", xy=(CX + BW/2, y5), xytext=(CX + BW/2 + 0.001, y5),
                arrowprops=dict(arrowstyle="-|>", color=C["sim"], lw=1.5))
    ax.text(0.97, (y16+y5)/2, "loop\nback", fontsize=7.5,
            color=C["sim"], ha="left", va="center",
            fontweight="bold", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — INTEGRATED (flow traced through architecture)
# ═══════════════════════════════════════════════════════════════════════════

def _step_badge(ax, x, y, n, color="#2c3e50", r=0.022):
    """Draw a numbered circle badge for flow steps."""
    circle = plt.Circle((x, y), r, color=color, zorder=6)
    ax.add_patch(circle)
    ax.text(x, y, str(n), ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white", zorder=7)

def _dashed_region(ax, x0, y0, x1, y1, color, label=None):
    rect = mpatches.FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=color, edgecolor=color,
        linewidth=1.5, linestyle="--",
        alpha=0.08, zorder=1,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x0 + 0.01, y1 - 0.01, label, fontsize=7,
                color=color, alpha=0.7, fontweight="bold",
                va="top", style="italic", zorder=2)

def draw_integrated() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle(
        "LLM-Backed Information Diffusion — Architecture & Simulation Flow",
        fontsize=14, fontweight="bold", color=C["text_dark"], y=0.98,
    )

    # Region backgrounds
    _dashed_region(ax, 0.01, 0.02, 0.20, 0.97, C["input"],  "INPUTS")
    _dashed_region(ax, 0.22, 0.02, 0.78, 0.97, C["sim"],    "SIMULATION ENGINE")
    _dashed_region(ax, 0.80, 0.02, 0.99, 0.97, C["output"], "OUTPUTS")

    # ── INPUTS ────────────────────────────────────────────────────────────
    _box(ax, 0.105, 0.855, 0.17, 0.075, "Network Graph",
         "Random · Scale-free\nSmall-world · Community",
         facecolor=C["input"], fontsize=8.5)
    _box(ax, 0.105, 0.740, 0.17, 0.075, "Agent Personas",
         "Open · Closed\nCredulous · Strategic",
         facecolor=C["input"], fontsize=8.5)
    _box(ax, 0.105, 0.625, 0.17, 0.075, "Seed Message",
         "True / Fake / Misleading",
         facecolor=C["input"], fontsize=8.5)

    # ── SIMULATION ENGINE ─────────────────────────────────────────────────
    CX = 0.50
    dx, dy = 0.09, 0.030

    _box(ax, CX, 0.885, 0.30, 0.065, "① Inject Seed Message",
         "origin agent fans out to all neighbours",
         facecolor=C["input"], fontsize=9)
    _box(ax, CX, 0.800, 0.30, 0.060, "② Message Queue  (step t = 0)",
         "one message copy per neighbour",
         facecolor=C["sim"], fontsize=9)

    _dashed_region(ax, 0.232, 0.12, 0.768, 0.755, C["sim"], "STEP LOOP  t = 0 ... T_max")

    _box(ax, CX, 0.715, 0.30, 0.055, "③ Pull messages for step t",
         None, facecolor=C["sim"], fontsize=9)

    cx4, cy4 = CX, 0.645
    ax.add_patch(plt.Polygon([[cx4,cy4+dy],[cx4+dx,cy4],[cx4,cy4-dy],[cx4-dx,cy4]],
        closed=True, facecolor=C["decision"], edgecolor="white", lw=1.5, zorder=3))
    ax.text(cx4, cy4, "④ Any messages?", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=4)

    _box(ax, CX, 0.572, 0.30, 0.055, "⑤ Agent receives message",
         None, facecolor=C["agent"], fontsize=9)

    cx6, cy6 = CX, 0.503
    ax.add_patch(plt.Polygon([[cx6,cy6+dy],[cx6+dx,cy6],[cx6,cy6-dy],[cx6-dx,cy6]],
        closed=True, facecolor=C["decision"], edgecolor="white", lw=1.5, zorder=3))
    ax.text(cx6, cy6, "⑥ Already forwarded?", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=4)

    _box(ax, CX, 0.425, 0.30, 0.060, "⑦ Build LLM prompt",
         "persona  +  memory  +  message content",
         facecolor=C["agent"], fontsize=9)
    _box(ax, CX, 0.348, 0.30, 0.060, "⑧ LLM call -> ForwardDecision",
         "forward  *  reasoning  *  rewritten_content",
         facecolor=C["llm"], fontsize=9)

    cx9, cy9 = CX, 0.273
    ax.add_patch(plt.Polygon([[cx9,cy9+dy],[cx9+dx,cy9],[cx9,cy9-dy],[cx9-dx,cy9]],
        closed=True, facecolor=C["decision"], edgecolor="white", lw=1.5, zorder=3))
    ax.text(cx9, cy9, "⑨ Forward?", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=4)

    _box(ax, CX, 0.200, 0.30, 0.060, "⑩ Rewrite in agent's own voice",
         "fan out child messages  ->  enqueue for t+1",
         facecolor=C["forward"], fontsize=9)
    _box(ax, CX, 0.143, 0.30, 0.042,
         "Update agent memory  *  log FORWARDED",
         None, facecolor=C["agent"], fontsize=8.5)

    # Drop branches
    drop_x = 0.730
    _box(ax, drop_x, cy6, 0.13, 0.045, "DROPPED\n(cycle guard)",
         facecolor=C["drop"], fontsize=8)
    ax.annotate("", xy=(drop_x-0.065, cy6), xytext=(cx6+dx, cy6),
                arrowprops=dict(arrowstyle="-|>", color=C["drop"], lw=1.5))
    ax.text(cx6+dx+0.01, cy6+0.022, "YES", fontsize=8, color=C["drop"], fontweight="bold")
    ax.text(cx6, cy6-0.045, "NO",  fontsize=8, color=C["sim"],  fontweight="bold", ha="center")

    _box(ax, drop_x, cy9, 0.13, 0.045, "DROPPED\n+ update memory",
         facecolor=C["drop"], fontsize=8)
    ax.annotate("", xy=(drop_x-0.065, cy9), xytext=(cx9+dx, cy9),
                arrowprops=dict(arrowstyle="-|>", color=C["drop"], lw=1.5))
    ax.text(cx9+dx+0.01, cy9+0.022, "NO",  fontsize=8, color=C["drop"],    fontweight="bold")
    ax.text(cx9, cy9-0.045, "YES", fontsize=8, color=C["forward"], fontweight="bold", ha="center")

    # No messages -> exit loop (left side)
    exit_x = 0.245
    ax.plot([cx4-dx, exit_x], [cy4, cy4], color=C["drop"], lw=1.5, zorder=2)
    ax.plot([exit_x, exit_x], [cy4, 0.085], color=C["drop"], lw=1.5, zorder=2)
    ax.annotate("", xy=(exit_x, 0.084), xytext=(exit_x, 0.086),
                arrowprops=dict(arrowstyle="-|>", color=C["drop"], lw=1.5))
    ax.text(exit_x-0.012, (cy4+0.085)/2, "NO\n(end)", fontsize=8,
            color=C["drop"], fontweight="bold", ha="right", va="center")
    ax.text(cx4, cy4-0.045, "YES", fontsize=8, color=C["sim"], fontweight="bold", ha="center")

    # t+1 loop-back (right dashed)
    lx = 0.763
    ax.plot([CX+0.15, lx, lx, CX+0.15],
            [0.122, 0.122, 0.743, 0.743],
            color=C["sim"], lw=1.8, linestyle="--", zorder=2)
    ax.annotate("", xy=(CX+0.15, 0.743), xytext=(CX+0.151, 0.743),
                arrowprops=dict(arrowstyle="-|>", color=C["sim"], lw=1.8))
    ax.text(lx+0.005, 0.43, "t = t+1\nloop back", fontsize=8, color=C["sim"],
            fontweight="bold", style="italic", ha="left", va="center")

    # Post-simulation
    _dashed_region(ax, 0.232, 0.02, 0.768, 0.115, "#7f8c8d", "POST-SIMULATION")
    _box(ax, 0.385, 0.068, 0.22, 0.042, "[11] Cascade metrics",
         "size  *  depth  *  virality  *  adoption",
         facecolor=C["output"], fontsize=8)
    _box(ax, 0.615, 0.068, 0.22, 0.042, "[12] Narrative drift",
         "embed messages  ->  drift per step",
         facecolor=C["output"], fontsize=8)
    ax.annotate("", xy=(0.335, 0.068), xytext=(exit_x, 0.082),
                arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=1.5))
    _arrow(ax, 0.496, 0.068, 0.504, 0.068, color="#7f8c8d")

    # Main vertical arrows
    _arrow(ax, CX, 0.852, CX, 0.830)
    _arrow(ax, CX, 0.770, CX, 0.743)
    _arrow(ax, CX, 0.687, CX, cy4+dy)
    _arrow(ax, CX, cy4-dy, CX, 0.599)
    _arrow(ax, CX, 0.544, CX, cy6+dy)
    _arrow(ax, CX, cy6-dy, CX, 0.455)
    _arrow(ax, CX, 0.395, CX, 0.378)
    _arrow(ax, CX, 0.318, CX, cy9+dy)
    _arrow(ax, CX, cy9-dy, CX, 0.230)
    _arrow(ax, CX, 0.170, CX, 0.164)

    # Inputs -> engine
    _arrow(ax, 0.190, 0.855, 0.235, 0.885)
    _arrow(ax, 0.190, 0.740, 0.235, 0.740)
    _arrow(ax, 0.190, 0.625, 0.235, 0.625)

    # Engine -> outputs
    _arrow(ax, 0.765, 0.855, 0.805, 0.855)
    _arrow(ax, 0.765, 0.740, 0.805, 0.740)
    _arrow(ax, 0.765, 0.348, 0.805, 0.580)
    _arrow(ax, 0.765, 0.143, 0.805, 0.430)
    _arrow(ax, 0.765, 0.068, 0.805, 0.290)

    # ── OUTPUTS (simplified) ──────────────────────────────────────────────
    _box(ax, 0.895, 0.855, 0.17, 0.070, "Network structure\n& cascade reach",
         None, facecolor=C["output"], fontsize=8.5)
    _box(ax, 0.895, 0.740, 0.17, 0.070, "Agent decisions\n& epistemic behaviour",
         None, facecolor=C["output"], fontsize=8.5)
    _box(ax, 0.895, 0.580, 0.17, 0.070, "Cascade metrics",
         "size  *  depth  *  virality",
         facecolor=C["output"], fontsize=8.5)
    _box(ax, 0.895, 0.430, 0.17, 0.070, "Narrative drift",
         "similarity  *  drift/step",
         facecolor=C["output"], fontsize=8.5)
    _box(ax, 0.895, 0.290, 0.17, 0.070, "Visualisations",
         "cascade tree\ndiffusion curve",
         facecolor=C["output"], fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating integrated diagram...")
    fig_int = draw_integrated()
    fig_int.savefig(OUT_DIR / "integrated.png", dpi=150, bbox_inches="tight")
    plt.close(fig_int)
    print("  → results/diagrams/integrated.png")

    print("Generating architecture diagram...")
    fig_arch = draw_architecture()
    fig_arch.savefig(OUT_DIR / "architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig_arch)
    print("  → results/diagrams/architecture.png")

    print("Generating flow diagram...")
    fig_flow = draw_flow()
    fig_flow.savefig(OUT_DIR / "flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig_flow)
    print("  → results/diagrams/flow.png")

    print("Generating combined figure...")
    fig_combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14),
                                             facecolor=C["bg"])
    for ax in (ax1, ax2):
        ax.set_facecolor(C["bg"])

    # Re-render both into subplots by saving/loading as images
    import numpy as np
    from matplotlib.image import imread
    import io

    def fig_to_img(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=C["bg"])
        buf.seek(0)
        return imread(buf)

    arch_fig = draw_architecture()
    flow_fig = draw_flow()

    ax1.imshow(fig_to_img(arch_fig))
    ax2.imshow(fig_to_img(flow_fig))
    for ax in (ax1, ax2):
        ax.axis("off")

    plt.close(arch_fig)
    plt.close(flow_fig)

    fig_combined.tight_layout(pad=1.0)
    fig_combined.savefig(OUT_DIR / "combined.png", dpi=130, bbox_inches="tight",
                         facecolor=C["bg"])
    plt.close(fig_combined)
    print("  → results/diagrams/combined.png")
    print("\nDone.")
