# LLM-Backed Information Diffusion Simulation

A multi-agent simulation of information spreading through social networks,
where each agent is an LLM-backed persona that decides whether to forward,
rewrite, or suppress a message based on its epistemic profile.

---

## Architecture

![Simulation Architecture](results/diagrams/integrated.png)

The simulation operates in three layers:

- **Input** — experiment config, agent personas (35 total across 4 epistemic types), and a seed network topology
- **Engine** — asyncio-based step runner where agents receive messages from graph neighbours, call the LLM, and produce forward/rewrite decisions
- **Output** — cascade metrics, narrative drift statistics, per-run JSON logs, and visualisations

---

## Experiment Runbook

The experiments below are the core thesis experiments, listed in the recommended
order of execution. Each section states what the experiment tests, the exact command
to run it, and what outputs to look at.

---

### 0. Smoke test (always run first)

Verifies the setup is working with a mock LLM (no Ollama needed, finishes in seconds).

```bash
python main.py --backend mock --n-agents 30 --steps 6
```

Expected: a `results/demo_YYYYMMDD_HHMM/` folder with `summary.json`, `cascade_metrics.json`, and plots.

---

### 1. Network Topology Comparison

**Research question:** Does network structure (random vs. scale-free vs. small-world)
affect how fast and how far true information spreads?

**What it runs:** 3 configs — one per topology — each seeding the same true-news message.

```bash
python main.py \
  --preset topology \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --steps 6 \
  --max-concurrent-llm 4
```

**For statistical replication (recommended for final thesis runs):**
```bash
python main.py \
  --preset topology \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --steps 6 \
  --max-concurrent-llm 4 \
  --runs 5
```

**Time estimate:** ~25–35 min per run at 30 agents on llama3.2:3b (GPU). With `--runs 5`: ~2–3 h per topology config.

**Key outputs to compare across topologies:**
- `mean_unique_receivers` — how many distinct agents were actually reached
- `mean_cascade_depth` — how many hops the message travelled
- `mean_structural_virality` — shape of the cascade (broadcast vs. chain)
- `mean_adoption_rate` — fraction of the network that forwarded

---

### 2. Narrative Drift — True vs. Fake vs. Misleading

**Research question:** Does content type (true / fake / misleading) change how much the
narrative mutates as it passes through agents? Does misinformation amplify more?

**What it runs:** 3 configs on a scale-free network — one per content type.

```bash
python main.py \
  --preset narrative \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --steps 6 \
  --max-concurrent-llm 4
```

**With replication:**
```bash
python main.py \
  --preset narrative \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --steps 6 \
  --max-concurrent-llm 4 \
  --runs 5
```

**Key outputs to compare across content types:**
- `mean_similarity_to_seed` — how faithful the forwarded versions are (1.0 = identical)
- `mean_semantic_drift_per_step` — rate of narrative change per hop
- `mean_unique_forwarders` — does misinformation recruit more forwarders?
- Visualisation: `narrative_similarity.png` in each result folder

---

### 3. Community Network — Cross-Clique Diffusion

**Research question:** When a network has strong community structure (dense in-group, sparse
bridges), does information cross community boundaries? Does narrative drift at the crossing point?

**What it runs:** 3 configs (true / fake / misleading) on a 3-clique community network with
p_in=0.8, p_out=0.05. Directly comparable to Experiment 2.

```bash
python main.py \
  --preset community \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --steps 6 \
  --max-concurrent-llm 4
```

**Key outputs:**
- `mean_cascade_depth` — does information cross into other cliques at all?
- `mean_adoption_rate` — compare to scale-free (Exp. 2) with same seed content
- `network_graph.png` — visualise which cliques were reached
- **Limitation to document:** If depth=1 for some conditions, drift=0.0 is a measurement
  artefact (not enough hops for regression), not a real finding.

---

### 4. Topology × Content Factorial (3 × 3)

**Research question:** Do topology effects and content-type effects interact? Is scale-free
always the most viral topology, or does it depend on the content type?

**What it runs:** 9 configs — all combinations of 3 topologies × 3 content types.

```bash
python main.py \
  --preset topology_content \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --steps 6 \
  --max-concurrent-llm 4
```

**Time estimate:** 9 separate runs — budget ~4–5 h total on local hardware.

**After running, generate comparison plots:**
```bash
python analysis/topo_x_content_plots.py
```

Plots saved to `results/topo_x_content_comparison/`:
- `01_cascade_size.png` — reach by topology and content type
- `02_semantic_drift.png` — drift rate across conditions
- `03_heatmaps.png` — 2D heatmap of key metrics
- `04_similarity_to_seed.png` — narrative fidelity

---

### 5. Memory Ablation — Epistemic Memory On vs. Off

**Research question:** Does giving agents memory of their past decisions change their
forwarding behaviour? Does memory reduce redundant forwarding (each agent only
forwards once per cascade) or change narrative drift?

**What it runs:** 4 configs — memory_on/off × true_news/fake_news on scale-free.

```bash
python main.py \
  --preset ablation_memory \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --steps 6 \
  --max-concurrent-llm 4
```

**Key outputs to compare (memory_on vs memory_off, per content type):**
- `mean_unique_forwarders` — does memory make agents more selective?
- `mean_adoption_rate` — does suppressing memory increase total reach?
- `mean_similarity_to_seed` — does memory change how agents rewrite content?

---

### 6. Dataset-Driven Experiment (optional, real-world content)

Use actual news headlines from a CSV dataset instead of hand-crafted seed messages.
Good for validating that findings generalise beyond the preset content strings.

```bash
python main.py \
  --dataset data/sample_news.csv \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --steps 6 \
  --n-per-label 5 \
  --max-concurrent-llm 4
```

For the ISOT dataset (with column renaming):
```bash
python main.py \
  --dataset data/isot.csv \
  --text-col text \
  --label-col label \
  --label-map '{"REAL":"true","FAKE":"fake"}' \
  --n-per-label 5 \
  --backend ollama \
  --model llama3.2:3b \
  --n-agents 30 \
  --max-concurrent-llm 4
```

---

## Recommended Execution Order for Thesis

| # | Experiment | Preset | Priority |
|---|---|---|---|
| 1 | Smoke test | (default, mock) | Always first |
| 2 | Narrative drift | `narrative` | Core — run with `--runs 5` |
| 3 | Network topology | `topology` | Core — run with `--runs 5` |
| 4 | Community network | `community` | Core |
| 5 | Topology × Content | `topology_content` | Core — most comprehensive |
| 6 | Memory ablation | `ablation_memory` | Ablation study |
| 7 | Dataset-driven | `--dataset` | Validation / optional |

---

## Practical Notes

**Model choice on local hardware (RTX A500, 4 GB VRAM):**
- `llama3.2:3b` — fits in GPU, ~5–10 s/call, recommended for all thesis runs
- `llama3.1:8b` — does NOT fit (4.5 GB required), falls back to CPU at ~68 s/call — avoid
- Pull the model once: `ollama pull llama3.2:3b`

**Concurrency:**
- `--max-concurrent-llm 4` is the tested sweet spot for llama3.2:3b on this machine
- Higher values may cause Ollama to queue or error; lower values waste GPU time

**Seeds and reproducibility:**
- All experiments use `--seed 42` by default
- For multi-run replication, seeds are offset automatically: run 0 uses seed 42, run 1 uses 43, etc.
- To reproduce any single result exactly, use the seed from that run's `config.json`

**Result location:**
```
results/
  {experiment_name}_{YYYYMMDD_HHMM}/
    config.json          # full experiment config (reproducible)
    log.json             # all messages and events
    cascade_metrics.json # per-cascade structural metrics
    narrative_stats.json # per-cascade semantic drift metrics
    summary.json         # aggregated numbers (copy this into your thesis table)
    network_graph.png
    diffusion_curve.png
    cascade_tree_*.png
    narrative_similarity.png
```

---

## Agent Personas

35 personas across 4 epistemic types (Cassam framework):

| Type | Count | Description |
|---|---|---|
| `open` | 9 | Calibrated, source-tracing, intellectually humble agents |
| `closed` | 9 | Tribal, motivated-reasoning, anti-expertise agents |
| `credulous` | 9 | Low-effort verifiers; trend-following, emotionally-driven sharers |
| `strategic` | 8 | Reputation/brand-driven agents; truth subordinated to goals |

Personas are sampled randomly per run from `data/personas.json`.
Each simulation of N agents draws N personas (with replacement if N > 35).
