# LLM-Backed Information Diffusion Simulation

A multi-agent simulation of information spreading through social networks,
where each agent is an LLM-backed persona that decides whether to forward,
rewrite, or suppress a message based on its epistemic profile.

All seed content is drawn automatically from the **ISOT Fake News Dataset**
(`data/True.csv` / `data/Fake.csv`). The same `--seed` value always produces
the same article; different seeds produce different articles.

---

## Architecture

![Simulation Architecture](results/diagrams/integrated.png)

The simulation operates in three layers:

- **Input** — experiment config, agent personas (35 total across 4 epistemic types), and a seed network topology
- **Engine** — asyncio-based step runner where agents receive messages from graph neighbours, call the LLM, and produce forward/rewrite decisions
- **Output** — cascade metrics, narrative drift statistics, per-run JSON logs, and visualisations

---

## Experiment Runbook

Three core experiments, listed in recommended execution order.
All commands use the MetaCentrum API with `--max-concurrent-llm 1`.

---

### 0. Smoke test (always run first)

Verifies the setup with a mock LLM — no API needed, finishes in seconds.

```bash
python main.py --backend mock --n-agents 10 --steps 3
```

Expected: two result folders — `results/demo_true_YYYYMMDD_HHMM/` and
`results/demo_fake_YYYYMMDD_HHMM/` — each with `summary.json`,
`cascade_metrics.json`, and plots.

---

### 1. Network Topology Comparison

**Research question:** Does network structure (random vs. scale-free vs. small-world)
affect how far and how faithfully true vs. fake information spreads?

**What it runs:** 6 configs — 3 topologies × 2 content types (true / fake from ISOT).
All conditions use the same two ISOT articles so topology is the only variable.

```bash
python main.py \
  --preset topology \
  --backend openai \
  --model agentic \
  --n-agents 30 \
  --steps 6 \
  --runs 5 \
  --max-concurrent-llm 1
```

**Key outputs to compare across topologies:**
- `mean_unique_receivers` — how many distinct agents were reached
- `mean_cascade_depth` — how many hops the message travelled
- `mean_structural_virality` — shape of the cascade (broadcast vs. chain)
- `mean_adoption_rate` — fraction of reached agents that forwarded
- `mean_similarity_to_seed` — narrative fidelity across the cascade
- `mean_semantic_drift_per_step` — rate of meaning change per hop

---

### 2. Community Network

**Research question:** When a network has strong community structure (dense
in-group, sparse bridges), does information cross community boundaries? Does
true vs. fake news behave differently at the crossing point?

**What it runs:** 2 configs (true / fake) on a 3-clique community network
with p_in=0.8, p_out=0.05.

```bash
python main.py \
  --preset community \
  --backend openai \
  --model agentic \
  --n-agents 30 \
  --steps 6 \
  --runs 5 \
  --max-concurrent-llm 1
```

**Key outputs:**
- `mean_cascade_depth` — does information cross into other cliques?
- `mean_adoption_rate` — compare to scale-free topology results
- `network_graph.png` — visualise which cliques were reached

---

### 3. Memory Ablation

**Research question:** Does giving agents episodic memory of their past
decisions change how selectively they forward, and how much they rewrite?

**What it runs:** 4 configs — memory on/off × true/fake on scale-free (ISOT).

```bash
python main.py \
  --preset ablation_memory \
  --backend openai \
  --model agentic \
  --n-agents 30 \
  --steps 6 \
  --runs 5 \
  --max-concurrent-llm 1
```

**Key outputs to compare (memory_on vs memory_off, per content type):**
- `mean_adoption_rate` — does suppressing memory increase forwarding?
- `mean_unique_forwarders` — does memory make agents more selective?
- `mean_similarity_to_seed` — does memory change how agents rewrite content?

---

### Run all experiments

```bash
python main.py \
  --preset all \
  --backend openai \
  --model agentic \
  --n-agents 30 \
  --steps 6 \
  --runs 5 \
  --max-concurrent-llm 1
```

Runs all 12 configs (6 topology + 2 community + 4 ablation) sequentially.

---

## Execution Order

| # | Experiment | Preset | Configs | Runs total |
|---|---|---|---|---|
| 0 | Smoke test | (default, mock) | 2 | 1 each |
| 1 | Network topology | `topology` | 6 | 30 (6×5) |
| 2 | Community network | `community` | 2 | 10 (2×5) |
| 3 | Memory ablation | `ablation_memory` | 4 | 20 (4×5) |

---

## Practical Notes

**Changing the news article:**
Each `--seed` value deterministically samples a different article from ISOT.
Different preset types always use different articles even at the same seed.

```bash
# Topology experiment with article set A (seed 42)
python main.py --preset topology --backend openai --model agentic \
  --n-agents 30 --steps 6 --runs 5 --max-concurrent-llm 1 --seed 42

# Topology experiment with article set B (seed 100)
python main.py --preset topology --backend openai --model agentic \
  --n-agents 30 --steps 6 --runs 5 --max-concurrent-llm 1 --seed 100
```

**Seeds and reproducibility:**
- Default seed is 42
- Multi-run seeds are offset automatically: run 0 → seed 42, run 1 → seed 43, …
- The ISOT article is fixed at config-creation time using the base seed
- To reproduce any result exactly, use the seed from that run's `config.json`

**Result location:**
```
results/
  {experiment_name}_{YYYYMMDD_HHMM}/
    config.json           # full experiment config (reproducible)
    log.json              # all messages and events
    cascade_metrics.json  # per-cascade structural metrics
    narrative_stats.json  # per-cascade semantic drift metrics
    summary.json          # aggregated numbers (use in thesis tables)
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
