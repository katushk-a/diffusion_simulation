"""
CLI entry point for the diffusion simulation.

Usage examples:

  # Quick smoke test with mock LLM
  python main.py --preset topology --backend mock

  # Dataset-driven experiment (true vs fake)
  python main.py --dataset data/sample_news.csv --backend ollama --model llama3.1:8b
  python main.py --dataset data/isot.csv --text-col text --label-col label \\
                 --label-map '{"REAL":"true","FAKE":"fake"}' --n-per-label 10

  # Run preset experiments
  python main.py --preset narrative --backend openai --model gpt-4o-mini --runs 3
"""

import argparse
import asyncio
import os
import pathlib
import sys
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logging import setup_logging


async def run_configs(configs, n_runs: int = 1, base_seed: int = 42) -> None:
    from experiments.runner import ExperimentRunner
    from experiments.multi_run import run_multi

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg.name}  (runs={n_runs})")
        print(f"{'='*60}")

        if n_runs > 1:
            multi = await run_multi(cfg, n_runs=n_runs, base_seed=base_seed)
            multi.print_summary()
            print(f"  Results saved → {multi.output_path}")
        else:
            runner = ExperimentRunner(cfg)
            result = await runner.run()
            summary = result.summary()
            for k, v in summary.items():
                print(f"  {k:<35} {v}")
            print(f"  Results saved → {result.output_path}")


def main() -> None:
    from datetime import datetime
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging("INFO", log_file=f"logs/run_{_ts}.log")

    parser = argparse.ArgumentParser(description="Information Diffusion Simulation")
    parser.add_argument(
        "--preset",
        choices=["topology", "community", "ablation_memory", "all"],
        help="Run a named group of pre-built experiments.",
    )
    parser.add_argument(
        "--config",
        help="Path to a JSON experiment config file.",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "openai", "ollama"],
        help="LLM backend to use (default: mock).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (e.g. gpt-4o-mini or llama3.2).",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=30,
        help="Number of agents (default: 30). 30 agents ~25 min/run on Ollama llama3.1:8b.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="Max simulation steps (default: 6). Cascades typically die naturally by step 5-6.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs for statistical aggregation (default: 1).",
    )
    parser.add_argument(
        "--max-concurrent-llm",
        type=int,
        default=None,
        help="Max concurrent LLM calls per step. Use 2 for local Ollama (tested: ~25 min/run at 30 agents), 8+ for hosted APIs.",
    )
    parser.add_argument(
        "--no-narrative",
        action="store_true",
        help="Skip narrative/embedding metrics.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        metavar="MODEL",
        help="Embedding model for narrative metrics (overrides the default for the chosen backend).",
    )
    # Dataset mode arguments
    parser.add_argument(
        "--dataset",
        help="Path to a news dataset file (CSV or JSON) for dataset-driven experiments.",
    )
    parser.add_argument(
        "--text-col", default="text",
        help="Column name for the news text (default: text).",
    )
    parser.add_argument(
        "--label-col", default="label",
        help="Column name for the content label (default: label).",
    )
    parser.add_argument(
        "--title-col", default=None,
        help="Column name for the title, prepended to text if provided.",
    )
    parser.add_argument(
        "--label-map", default=None,
        help='JSON dict to normalise raw labels, e.g. \'{"REAL":"true","FAKE":"fake"}\'',
    )
    parser.add_argument(
        "--n-per-label", type=int, default=5,
        help="Number of news items to sample per label (default: 5).",
    )
    parser.add_argument(
        "--max-chars", type=int, default=600,
        help="Truncate news text to this many characters (default: 600).",
    )
    parser.add_argument(
        "--network-file",
        default=None,
        help="Path to a real-world network file (GraphML, GML, GEXF, or edge list). "
             "Overrides --n-agents and network type.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Show available preset experiments and exit.",
    )
    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        print("  topology       – 3×2 factorial: random/scale_free/small_world × true/fake (ISOT)")
        print("  community      – 3-clique community network × true/fake (ISOT)")
        print("  ablation_memory – memory on/off × true/fake on scale-free (ISOT)")
        print("  all            – run all of the above")
        return

    kwargs = dict(
        llm_backend=args.backend,
        n_agents=args.n_agents,
        max_steps=args.steps,
        seed=args.seed,
    )
    if args.model:
        kwargs["llm_model"] = args.model
    if args.no_narrative:
        kwargs["compute_narrative_metrics"] = False
    if args.embed_model:
        kwargs["llm_embedding_model"] = args.embed_model
    if args.network_file:
        kwargs["network_file"] = args.network_file
    if args.max_concurrent_llm is not None:
        kwargs["max_concurrent_llm"] = args.max_concurrent_llm

    # ----------------------------------------------------------------
    # Dataset mode — run independently, then exit
    # ----------------------------------------------------------------
    if args.dataset:
        import json as _json
        from data.dataset import NewsDataset
        from experiments.config import ExperimentConfig
        from experiments.dataset_experiment import DatasetExperiment

        label_map = _json.loads(args.label_map) if args.label_map else None
        ext = pathlib.Path(args.dataset).suffix.lower()
        if ext == ".json":
            ds = NewsDataset.from_json(
                args.dataset,
                text_col=args.text_col,
                label_col=args.label_col,
                title_col=args.title_col,
                label_map=label_map,
                max_chars=args.max_chars,
            )
        else:
            ds = NewsDataset.from_csv(
                args.dataset,
                text_col=args.text_col,
                label_col=args.label_col,
                title_col=args.title_col,
                label_map=label_map,
                max_chars=args.max_chars,
            )

        print(f"Dataset loaded: {len(ds)} items  |  labels: {ds.summary()}")

        base_cfg = ExperimentConfig(
            name=pathlib.Path(args.dataset).stem,
            n_agents=args.n_agents,
            network_type="scale_free",
            network_file=args.network_file,
            max_steps=args.steps,
            llm_backend=args.backend,
            llm_model=args.model,
            seed=args.seed,
            compute_narrative_metrics=not args.no_narrative,
            seed_messages=[],
        )

        exp = DatasetExperiment(
            base_config=base_cfg,
            dataset=ds,
            n_per_label=args.n_per_label,
        )
        result = asyncio.run(exp.run())
        result.print_comparison()
        print(f"\nResults saved → results/{base_cfg.name}/")
        return

    configs = []

    if args.config:
        from experiments.config import ExperimentConfig
        cfg = ExperimentConfig.from_json(args.config)
        configs.append(cfg)

    elif args.preset:
        from experiments.presets import (
            all_presets,
            community_experiment,
            memory_ablation_experiments,
            network_topology_experiments,
        )
        match args.preset:
            case "topology":
                configs = network_topology_experiments(**kwargs)
            case "community":
                configs = community_experiment(**kwargs)
            case "ablation_memory":
                configs = memory_ablation_experiments(**kwargs)
            case "all":
                configs = all_presets(**kwargs)
    else:
        # Default: single quick demo run — samples one true and one fake article from ISOT
        from experiments.config import ExperimentConfig, SeedMessage
        from experiments.presets import _sample
        configs = []
        for label in ["true", "fake"]:
            configs.append(ExperimentConfig(
                name=f"demo_{label}",
                description=f"Quick demo run ({label} news from ISOT).",
                seed=args.seed,
                n_agents=args.n_agents,
                network_type="scale_free",
                max_steps=args.steps,
                seed_messages=[
                    SeedMessage(
                        content=_sample(label, args.seed, experiment="demo"),
                        origin_agent_id="agent_000",
                        label=label,
                    )
                ],
                llm_backend=args.backend,
                compute_narrative_metrics=True,
            ))

    asyncio.run(run_configs(configs, n_runs=args.runs, base_seed=args.seed))


if __name__ == "__main__":
    main()
