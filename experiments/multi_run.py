"""
Multi-run experiment wrapper for statistical reproducibility.

Runs the same ExperimentConfig k times with seeds base_seed, base_seed+1, …,
then aggregates per-metric mean ± std across runs.

Usage:
    result = await run_multi(config, n_runs=5, base_seed=42)
    print(result.aggregated)
    result.save()
"""

from __future__ import annotations

import json
import logging
import pathlib
import statistics
from dataclasses import dataclass

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentResult, ExperimentRunner

logger = logging.getLogger(__name__)


@dataclass
class MultiRunResult:
    config_name: str
    n_runs: int
    runs: list[ExperimentResult]
    # metric → {"mean": float, "std": float, "values": list[float]}
    aggregated: dict

    def print_summary(self) -> None:
        print(f"\nMulti-run summary: {self.config_name}  (n={self.n_runs})")
        print(f"{'metric':<40} {'mean':>10}  {'std':>10}")
        print("-" * 64)
        for k, v in self.aggregated.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"  {k:<38} {v['mean']:>10.4f}  {v['std']:>10.4f}")

    def save(self, output_dir: str = "results") -> pathlib.Path:
        out = pathlib.Path(output_dir) / f"{self.config_name}_multi"
        out.mkdir(parents=True, exist_ok=True)

        (out / "aggregated.json").write_text(
            json.dumps(self.aggregated, indent=2)
        )

        # Per-run summaries for traceability
        run_summaries = [r.summary() for r in self.runs]
        (out / "per_run_summaries.json").write_text(
            json.dumps(run_summaries, indent=2)
        )

        logger.info("Multi-run results saved to %s", out)
        return out


async def run_multi(
    config: ExperimentConfig,
    n_runs: int = 5,
    base_seed: int = 42,
) -> MultiRunResult:
    """
    Run *config* n_runs times with seeds [base_seed, base_seed+1, …].
    Each individual run is saved to its own subdirectory.
    """
    results: list[ExperimentResult] = []

    for i in range(n_runs):
        run_seed = base_seed + i
        cfg_i = config.model_copy(
            update={
                "seed": run_seed,
                "name": f"{config.name}_run{i:02d}",
            }
        )
        logger.info(
            "Multi-run %d/%d  (seed=%d, name=%s)",
            i + 1, n_runs, run_seed, cfg_i.name,
        )
        runner = ExperimentRunner(cfg_i)
        result = await runner.run()
        results.append(result)

    # Aggregate numeric metrics across runs
    summaries = [r.summary() for r in results]
    numeric_keys = [
        k for k, v in summaries[0].items()
        if isinstance(v, (int, float))
    ]

    aggregated: dict = {
        "config_name": config.name,
        "n_runs": n_runs,
        "base_seed": base_seed,
    }
    for k in numeric_keys:
        vals = [s[k] for s in summaries]
        aggregated[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals,
        }

    multi_result = MultiRunResult(
        config_name=config.name,
        n_runs=n_runs,
        runs=results,
        aggregated=aggregated,
    )
    multi_result.save(config.output_dir)
    return multi_result
