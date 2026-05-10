"""
Dataset-driven diffusion experiment.

Instead of a fixed seed message, news items are drawn from a dataset
(true / fake). One simulation is run per news item.
Results are then aggregated by label so you can directly compare
diffusion dynamics across content types.

Usage
-----
    from data.dataset import NewsDataset
    from experiments.config import ExperimentConfig
    from experiments.dataset_experiment import DatasetExperiment

    ds = NewsDataset.from_csv("data/isot.csv", text_col="text", label_col="label",
                              label_map={"REAL": "true", "FAKE": "fake"})

    cfg = ExperimentConfig(
        name="isot_run",
        n_agents=30,
        network_type="scale_free",
        max_steps=6,
        llm_backend="ollama",
        llm_model="llama3.1:8b",
        seed_messages=[],          # dataset_experiment fills these
    )

    exp = DatasetExperiment(
        base_config=cfg,
        dataset=ds,
        n_per_label=5,             # run 5 true + 5 fake items
        origin_agent_id="agent_000",
    )
    result = await exp.run()
    result.print_comparison()
    result.save()
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time
from dataclasses import asdict, dataclass

from data.dataset import NewsDataset, NewsItem
from experiments.config import ExperimentConfig, SeedMessage
from experiments.runner import ExperimentResult, ExperimentRunner
from metrics.cascade import CascadeMetrics, compare_by_label
from metrics.narrative import CascadeNarrativeStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-item result
# ---------------------------------------------------------------------------

@dataclass
class ItemResult:
    item: NewsItem
    experiment: ExperimentResult

    @property
    def cascade_metrics(self) -> list[CascadeMetrics]:
        return self.experiment.cascade_metrics

    @property
    def narrative_stats(self) -> list[CascadeNarrativeStats]:
        return self.experiment.narrative_stats


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------

@dataclass
class DatasetExperimentResult:
    base_config: ExperimentConfig
    item_results: list[ItemResult]
    # cascade metrics from all items, each tagged with its label
    all_cascade_metrics: list[CascadeMetrics]
    comparison: dict          # label → aggregated metrics (from compare_by_label)
    elapsed_seconds: float

    def print_comparison(self) -> None:
        print(f"\n{'='*64}")
        print(f"Dataset experiment: {self.base_config.name}")
        print(f"Items run: {len(self.item_results)}   "
              f"Time: {self.elapsed_seconds:.1f}s")
        print(f"{'='*64}")
        header = f"{'label':<14} {'n':>4} {'size':>6} {'depth':>6} "  \
                 f"{'breadth':>8} {'sv':>7} {'adopt':>7}"
        print(header)
        print("-" * 64)
        for label, v in self.comparison.items():
            print(
                f"{label:<14} {v['n_cascades']:>4} "
                f"{v['mean_size']:>6.2f} {v['mean_depth']:>6.2f} "
                f"{v['mean_max_breadth']:>8.2f} "
                f"{v['mean_structural_virality']:>7.3f} "
                f"{v['mean_adoption_rate']:>7.3f}"
            )

    def save(self) -> pathlib.Path:
        out = pathlib.Path(self.base_config.output_dir) / self.base_config.name
        out.mkdir(parents=True, exist_ok=True)

        # Comparison table
        (out / "comparison_by_label.json").write_text(
            json.dumps(self.comparison, indent=2)
        )

        # All cascade metrics (with labels)
        (out / "all_cascade_metrics.json").write_text(
            json.dumps([asdict(c) for c in self.all_cascade_metrics], indent=2)
        )

        # Per-item summary
        per_item = []
        for ir in self.item_results:
            per_item.append({
                "item_id": ir.item.item_id,
                "label": ir.item.label,
                "title": ir.item.title,
                "text_preview": ir.item.text[:120],
                "summary": ir.experiment.summary(),
            })
        (out / "per_item_summary.json").write_text(
            json.dumps(per_item, indent=2)
        )

        # Narrative records per label (if computed)
        narrative_by_label: dict[str, list] = {}
        for ir in self.item_results:
            label = ir.item.label
            for ns in ir.narrative_stats:
                narrative_by_label.setdefault(label, []).append({
                    "item_id": ir.item.item_id,
                    "cascade_id": ns.cascade_id,
                    "seed_content": ns.seed_content,
                    "mean_similarity_to_seed": ns.mean_similarity_to_seed,
                    "min_similarity_to_seed": ns.min_similarity_to_seed,
                    "mean_edit_distance": ns.mean_edit_distance,
                    "semantic_drift_per_step": ns.semantic_drift_per_step,
                })
        if narrative_by_label:
            (out / "narrative_by_label.json").write_text(
                json.dumps(narrative_by_label, indent=2)
            )

        # Visualisations
        try:
            from analysis.visualise import save_dataset_plots
            save_dataset_plots(self, out)
        except Exception as exc:
            logger.warning("Could not save dataset plots: %s", exc)

        logger.info("Dataset experiment results saved to %s", out)
        return out


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class DatasetExperiment:
    """
    Runs one simulation per sampled news item and compares results by label.

    Parameters
    ----------
    base_config       : ExperimentConfig used as template for every item run.
                        seed_messages will be overwritten per item.
    dataset           : NewsDataset to sample from
    n_per_label       : how many items to sample from each label
    origin_agent_id   : which agent injects the seed (default: agent_000)
    labels            : which labels to include (default: all in dataset)
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        dataset: NewsDataset,
        n_per_label: int = 5,
        origin_agent_id: str = "agent_000",
        labels: list[str] | None = None,
    ) -> None:
        self.base_config = base_config
        self.dataset = dataset
        self.n_per_label = n_per_label
        self.origin_agent_id = origin_agent_id
        self.labels = labels or dataset.labels()

    async def run(self) -> DatasetExperimentResult:
        start = time.monotonic()
        item_results: list[ItemResult] = []

        for label in self.labels:
            try:
                items = self.dataset.sample(
                    n=self.n_per_label,
                    label=label,
                    seed=self.base_config.seed,
                )
            except ValueError as e:
                logger.warning("Skipping label %r: %s", label, e)
                continue

            logger.info(
                "Label %r: running %d items.", label, len(items)
            )
            for i, item in enumerate(items):
                result = await self._run_item(item, item_index=i)
                item_results.append(ItemResult(item=item, experiment=result))

        # Collect all cascade metrics across items, preserving label
        all_metrics: list[CascadeMetrics] = []
        for ir in item_results:
            all_metrics.extend(ir.cascade_metrics)

        comparison = compare_by_label(all_metrics)
        elapsed = time.monotonic() - start

        result = DatasetExperimentResult(
            base_config=self.base_config,
            item_results=item_results,
            all_cascade_metrics=all_metrics,
            comparison=comparison,
            elapsed_seconds=elapsed,
        )
        result.save()
        return result

    async def _run_item(self, item: NewsItem, item_index: int) -> ExperimentResult:
        """Run one simulation seeded with a single news item."""
        cfg = self.base_config.model_copy(update={
            # Unique name so each item gets its own results folder
            "name": f"{self.base_config.name}/{item.label}_{item.item_id}",
            # Vary seed slightly per item so network topology differs
            "seed": self.base_config.seed + item_index,
            "seed_messages": [
                SeedMessage(
                    content=item.text,
                    origin_agent_id=self.origin_agent_id,
                    step=0,
                    label=item.label,
                    item_id=item.item_id,
                )
            ],
        })
        logger.info(
            "  Running item %s [%s]: %s...",
            item.item_id, item.label, item.text[:60],
        )
        runner = ExperimentRunner(cfg)
        return await runner.run()
