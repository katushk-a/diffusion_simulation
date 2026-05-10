"""
News dataset loader for the diffusion simulation.

Supports any CSV or JSON file where each row is a news item with at least:
  - a text field  (the story content or headline)
  - a label field (e.g. "true", "fake", "real", "misleading", 0, 1 ...)

Usage
-----
    ds = NewsDataset.from_csv(
        "data/isot.csv",
        text_col="text",
        label_col="label",
        label_map={"REAL": "true", "FAKE": "fake"},   # normalise labels
        title_col="title",       # optional, prepended to text if given
        max_chars=500,           # truncate long articles
    )
    true_sample  = ds.sample(n=10, label="true",  seed=42)
    fake_sample  = ds.sample(n=10, label="fake",  seed=42)

Common datasets and their column names
---------------------------------------
ISOT (kaggle):        text_col="text",  label_col="label",  label_map={"REAL":"true","FAKE":"fake"}
FakeNewsNet:          text_col="text",  label_col="label",  label_map={"real":"true","fake":"fake"}
LIAR (tsv):           use from_csv with sep="\\t", text_col="statement", label_col="label"
Custom JSON array:    from_json("file.json", text_col=..., label_col=...)
"""

from __future__ import annotations

import csv
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class NewsItem:
    """One row from the dataset."""
    item_id: str
    text: str           # the content used as seed message
    label: str          # normalised label: "true", "fake", "misleading", etc.
    title: str = ""     # optional headline (for reference / display)
    source: str = ""    # optional origin


class NewsDataset:
    """
    In-memory collection of NewsItems loaded from a file.
    """

    def __init__(self, items: list[NewsItem]) -> None:
        self._items = items

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | pathlib.Path,
        text_col: str,
        label_col: str,
        id_col: Optional[str] = None,
        title_col: Optional[str] = None,
        source_col: Optional[str] = None,
        label_map: Optional[dict[str, str]] = None,
        max_chars: int = 600,
        sep: str = ",",
        encoding: str = "utf-8",
    ) -> "NewsDataset":
        """
        Load from a CSV (or TSV) file.

        label_map: optional dict to normalise raw label strings,
                   e.g. {"REAL": "true", "FAKE": "fake", "0": "true", "1": "fake"}
        max_chars: truncate text to this many characters so LLM prompts stay short.
        """
        items = []
        with open(path, newline="", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=sep)
            for i, row in enumerate(reader):
                raw_label = row.get(label_col, "").strip()
                label = (label_map or {}).get(raw_label, raw_label).lower()
                title = row.get(title_col, "").strip() if title_col else ""
                text = row.get(text_col, "").strip()
                # Prepend title if present and not already in text
                if title and not text.startswith(title):
                    text = f"{title}. {text}"
                text = text[:max_chars]
                items.append(NewsItem(
                    item_id=row.get(id_col, str(i)).strip() if id_col else str(i),
                    text=text,
                    label=label,
                    title=title,
                    source=row.get(source_col, "").strip() if source_col else "",
                ))
        return cls(items)

    @classmethod
    def from_json(
        cls,
        path: str | pathlib.Path,
        text_col: str,
        label_col: str,
        id_col: Optional[str] = None,
        title_col: Optional[str] = None,
        source_col: Optional[str] = None,
        label_map: Optional[dict[str, str]] = None,
        max_chars: int = 600,
    ) -> "NewsDataset":
        """Load from a JSON array of objects."""
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        items = []
        for i, row in enumerate(rows):
            raw_label = str(row.get(label_col, "")).strip()
            label = (label_map or {}).get(raw_label, raw_label).lower()
            title = str(row.get(title_col, "")).strip() if title_col else ""
            text = str(row.get(text_col, "")).strip()
            if title and not text.startswith(title):
                text = f"{title}. {text}"
            text = text[:max_chars]
            items.append(NewsItem(
                item_id=str(row.get(id_col, i)) if id_col else str(i),
                text=text,
                label=label,
                title=title,
                source=str(row.get(source_col, "")).strip() if source_col else "",
            ))
        return cls(items)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def labels(self) -> list[str]:
        """All unique label values present in the dataset."""
        return sorted({item.label for item in self._items})

    def by_label(self, label: str) -> list[NewsItem]:
        return [item for item in self._items if item.label == label]

    def sample(
        self,
        n: int,
        label: Optional[str] = None,
        seed: int = 42,
    ) -> list[NewsItem]:
        """
        Return n items, optionally filtered to a single label.
        Samples without replacement if n ≤ pool size, otherwise with replacement.
        """
        pool = self.by_label(label) if label else list(self._items)
        if not pool:
            raise ValueError(f"No items with label={label!r}. Available: {self.labels()}")
        rng = random.Random(seed)
        if n <= len(pool):
            return rng.sample(pool, n)
        # Sample with replacement for small datasets
        return [rng.choice(pool) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._items)

    def summary(self) -> dict[str, int]:
        """Count items per label."""
        counts: dict[str, int] = {}
        for item in self._items:
            counts[item.label] = counts.get(item.label, 0) + 1
        return dict(sorted(counts.items()))

    @classmethod
    def load_isot(
        cls,
        data_dir: str | pathlib.Path | None = None,
        max_chars: int = 600,
    ) -> "NewsDataset":
        """
        Load the ISOT Fake News Dataset from True.csv and Fake.csv.
        Labels are inferred from the filename (true / fake).
        Falls back gracefully if a file is missing.
        """
        if data_dir is None:
            data_dir = pathlib.Path(__file__).parent
        data_dir = pathlib.Path(data_dir)

        items: list[NewsItem] = []
        for filename, label in [("True.csv", "true"), ("Fake.csv", "fake")]:
            path = data_dir / filename
            if not path.exists():
                continue
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    title = row.get("title", "").strip()
                    text = row.get("text", "").strip()
                    if title and not text.startswith(title):
                        text = f"{title}. {text}"
                    items.append(NewsItem(
                        item_id=f"{label}_{i}",
                        text=text[:max_chars],
                        label=label,
                        title=title,
                    ))
        return cls(items)
