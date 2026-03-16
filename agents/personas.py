"""
Persona and disposition loading for the diffusion simulation.

All persona templates and disposition descriptions live in:
    data/personas.json

This module loads them at import time and exposes:
  - DISPOSITION_DESCRIPTIONS  dict[str, str]
  - PERSONA_TEMPLATES         list[dict]  (name, background, traits)
  - build_agent_personas()    main factory used by ExperimentRunner
"""

from __future__ import annotations

import json
import pathlib
from typing import Literal

from agents.base import AgentPersona

# ---------------------------------------------------------------------------
# Load data file
# ---------------------------------------------------------------------------

_DATA_FILE = pathlib.Path(__file__).parent.parent / "data" / "personas.json"


def _load() -> tuple[dict[str, str], list[dict]]:
    with _DATA_FILE.open() as f:
        raw = json.load(f)
    return raw["dispositions"], raw["personas"]


DISPOSITION_DESCRIPTIONS, PERSONA_TEMPLATES = _load()

DispositionType = Literal[
    "accuracy-oriented",
    "persuasion-oriented",
    "skeptical",
    "credulous",
    "neutral",
    "novelty-seeking",
]

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_agent_personas(
    n: int,
    disposition_mix: dict[str, float] | None = None,
    seed: int = 42,
    personas_file: pathlib.Path | None = None,
) -> list[AgentPersona]:
    """
    Build *n* AgentPersona instances by sampling from PERSONA_TEMPLATES.

    Parameters
    ----------
    n               : number of agents
    disposition_mix : fraction of agents per disposition type (must sum ~1.0).
                      Keys must match entries in data/personas.json "dispositions".
                      Defaults to all-neutral.
    seed            : for reproducible sampling
    personas_file   : override the default data/personas.json path
    """
    import random

    # Optionally reload from a different file
    if personas_file is not None:
        with personas_file.open() as f:
            raw = json.load(f)
        dispositions = raw["dispositions"]
        templates = raw["personas"]
    else:
        dispositions = DISPOSITION_DESCRIPTIONS
        templates = PERSONA_TEMPLATES

    rng = random.Random(seed)

    if disposition_mix is None:
        disposition_mix = {"neutral": 1.0}

    # Validate disposition keys
    unknown = set(disposition_mix) - set(dispositions)
    if unknown:
        raise ValueError(
            f"Unknown disposition(s): {unknown}. "
            f"Available: {set(dispositions)}"
        )

    # Normalize fractions
    total = sum(disposition_mix.values())
    disposition_mix = {k: v / total for k, v in disposition_mix.items()}

    # Build disposition list of length n
    disp_list: list[str] = []
    remaining = n
    items = list(disposition_mix.items())
    for i, (disp, frac) in enumerate(items):
        count = round(frac * n) if i < len(items) - 1 else remaining
        disp_list.extend([disp] * count)
        remaining -= count
    rng.shuffle(disp_list)

    personas = []
    for i in range(n):
        template = templates[i % len(templates)]
        disp_label = disp_list[i]
        # Append index suffix when wrapping around the template list
        name = (
            f"{template['name']} #{i // len(templates) + 1}"
            if i >= len(templates)
            else template["name"]
        )
        personas.append(AgentPersona(
            agent_id=f"agent_{i:03d}",
            name=name,
            background=template["background"],
            traits=template["traits"],
            disposition=dispositions[disp_label],
        ))

    return personas
