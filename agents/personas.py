"""
Persona loading for the diffusion simulation.

All persona templates live in:
    data/personas.json

This module loads them at import time and exposes:
  - PERSONA_TEMPLATES      list[dict]  (name, background, traits, epistemic_profile)
  - build_agent_personas() main factory used by ExperimentRunner
"""

from __future__ import annotations

import json
import pathlib

from agents.base import AgentPersona

# ---------------------------------------------------------------------------
# Load data file
# ---------------------------------------------------------------------------

_DATA_FILE = pathlib.Path(__file__).parent.parent / "data" / "personas.json"


def _load() -> list[dict]:
    with _DATA_FILE.open() as f:
        raw = json.load(f)
    return raw["personas"]


PERSONA_TEMPLATES: list[dict] = _load()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_agent_personas(
    n: int,
    seed: int = 42,
    personas_file: pathlib.Path | None = None,
    persona_mix: dict[str, float] | None = None,
) -> list[AgentPersona]:
    """
    Build *n* AgentPersona instances from PERSONA_TEMPLATES.

    Parameters
    ----------
    n            : number of agents
    seed         : random seed used when persona_mix is specified
    personas_file: override the default data/personas.json path
    persona_mix  : optional dict mapping epistemic_type → fraction of agents.
                   e.g. {"open": 0.5, "closed": 0.5}
                   When None, all 25 templates are cycled in order.
                   Available types: "open", "closed", "credulous", "strategic"
    """
    import random

    if personas_file is not None:
        with personas_file.open() as f:
            raw = json.load(f)
        templates = raw["personas"]
    else:
        templates = PERSONA_TEMPLATES

    if persona_mix is None:
        return _build_cycling(n, templates)
    return _build_mixed(n, templates, persona_mix, seed)  # seed used inside _build_mixed


def _build_cycling(n: int, templates: list[dict]) -> list[AgentPersona]:
    """Default: cycle through all templates in order."""
    personas = []
    for i in range(n):
        template = templates[i % len(templates)]
        name = (
            f"{template['name']} #{i // len(templates) + 1}"
            if i >= len(templates)
            else template["name"]
        )
        personas.append(_make_persona(f"agent_{i:03d}", name, template))
    return personas


def _build_mixed(
    n: int,
    templates: list[dict],
    persona_mix: dict[str, float],
    seed: int,
) -> list[AgentPersona]:
    """Composition-controlled: draw proportionally from epistemic-type pools."""
    import random  # noqa: PLC0415

    available_types = {t["epistemic_type"] for t in templates}
    unknown = set(persona_mix) - available_types
    if unknown:
        raise ValueError(
            f"Unknown epistemic_type(s): {unknown}. "
            f"Available: {available_types}"
        )

    # Normalize fractions
    total = sum(persona_mix.values())
    mix = {k: v / total for k, v in persona_mix.items()}

    # Build a flat list of n type labels
    type_list: list[str] = []
    remaining = n
    items = list(mix.items())
    for i, (ep_type, frac) in enumerate(items):
        count = round(frac * n) if i < len(items) - 1 else remaining
        type_list.extend([ep_type] * count)
        remaining -= count
    random.Random(seed).shuffle(type_list)

    # Pool of templates per type
    pools: dict[str, list[dict]] = {
        ep_type: [t for t in templates if t["epistemic_type"] == ep_type]
        for ep_type in mix
    }
    counters: dict[str, int] = {ep_type: 0 for ep_type in mix}

    personas = []
    for i, ep_type in enumerate(type_list):
        pool = pools[ep_type]
        idx = counters[ep_type] % len(pool)
        template = pool[idx]
        wrap = counters[ep_type] // len(pool) + 1
        name = f"{template['name']} #{wrap}" if counters[ep_type] >= len(pool) else template["name"]
        counters[ep_type] += 1
        personas.append(_make_persona(f"agent_{i:03d}", name, template))

    return personas


def _make_persona(agent_id: str, name: str, template: dict) -> AgentPersona:
    return AgentPersona(
        agent_id=agent_id,
        name=name,
        epistemic_type=template["epistemic_type"],
        background=template["background"],
        epistemic_profile=template["epistemic_profile"],
        communication_style=template["communication_style"],
        information_behavior=template["information_behavior"],
    )
