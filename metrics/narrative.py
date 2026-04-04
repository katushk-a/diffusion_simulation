"""
Narrative evolution analysis.

Tracks how message content changes as it propagates through the network.

Two complementary measures:
  1. Embedding similarity  – semantic closeness to the original seed message
  2. Textual edit distance – character-level Levenshtein distance
     (normalized by max length)

Requires an LLMBackend capable of embedding (or sentence-transformers locally).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from simulation.message import Message
from simulation.runner import SimulationLog


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MessageNarrativeRecord:
    message_id: str
    cascade_id: str
    step: int
    sender_agent_id: str
    content: str
    similarity_to_seed: Optional[float] = None   # cosine similarity [0, 1]
    edit_distance_to_seed: Optional[float] = None  # normalized Levenshtein [0, 1]
    parent_similarity: Optional[float] = None    # similarity to direct parent


@dataclass
class CascadeNarrativeStats:
    cascade_id: str
    seed_content: str
    records: list[MessageNarrativeRecord]
    # Summary stats across non-seed messages
    mean_similarity_to_seed: float
    min_similarity_to_seed: float
    mean_edit_distance: float
    semantic_drift_per_step: float  # avg drop in similarity per hop


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NarrativeTracker:
    """
    Computes narrative evolution metrics for diffusion cascades.

    Usage:
        tracker = NarrativeTracker(llm_backend)
        stats = await tracker.analyze_cascade(log, cascade_id)
    """

    def __init__(self, llm) -> None:
        """llm: any object with async embed(texts: list[str]) -> list[list[float]]"""
        self.llm = llm

    async def analyze_cascade(
        self,
        log: SimulationLog,
        cascade_id: str,
    ) -> Optional[CascadeNarrativeStats]:
        messages = log.cascade_messages(cascade_id)
        if not messages:
            return None

        # Sort by step
        messages = sorted(messages, key=lambda m: m.step)

        # Identify seed(s) – step 0 or no parent
        seeds = [m for m in messages if m.is_seed()]
        if not seeds:
            seeds = [messages[0]]
        seed_content = seeds[0].content

        # Build id → message index for parent lookup
        msg_by_id = {m.id: m for m in messages}

        # Collect all texts for batch embedding
        all_texts = [seed_content] + [m.content for m in messages]
        embeddings = await self.llm.embed(all_texts)
        seed_emb = embeddings[0]
        msg_embeddings = {m.id: embeddings[i + 1] for i, m in enumerate(messages)}

        records = []
        for msg in messages:
            msg_emb = msg_embeddings[msg.id]
            sim_to_seed = _cosine_similarity(seed_emb, msg_emb)
            edit_to_seed = _normalized_levenshtein(seed_content, msg.content)

            # Parent similarity
            parent_sim = None
            if msg.parent_message_id and msg.parent_message_id in msg_embeddings:
                parent_emb = msg_embeddings[msg.parent_message_id]
                parent_sim = _cosine_similarity(parent_emb, msg_emb)

            records.append(
                MessageNarrativeRecord(
                    message_id=msg.id,
                    cascade_id=cascade_id,
                    step=msg.step,
                    sender_agent_id=msg.sender_agent_id,
                    content=msg.content,
                    similarity_to_seed=sim_to_seed,
                    edit_distance_to_seed=edit_to_seed,
                    parent_similarity=parent_sim,
                )
            )

        # Compute summary statistics (exclude seed messages from averages)
        non_seed = [r for r in records if r.similarity_to_seed is not None and r.step > 0]
        if not non_seed:
            mean_sim = 1.0
            min_sim = 1.0
            mean_edit = 0.0
            drift = 0.0
        else:
            sims = [r.similarity_to_seed for r in non_seed]
            edits = [r.edit_distance_to_seed for r in non_seed]
            mean_sim = sum(sims) / len(sims)
            min_sim = min(sims)
            mean_edit = sum(edits) / len(edits)

            # Drift: fit a simple slope of similarity vs step
            drift = _linear_slope(
                [(r.step, r.similarity_to_seed) for r in non_seed]
            )

        return CascadeNarrativeStats(
            cascade_id=cascade_id,
            seed_content=seed_content,
            records=records,
            mean_similarity_to_seed=mean_sim,
            min_similarity_to_seed=min_sim,
            mean_edit_distance=mean_edit,
            semantic_drift_per_step=drift,
        )

    async def analyze_all(
        self,
        log: SimulationLog,
    ) -> list[CascadeNarrativeStats]:
        cascade_ids = {m.cascade_id for m in log.messages}
        results = []
        for cid in cascade_ids:
            stats = await self.analyze_cascade(log, cid)
            if stats:
                results.append(stats)
        return results


# ---------------------------------------------------------------------------
# Agent-level narrative contribution
# ---------------------------------------------------------------------------

def agent_narrative_contributions(
    stats: CascadeNarrativeStats,
) -> list[dict]:
    """
    Rank agents by how much they mutated the message they forwarded.

    Uses parent_similarity from each MessageNarrativeRecord:
    a lower parent_similarity means the agent changed the content more.

    Returns a list of dicts sorted by mutation_score descending (most mutating first):
        {
          "agent_id":              str,
          "mean_parent_similarity": float,   # avg similarity to direct parent message
          "mutation_score":         float,   # 1 - mean_parent_similarity
          "n_messages":             int,     # number of messages sent by this agent
        }
    Only agents who actually forwarded (and have a measurable parent) are included.
    """
    from collections import defaultdict

    agent_sims: dict[str, list[float]] = defaultdict(list)
    for r in stats.records:
        if r.parent_similarity is not None and r.step > 0:
            agent_sims[r.sender_agent_id].append(r.parent_similarity)

    result = []
    for agent_id, sims in agent_sims.items():
        mean_sim = sum(sims) / len(sims)
        result.append({
            "agent_id": agent_id,
            "mean_parent_similarity": round(mean_sim, 4),
            "mutation_score": round(1.0 - mean_sim, 4),
            "n_messages": len(sims),
        })

    return sorted(result, key=lambda x: x["mutation_score"], reverse=True)


def top_mutating_agents(
    all_stats: list[CascadeNarrativeStats],
    top_n: int = 10,
) -> list[dict]:
    """
    Aggregate mutation scores across multiple cascades.

    Returns top_n agents ranked by mean mutation score across all cascades
    they participated in:
        {
          "agent_id":          str,
          "mean_mutation_score": float,
          "n_cascades":          int,
        }
    """
    from collections import defaultdict

    scores: dict[str, list[float]] = defaultdict(list)
    for stats in all_stats:
        for entry in agent_narrative_contributions(stats):
            scores[entry["agent_id"]].append(entry["mutation_score"])

    result = [
        {
            "agent_id": agent_id,
            "mean_mutation_score": round(sum(s) / len(s), 4),
            "n_cascades": len(s),
        }
        for agent_id, s in scores.items()
    ]
    return sorted(result, key=lambda x: x["mean_mutation_score"], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalized_levenshtein(s1: str, s2: str) -> float:
    """Levenshtein distance normalized by max(len(s1), len(s2))."""
    m, n = len(s1), len(s2)
    if m == 0 and n == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n] / max(m, n)


def _linear_slope(points: list[tuple[int, float]]) -> float:
    """Ordinary least-squares slope for (x, y) pairs."""
    n = len(points)
    if n < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den != 0 else 0.0
