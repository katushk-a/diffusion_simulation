"""
Cascade structure analysis.

Given the message log from a simulation, this module reconstructs
diffusion cascades as trees and computes structural metrics.

Metrics implemented:
  - size:               total number of unique agents that forwarded
  - depth:              longest path from root to leaf (in hops)
  - max_breadth:        maximum number of nodes at any single depth level
  - structural_virality: average distance between all pairs of nodes (Goel et al. 2016)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx

from simulation.message import Message
from simulation.runner import SimulationLog


# ---------------------------------------------------------------------------
# Cascade tree
# ---------------------------------------------------------------------------

def build_cascade_tree(
    messages: list[Message],
) -> nx.DiGraph:
    """
    Build a directed tree (parent → child) from a list of messages
    belonging to the same cascade.

    Nodes are message IDs. Node attribute 'agent_id' is the sender.
    """
    g = nx.DiGraph()
    for msg in messages:
        g.add_node(msg.id, agent_id=msg.sender_agent_id, step=msg.step, content=msg.content)
        if msg.parent_message_id:
            g.add_edge(msg.parent_message_id, msg.id)
    return g


# ---------------------------------------------------------------------------
# Metric dataclass
# ---------------------------------------------------------------------------

@dataclass
class CascadeMetrics:
    cascade_id: str
    size: int                        # nodes in the cascade tree (excl. root)
    depth: int                       # longest root-to-leaf path
    max_breadth: int                 # max nodes at any single step
    structural_virality: float       # avg pairwise shortest path length
    unique_forwarders: int           # unique agent IDs that forwarded
    step_distribution: dict[int, int]  # step → count of messages at that step
    received_count: int = 0          # agents who received the message
    forwarded_count: int = 0         # agents who chose to forward
    adoption_rate: float = 0.0       # forwarded_count / received_count
    label: str = ""                  # content label inherited from seed ("true"/"fake"/...)


def compute_cascade_metrics(
    log: SimulationLog,
    cascade_id: str,
) -> Optional[CascadeMetrics]:
    """Compute CascadeMetrics for one cascade from the simulation log."""
    messages = log.cascade_messages(cascade_id)
    if not messages:
        return None

    tree = build_cascade_tree(messages)

    # Size = non-root nodes
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    size = tree.number_of_nodes() - len(roots)

    # Depth
    depth = 0
    if roots:
        # Use longest shortest path from any root
        for root in roots:
            lengths = nx.single_source_shortest_path_length(tree, root)
            depth = max(depth, max(lengths.values(), default=0))

    # Step distribution and breadth
    step_dist: dict[int, int] = {}
    for msg in messages:
        step_dist[msg.step] = step_dist.get(msg.step, 0) + 1
    max_breadth = max(step_dist.values(), default=0)

    # Structural virality (Goel et al. 2016)
    # = average of all pairwise shortest path lengths in the undirected tree
    sv = _structural_virality(tree)

    unique_forwarders = len({msg.sender_agent_id for msg in messages})

    # Label: take from any seed message in this cascade
    label = next((m.label for m in messages if m.is_seed()), "")

    # Adoption rate: count "received" and "forwarded" events for this cascade
    cascade_events = [e for e in log.events if e.cascade_id == cascade_id]
    received_count = sum(1 for e in cascade_events if e.event_type == "received")
    forwarded_count = sum(1 for e in cascade_events if e.event_type == "forwarded")
    adoption_rate = forwarded_count / received_count if received_count > 0 else 0.0

    return CascadeMetrics(
        cascade_id=cascade_id,
        size=size,
        depth=depth,
        max_breadth=max_breadth,
        structural_virality=sv,
        unique_forwarders=unique_forwarders,
        step_distribution=step_dist,
        received_count=received_count,
        forwarded_count=forwarded_count,
        adoption_rate=adoption_rate,
        label=label,
    )


def compute_all_cascades(log: SimulationLog) -> list[CascadeMetrics]:
    """Compute metrics for every cascade in the log."""
    cascade_ids = {m.cascade_id for m in log.messages}
    results = []
    for cid in cascade_ids:
        m = compute_cascade_metrics(log, cid)
        if m:
            results.append(m)
    return results


# ---------------------------------------------------------------------------
# Structural virality helper
# ---------------------------------------------------------------------------

def compute_adoption_by_disposition(
    log: SimulationLog,
    agent_dispositions: dict[str, str],
) -> dict[str, dict]:
    """
    Compute adoption rate broken down by agent disposition group.

    Parameters
    ----------
    log:                 SimulationLog from a completed simulation run
    agent_dispositions:  mapping of agent_id → disposition label
                         (e.g. "skeptical", "credulous", "neutral")

    Returns
    -------
    dict keyed by disposition label:
        {
          "received":  int,
          "forwarded": int,
          "adoption_rate": float,
        }
    """
    received: dict[str, int] = {}
    forwarded: dict[str, int] = {}

    for event in log.events:
        agent_id = event.agent_id
        disposition = agent_dispositions.get(agent_id, "unknown")

        if event.event_type == "received":
            received[disposition] = received.get(disposition, 0) + 1
        elif event.event_type == "forwarded":
            forwarded[disposition] = forwarded.get(disposition, 0) + 1

    all_dispositions = set(received) | set(forwarded)
    result = {}
    for disp in sorted(all_dispositions):
        r = received.get(disp, 0)
        f = forwarded.get(disp, 0)
        result[disp] = {
            "received": r,
            "forwarded": f,
            "adoption_rate": f / r if r > 0 else 0.0,
        }
    return result


def compare_by_label(
    cascade_metrics: list[CascadeMetrics],
) -> dict[str, dict]:
    """
    Aggregate cascade metrics by content label (true / fake / misleading).

    Returns a dict keyed by label:
        {
          "n_cascades": int,
          "mean_size": float,
          "mean_depth": float,
          "mean_max_breadth": float,
          "mean_structural_virality": float,
          "mean_adoption_rate": float,
          "mean_unique_forwarders": float,
        }
    """
    from collections import defaultdict

    groups: dict[str, list[CascadeMetrics]] = defaultdict(list)
    for cm in cascade_metrics:
        groups[cm.label].append(cm)

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    result = {}
    for label, group in sorted(groups.items()):
        result[label] = {
            "n_cascades": len(group),
            "mean_size": _mean([g.size for g in group]),
            "mean_depth": _mean([g.depth for g in group]),
            "mean_max_breadth": _mean([g.max_breadth for g in group]),
            "mean_structural_virality": _mean([g.structural_virality for g in group]),
            "mean_adoption_rate": _mean([g.adoption_rate for g in group]),
            "mean_unique_forwarders": _mean([g.unique_forwarders for g in group]),
        }
    return result


def _structural_virality(tree: nx.DiGraph) -> float:
    """
    Structural virality = (1 / (n*(n-1))) * sum of all pairwise distances.
    Uses the undirected version of the tree.
    Returns 0.0 for trees with fewer than 2 nodes.
    """
    n = tree.number_of_nodes()
    if n < 2:
        return 0.0

    undirected = tree.to_undirected()
    if not nx.is_connected(undirected):
        # Handle disconnected: average over connected pairs only
        total, count = 0.0, 0
        for component in nx.connected_components(undirected):
            sub = undirected.subgraph(component)
            m = sub.number_of_nodes()
            if m < 2:
                continue
            lengths = dict(nx.all_pairs_shortest_path_length(sub))
            for u in lengths:
                for v, d in lengths[u].items():
                    if u != v:
                        total += d
                        count += 1
        return total / count if count else 0.0

    lengths = dict(nx.all_pairs_shortest_path_length(undirected))
    total = sum(
        d
        for u in lengths
        for v, d in lengths[u].items()
        if u != v
    )
    return total / (n * (n - 1))
