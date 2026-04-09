"""
NetworkX graph factories for the diffusion simulation.

Three standard network types used in social network research:
  - Random (Erdős–Rényi)
  - Scale-free (Barabási–Albert)  – hubs + long tail, closest to real social nets
  - Small-world (Watts–Strogatz)  – high clustering + short paths
"""

from __future__ import annotations

import pathlib
from typing import Literal

import networkx as nx

NetworkType = Literal["random", "scale_free", "small_world"]


def build_graph(
    n: int,
    network_type: NetworkType,
    seed: int = 42,
    **kwargs,
) -> nx.DiGraph:
    """
    Build a directed graph with *n* nodes.

    Random:       p=0.15 (edge probability)
    Scale-free:   m=2 (edges per new node)
    Small-world:  k=4 (neighbors), p=0.1 (rewiring probability)

    All defaults can be overridden via **kwargs.
    """
    match network_type:
        case "random":
            p = kwargs.get("p", 0.15)
            g = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)

        case "scale_free":
            m = kwargs.get("m", 2)
            # Barabási–Albert preferential attachment (undirected → directed)
            # m = number of edges each new node attaches to; higher = denser
            ug = nx.barabasi_albert_graph(n, m, seed=seed)
            g = ug.to_directed()

        case "small_world":
            k = kwargs.get("k", 4)
            p = kwargs.get("p", 0.1)
            # Watts–Strogatz is undirected; convert to directed (both directions)
            ug = nx.watts_strogatz_graph(n, k, p, seed=seed)
            g = ug.to_directed()

        case _:
            raise ValueError(f"Unknown network_type: {network_type!r}")

    # Ensure nodes are labeled 0..n-1
    g = nx.convert_node_labels_to_integers(g)
    # Remove self-loops
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def load_graph_from_file(
    path: str | pathlib.Path,
    directed: bool = True,
) -> nx.DiGraph:
    """
    Load a real-world network from a file.

    Supported formats (detected by file extension):
      .graphml / .xml   – GraphML
      .gml              – GML
      .gexf             – GEXF
      .adjlist          – adjacency list
      anything else     – edge list (whitespace- or comma-separated)

    The loaded graph is always returned as a DiGraph (bidirectional edges
    for undirected sources). Self-loops are removed.
    """
    path = pathlib.Path(path)
    ext = path.suffix.lower()

    if ext in (".graphml", ".xml"):
        g = nx.read_graphml(path)
    elif ext == ".gml":
        g = nx.read_gml(path)
    elif ext == ".gexf":
        g = nx.read_gexf(path)
    elif ext == ".adjlist":
        g = nx.read_adjlist(path)
    else:
        # Edge list: try comma-delimited first, fall back to whitespace
        try:
            g = nx.read_edgelist(path, delimiter=",")
        except Exception:
            g = nx.read_edgelist(path)

    if directed and not isinstance(g, nx.DiGraph):
        g = g.to_directed()
    elif not directed and not isinstance(g, nx.Graph):
        g = g.to_undirected()

    g = nx.convert_node_labels_to_integers(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def assign_agents_to_graph(
    graph: nx.DiGraph,
    agent_ids: list[str],
) -> nx.DiGraph:
    """
    Attach agent IDs to graph nodes as 'agent_id' attribute.
    Nodes are matched by position (node 0 → agent_ids[0]).
    """
    if len(agent_ids) != graph.number_of_nodes():
        raise ValueError(
            f"Graph has {graph.number_of_nodes()} nodes but "
            f"{len(agent_ids)} agent IDs provided."
        )
    mapping = {i: aid for i, aid in enumerate(agent_ids)}
    nx.set_node_attributes(graph, mapping, "agent_id")
    return graph


def get_neighbors(graph: nx.DiGraph, agent_id: str) -> list[str]:
    """
    Return the agent IDs of all out-neighbors of *agent_id*.
    Uses the 'agent_id' node attribute to look up and return neighbors.
    """
    # Build reverse map once: agent_id → node index
    rev = {data["agent_id"]: node for node, data in graph.nodes(data=True)}
    node = rev[agent_id]
    return [graph.nodes[nb]["agent_id"] for nb in graph.successors(node)]


def graph_summary(graph: nx.DiGraph) -> dict:
    """Return basic structural statistics."""
    degrees = [d for _, d in graph.out_degree()]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_out_degree": sum(degrees) / len(degrees) if degrees else 0,
        "max_out_degree": max(degrees) if degrees else 0,
        "is_weakly_connected": nx.is_weakly_connected(graph),
        "num_weakly_connected_components": nx.number_weakly_connected_components(graph),
    }
