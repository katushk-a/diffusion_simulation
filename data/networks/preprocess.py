"""
Preprocessing script for real-world network datasets.

Downloads a SNAP-format dataset (or reads a local file), extracts a
manageable subgraph via BFS from the highest-degree hub, and saves it
as GraphML for direct use with --network-file.

Usage
-----
    # Show available built-in datasets
    python data/networks/preprocess.py --list

    # Download + preprocess to 60 nodes (saves alongside this script)
    python data/networks/preprocess.py --dataset email-eu-core --n-nodes 60

    # Already downloaded file
    python data/networks/preprocess.py --file email-Eu-core.txt --n-nodes 60

    # Custom URL
    python data/networks/preprocess.py \\
        --url https://snap.stanford.edu/data/email-Eu-core.txt.gz \\
        --n-nodes 80 --output data/networks/my_net.graphml

Then run the simulation with:
    python main.py --network-file data/networks/email-eu-core_n60.graphml \\
                   --dataset data/sample_news.csv --backend ollama --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import pathlib
import sys
import urllib.request
from collections import deque

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known datasets
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "email-eu-core": {
        "url": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        "description": "EU research institution email network (1,005 nodes, 25,571 edges). "
                       "Directed. Good balance of size and community structure.",
        "format": "edgelist",
        "comment_char": "#",
    },
    "facebook": {
        "url": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        "description": "Facebook ego-networks combined (4,039 nodes, 88,234 edges). "
                       "Undirected. Converted to directed (both directions).",
        "format": "edgelist",
        "comment_char": "#",
    },
    "slashdot": {
        "url": "https://snap.stanford.edu/data/soc-Slashdot0811.txt.gz",
        "description": "Slashdot friend/foe social network (77,360 nodes). "
                       "Directed. Subsample recommended: --n-nodes 80.",
        "format": "edgelist",
        "comment_char": "#",
    },
    "epinions": {
        "url": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
        "description": "Epinions trust network (75,879 nodes, 508,837 edges). "
                       "Directed trust/distrust. Subsample heavily: --n-nodes 60.",
        "format": "edgelist",
        "comment_char": "#",
    },
    "wiki-vote": {
        "url": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
        "description": "Wikipedia admin election votes (7,115 nodes, 103,689 edges). "
                       "Directed. Good for studying influence propagation.",
        "format": "edgelist",
        "comment_char": "#",
    },
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Download *url* to *dest* if not already present. Returns dest."""
    if dest.exists():
        logger.info("Already downloaded: %s", dest.name)
        return dest
    logger.info("Downloading %s …", url)
    urllib.request.urlretrieve(url, dest)
    logger.info("Saved to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_snap_edgelist(path: pathlib.Path, comment_char: str = "#") -> nx.DiGraph:
    """
    Parse a SNAP-style edge list (optionally gzip-compressed).
    Lines starting with *comment_char* are skipped.
    """
    opener = gzip.open if path.suffix == ".gz" else open

    g = nx.DiGraph()
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(comment_char):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                u, v = parts[0], parts[1]
            g.add_edge(u, v)

    g.remove_edges_from(nx.selfloop_edges(g))
    logger.info(
        "Loaded graph: %d nodes, %d edges, directed=%s",
        g.number_of_nodes(), g.number_of_edges(), isinstance(g, nx.DiGraph),
    )
    return g


# ---------------------------------------------------------------------------
# Subsampling strategies
# ---------------------------------------------------------------------------

def _bfs_subgraph(g: nx.DiGraph, start: int, n_nodes: int, seed: int) -> nx.DiGraph:
    """
    BFS from *start*, collecting up to *n_nodes* nodes.
    Returns the induced subgraph.
    """
    import random
    rng = random.Random(seed)

    visited: list = [start]
    seen: set = {start}
    queue: deque = deque([start])

    while queue and len(visited) < n_nodes:
        node = queue.popleft()
        neighbours = list(g.successors(node)) + list(g.predecessors(node))
        rng.shuffle(neighbours)
        for nb in neighbours:
            if nb not in seen:
                seen.add(nb)
                visited.append(nb)
                queue.append(nb)
                if len(visited) >= n_nodes:
                    break

    sub = g.subgraph(visited).copy()
    logger.info(
        "BFS subgraph: %d nodes, %d edges (started from node %s)",
        sub.number_of_nodes(), sub.number_of_edges(), start,
    )
    return sub


def extract_subgraph(
    g: nx.DiGraph,
    n_nodes: int,
    method: str = "bfs-hub",
    seed: int = 42,
) -> nx.DiGraph:
    """
    Extract a connected subgraph of at most *n_nodes* nodes.

    Methods
    -------
    bfs-hub      BFS from the node with the highest out-degree (default).
                 Mimics ego-network of the most connected user.
    bfs-random   BFS from a random node.
    lscc         Use the largest strongly connected component, then BFS-hub
                 if it is still larger than n_nodes.
    lwcc         Use the largest weakly connected component, then BFS-hub
                 if it is still larger than n_nodes.
    """
    import random
    rng = random.Random(seed)

    if method in ("lscc", "lwcc"):
        if method == "lscc":
            components = sorted(
                nx.strongly_connected_components(g), key=len, reverse=True
            )
        else:
            components = sorted(
                nx.weakly_connected_components(g), key=len, reverse=True
            )
        if not components:
            raise ValueError("Graph has no connected components.")
        biggest = g.subgraph(components[0]).copy()
        logger.info(
            "%s size: %d nodes, %d edges",
            method.upper(), biggest.number_of_nodes(), biggest.number_of_edges(),
        )
        if biggest.number_of_nodes() <= n_nodes:
            return _relabel(biggest)
        # Still too big — fall back to bfs-hub within the component
        g = biggest
        method = "bfs-hub"

    # Determine start node
    if method == "bfs-hub":
        start = max(g.nodes, key=lambda n: g.out_degree(n))
        logger.info("Hub node: %s (out-degree=%d)", start, g.out_degree(start))
    elif method == "bfs-random":
        start = rng.choice(list(g.nodes))
        logger.info("Random start node: %s", start)
    else:
        raise ValueError(f"Unknown method: {method!r}. "
                         "Choose from: bfs-hub, bfs-random, lscc, lwcc")

    if g.number_of_nodes() <= n_nodes:
        logger.info("Graph already fits in %d nodes — no subsampling needed.", n_nodes)
        return _relabel(g)

    sub = _bfs_subgraph(g, start, n_nodes, seed)
    return _relabel(sub)


def _relabel(g: nx.DiGraph) -> nx.DiGraph:
    """Relabel nodes to consecutive integers 0..n-1 and return a clean DiGraph."""
    g = nx.convert_node_labels_to_integers(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and subsample a real-world network for the diffusion simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Show available built-in datasets and exit.",
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS),
        help="Name of a built-in dataset to download and process.",
    )
    parser.add_argument(
        "--url",
        help="Direct URL to a SNAP edge-list file (plain or .gz).",
    )
    parser.add_argument(
        "--file",
        help="Path to a locally downloaded edge-list file.",
    )
    parser.add_argument(
        "--n-nodes", type=int, default=60,
        help="Target number of nodes in the subgraph (default: 60).",
    )
    parser.add_argument(
        "--method",
        choices=["bfs-hub", "bfs-random", "lscc", "lwcc"],
        default="bfs-hub",
        help="Subsampling strategy (default: bfs-hub).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        help="Output GraphML path. Defaults to data/networks/<name>_n<N>.graphml.",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for name, info in DATASETS.items():
            print(f"  {name}")
            print(f"    {info['description']}")
            print(f"    URL: {info['url']}\n")
        return

    script_dir = pathlib.Path(__file__).parent

    # Resolve source
    if args.dataset:
        info = DATASETS[args.dataset]
        url = info["url"]
        raw_filename = url.split("/")[-1]
        raw_path = script_dir / raw_filename
        _download(url, raw_path)
        source_path = raw_path
        name = args.dataset
        comment_char = info.get("comment_char", "#")

    elif args.url:
        raw_filename = args.url.split("/")[-1]
        raw_path = script_dir / raw_filename
        _download(args.url, raw_path)
        source_path = raw_path
        name = raw_filename.split(".")[0]
        comment_char = "#"

    elif args.file:
        source_path = pathlib.Path(args.file)
        if not source_path.exists():
            logger.error("File not found: %s", source_path)
            sys.exit(1)
        name = source_path.stem.split(".")[0]
        comment_char = "#"

    else:
        parser.print_help()
        print("\nError: provide --dataset, --url, or --file.\n")
        sys.exit(1)

    # Load
    g = _load_snap_edgelist(source_path, comment_char=comment_char)

    # Convert undirected to directed if needed
    if not isinstance(g, nx.DiGraph):
        g = g.to_directed()
        logger.info("Converted to directed graph.")

    # Extract subgraph
    sub = extract_subgraph(g, n_nodes=args.n_nodes, method=args.method, seed=args.seed)

    # Print summary
    degrees = [d for _, d in sub.out_degree()]
    print(f"\n{'='*52}")
    print(f"  Subgraph summary")
    print(f"{'='*52}")
    print(f"  Nodes          : {sub.number_of_nodes()}")
    print(f"  Edges          : {sub.number_of_edges()}")
    print(f"  Density        : {nx.density(sub):.4f}")
    print(f"  Avg out-degree : {sum(degrees)/len(degrees):.2f}")
    print(f"  Max out-degree : {max(degrees)}")
    print(f"  Weakly conn.   : {nx.is_weakly_connected(sub)}")
    print(f"{'='*52}\n")

    # Save
    if args.output:
        out_path = pathlib.Path(args.output)
    else:
        out_path = script_dir / f"{name}_n{sub.number_of_nodes()}.graphml"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(sub, out_path)
    logger.info("Saved to %s", out_path)

    print(f"Run the simulation with:")
    print(f"  python main.py \\")
    print(f"    --network-file {out_path} \\")
    print(f"    --dataset data/sample_news.csv \\")
    print(f"    --backend mock --steps 6\n")


if __name__ == "__main__":
    main()
