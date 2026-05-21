# traversal/eulerize.py
"""
Stage 4a: Make graph Eulerian so a single continuous path can traverse all edges.

The Chinese Postman Problem:
  - Find all odd-degree vertices
  - Compute minimum-weight perfect matching between them
  - Add "duplicate" edges along shortest paths between matched pairs
  - Result: all vertices have even degree → Eulerian circuit exists

For disconnected components:
  - Solve each component separately
  - Stitch components together with minimum-cost "air move" edges (pen lifts)
"""

import numpy as np
import networkx as nx
from itertools import combinations
from image_to_trajectory.config import TraversalConfig


def odd_degree_nodes(G: nx.Graph) -> list:
    """Return nodes with odd degree."""
    return [n for n, d in G.degree() if d % 2 == 1]


def shortest_path_lengths_subset(
    G: nx.Graph,
    sources: list,
    weight: str = "weight"
) -> dict:
    """
    Compute shortest path lengths from each source to all others.
    Returns dict: {source: {target: distance}}
    Skips unreachable pairs.
    """
    distances = {}
    for s in sources:
        try:
            lengths = nx.single_source_dijkstra_path_length(G, s, weight=weight)
            distances[s] = {t: lengths[t] for t in sources if t in lengths and t != s}
        except nx.NetworkXError:
            distances[s] = {}
    return distances


def greedy_min_weight_matching(
    odd_nodes: list,
    distances: dict,
    k: int
) -> list[tuple]:
    """
    Greedy approximation to minimum-weight perfect matching.

    True minimum matching (Blossom V) is O(n³) — prohibitive for large graphs.
    This greedy version:
      1. Sort all pairs by distance
      2. Greedily pick shortest pair where neither node is matched yet
    Quality: typically within 1.5x of optimal for spatial graphs.

    Args:
        odd_nodes: list of odd-degree node IDs
        distances: precomputed pairwise distances
        k: consider only k nearest neighbors per node

    Returns:
        List of (node_a, node_b) pairs to connect
    """
    # Build sorted candidate pairs
    pairs = []
    for i, u in enumerate(odd_nodes):
        # Get k nearest odd neighbors
        neighbors = sorted(
            [(distances.get(u, {}).get(v, float('inf')), v)
             for v in odd_nodes if v != u],
            key=lambda x: x[0]
        )[:k]
        for dist, v in neighbors:
            pairs.append((dist, u, v))

    pairs.sort()

    matched = set()
    result = []
    for dist, u, v in pairs:
        if u not in matched and v not in matched and dist < float('inf'):
            result.append((u, v))
            matched.add(u)
            matched.add(v)

    return result


def eulerize_component(G: nx.Graph, cfg: TraversalConfig) -> nx.MultiGraph:
    """
    Make a single connected graph component Eulerian by duplicating edges.

    Returns a MultiGraph (allows parallel edges for duplicated paths).
    """
    MG = nx.MultiGraph(G)

    odd_nodes = odd_degree_nodes(MG)
    if not odd_nodes:
        return MG  # already Eulerian

    distances = shortest_path_lengths_subset(MG, odd_nodes)
    pairs = greedy_min_weight_matching(odd_nodes, distances, cfg.postman_k_neighbors)

    for u, v in pairs:
        try:
            path = nx.shortest_path(G, u, v, weight="weight")
        except nx.NetworkXNoPath:
            continue

        # Duplicate edges along this path
        for a, b in zip(path[:-1], path[1:]):
            if G.has_edge(a, b):
                edge_data = dict(G[a][b])
                edge_data['is_air'] = False
                MG.add_edge(a, b, **edge_data)

    return MG


def stitch_components(
    components: list[nx.Graph],
    cfg: TraversalConfig
) -> nx.MultiGraph:
    """
    Connect disconnected graph components with minimum-cost air-move edges.

    Strategy: nearest-neighbor — find closest endpoint pairs across components
    and link them with zero-weight 'air' edges (pen lifts).

    Args:
        components: list of connected subgraphs

    Returns:
        Single connected MultiGraph
    """
    if len(components) == 1:
        return nx.MultiGraph(components[0])

    # Start with largest component, greedily absorb nearest others
    merged = nx.MultiGraph(components[0])
    remaining = list(components[1:])

    def component_endpoint_positions(comp):
        """Get positions of endpoint/leaf nodes (degree 1 in original)."""
        leaves = [n for n, d in comp.degree() if d == 1]
        if not leaves:
            leaves = list(comp.nodes())
        return leaves

    while remaining:
        best_dist = float('inf')
        best_comp = None
        best_pair = None

        merged_leaves = component_endpoint_positions(merged)

        for comp in remaining:
            comp_leaves = component_endpoint_positions(comp)

            # Find closest pair between merged and this component
            for u in merged_leaves:
                for v in comp_leaves:
                    r1, c1 = u
                    r2, c2 = v
                    dist = np.hypot(r2 - r1, c2 - c1)
                    if dist < best_dist:
                        best_dist = dist
                        best_comp = comp
                        best_pair = (u, v)

        # Add air-move edge between closest endpoints
        u, v = best_pair
        merged.add_nodes_from(best_comp.nodes(data=True))
        merged.add_edges_from(best_comp.edges(data=True))
        merged.add_edge(
            u, v,
            pixels=[u, v],
            weight=best_dist,
            is_air=True
        )

        remaining.remove(best_comp)

    return merged


def make_eulerian(G: nx.Graph, cfg: TraversalConfig | None = None) -> nx.MultiGraph:
    """
    Full eulerization pipeline.

    1. Split into connected components
    2. Eulerize each component (Chinese Postman)
    3. Stitch components together with air-move edges
    4. Return one Eulerian MultiGraph
    """
    cfg = cfg or TraversalConfig()

    # Remove isolated nodes (no edges)
    G = G.copy()
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)

    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    if not components:
        return nx.MultiGraph()

    eulerized = [eulerize_component(c, cfg) for c in components]
    stitched = stitch_components(eulerized, cfg)

    return stitched

if __name__ == "__main__":
    """
    Standalone test runner for Eulerization pipeline.

    Usage:
        python -m image_to_trajectory.traversal.eulerize
    """

    import matplotlib.pyplot as plt
    from pathlib import Path

    from image_to_trajectory.preprocessing.binarize import binarize
    from image_to_trajectory.preprocessing.skeletonize import skeletonize_image
    from image_to_trajectory.graph.extractor import extract_graph

    # Input image path
    TEST_IMAGE = Path(__file__).parent.parent / "input.png"

    try:
        print("[INFO] Running binarization...")
        binary = binarize(TEST_IMAGE)

        print("[INFO] Running skeletonization...")
        skeleton = skeletonize_image(binary)

        print("[INFO] Extracting graph...")
        G = extract_graph(skeleton)

        print(f"[INFO] Original graph:")
        print(f"        Nodes: {G.number_of_nodes()}")
        print(f"        Edges: {G.number_of_edges()}")

        odd_before = odd_degree_nodes(G)
        print(f"        Odd-degree nodes: {len(odd_before)}")

        print("[INFO] Eulerizing graph...")
        MG = make_eulerian(G)

        print(f"[INFO] Eulerized graph:")
        print(f"        Nodes: {MG.number_of_nodes()}")
        print(f"        Edges: {MG.number_of_edges()}")

        odd_after = odd_degree_nodes(MG)
        print(f"        Odd-degree nodes after: {len(odd_after)}")

        if len(odd_after) == 0:
            print("[SUCCESS] Graph is Eulerian!")
        else:
            print("[WARNING] Graph still has odd nodes.")

        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(12, 12))

        ax.imshow(skeleton, cmap="gray")

        # Draw edges
        for u, v, data in MG.edges(data=True):

            pixels = data.get("pixels", [u, v])

            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]

            # Air edges = dashed red
            if data.get("is_air", False):
                ax.plot(xs, ys,
                        linestyle="dashed",
                        linewidth=2)
            else:
                ax.plot(xs, ys, linewidth=1)

        # Draw nodes
        node_y = [n[0] for n in MG.nodes()]
        node_x = [n[1] for n in MG.nodes()]

        ax.scatter(node_x, node_y, s=10)

        ax.set_title(
            "Eulerized Graph\n"
            f"Nodes: {MG.number_of_nodes()} | "
            f"Edges: {MG.number_of_edges()} | "
            f"Odd After: {len(odd_after)}"
        )

        ax.invert_yaxis()
        plt.tight_layout()

        # Save output
        out_path = "debug_eulerized.png"
        plt.savefig(out_path, dpi=300)

        print(f"[INFO] Eulerized graph visualization saved: {out_path}")

        plt.show()

    except Exception as e:
        print(f"[ERROR] {e}")