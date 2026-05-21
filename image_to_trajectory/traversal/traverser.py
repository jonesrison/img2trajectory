# traversal/traverser.py
"""
Stage 4b: Extract ordered Eulerian path from the eulerized graph.

Uses Hierholzer's algorithm — O(E) time.
Selects the starting node strategically (endpoint preference).
"""

import networkx as nx
import numpy as np
from typing import Generator


def best_start_node(G: nx.MultiGraph) -> object:
    """
    Choose the best starting node for Eulerian path traversal.

    Preference order:
    1. Odd-degree node (if graph has exactly 2, this is required for Euler PATH)
    2. Endpoint-like node (degree 1 in original — visually natural start)
    3. Any node

    After eulerization, graph should be fully even-degree (circuit).
    We still prefer a perceptually good starting point.
    """
    nodes = list(G.nodes())
    if not nodes:
        raise ValueError("Empty graph — no traversal possible")

    # Prefer nodes with degree 2 (original endpoints became degree-2 after
    # eulerization added one duplicate edge to them)
    degree_2 = [n for n in nodes if G.degree(n) == 2]
    if degree_2:
        # Among these, pick the one closest to image corner (top-left)
        return min(degree_2, key=lambda n: n[0] + n[1])

    return nodes[0]


def hierholzer_path(G: nx.MultiGraph, start: object) -> list:
    """
    Hierholzer's algorithm for Eulerian circuit on a MultiGraph.

    Finds a circuit that uses every edge exactly once.

    Returns:
        Ordered list of nodes forming the Eulerian circuit.
    """
    # Work on a copy to track which edges are consumed
    graph = {n: list(G[n].keys()) for n in G.nodes()}
    # For multigraph: track edge indices
    edge_tracker = {}
    for u in G.nodes():
        for v in G[u]:
            for key in G[u][v]:
                edge_tracker[(u, v, key)] = True

    # Use edge consumption tracking via remaining adjacency
    adj = {n: {} for n in G.nodes()}
    for u, v, key, data in G.edges(keys=True, data=True):
        if u not in adj[v]:
            adj[v][u] = []
        if v not in adj[u]:
            adj[u][v] = []
        adj[u][v].append(key)
        adj[v][u].append(key)

    stack = [start]
    path = []

    while stack:
        v = stack[-1]
        # Find any unvisited edge from v
        moved = False
        for u in list(adj[v].keys()):
            if adj[v][u]:
                key = adj[v][u].pop()
                # Remove the reverse direction too
                if key in adj[u].get(v, []):
                    adj[u][v].remove(key)
                stack.append(u)
                moved = True
                break
        if not moved:
            path.append(stack.pop())

    return path


def path_to_coordinates(
    node_path: list,
    G: nx.MultiGraph
) -> list[tuple[float, float]]:
    """
    Convert ordered node list to coordinate sequence.

    For each consecutive node pair, look up the edge's pixel chain
    and expand it into individual coordinates.

    This preserves the actual curved path of the skeleton edge,
    not just straight lines between junction nodes.

    Returns:
        List of (col, row) = (x, y) tuples in image space.
        Note: col=x, row=y is the standard image coordinate convention.
    """
    if not node_path:
        return []

    coords = []

    for i in range(len(node_path) - 1):
        u = node_path[i]
        v = node_path[i + 1]

        # Find the edge between u and v (first available in multigraph)
        edge_data = None
        if G.has_edge(u, v):
            # Get first edge (multigraph may have duplicates)
            for key in G[u][v]:
                edge_data = G[u][v][key]
                break

        if edge_data and 'pixels' in edge_data:
            pixels = edge_data['pixels']
            # Determine direction: does the pixel chain go u→v or v→u?
            if pixels and pixels[0] == u:
                ordered_pixels = pixels
            else:
                ordered_pixels = list(reversed(pixels))

            for r, c in ordered_pixels:
                # Convert to (x=col, y=row)
                coords.append((float(c), float(r)))
        else:
            # Fallback: straight line between nodes
            coords.append((float(u[1]), float(u[0])))
            coords.append((float(v[1]), float(v[0])))

    # Add final node
    if node_path:
        last = node_path[-1]
        coords.append((float(last[1]), float(last[0])))

    return coords


def extract_trajectory(G: nx.MultiGraph) -> list[tuple[float, float]]:
    """
    Full traversal pipeline.

    Returns:
        Ordered list of (x, y) pixel coordinates.
    """
    if G.number_of_edges() == 0:
        return []

    start = best_start_node(G)
    node_path = hierholzer_path(G, start)
    coords = path_to_coordinates(node_path, G)
    return coords


if __name__ == "__main__":
    """
    Standalone test runner for trajectory extraction.

    Usage:
        python -m image_to_trajectory.traversal.traverser
    """

    import matplotlib.pyplot as plt
    from pathlib import Path

    from image_to_trajectory.preprocessing.binarize import binarize
    from image_to_trajectory.preprocessing.skeletonize import skeletonize_image
    from image_to_trajectory.graph.extractor import extract_graph
    from image_to_trajectory.traversal.eulerize import make_eulerian

    # Input image
    TEST_IMAGE = Path(__file__).parent.parent / "input.png"

    try:
        print("[INFO] Running binarization...")
        binary = binarize(TEST_IMAGE)

        print("[INFO] Running skeletonization...")
        skeleton = skeletonize_image(binary)

        print("[INFO] Extracting graph...")
        G = extract_graph(skeleton)

        print("[INFO] Eulerizing graph...")
        MG = make_eulerian(G)

        print("[INFO] Extracting trajectory...")
        trajectory = extract_trajectory(MG)

        print(f"[INFO] Total trajectory points: {len(trajectory)}")

        if len(trajectory) > 0:
            print(f"[INFO] First point: {trajectory[0]}")
            print(f"[INFO] Last point: {trajectory[-1]}")

        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(12, 12))

        # Background skeleton
        ax.imshow(skeleton, cmap="gray")

        # Draw trajectory
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        ax.plot(xs, ys, linewidth=1)

        # Mark start/end
        if trajectory:
            ax.scatter(xs[0], ys[0], s=60)
            ax.scatter(xs[-1], ys[-1], s=60)

        ax.set_title(
            "Extracted Continuous Trajectory\n"
            f"Points: {len(trajectory)}"
        )

        ax.invert_yaxis()
        plt.tight_layout()

        # Save visualization
        out_path = "debug_trajectory.png"
        plt.savefig(out_path, dpi=300)

        print(f"[INFO] Trajectory visualization saved: {out_path}")

        plt.show()

    except Exception as e:
        print(f"[ERROR] {e}")