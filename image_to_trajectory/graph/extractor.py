# graph/extractor.py
"""
Stage 3: Convert skeleton pixels into a traversable NetworkX graph.

Key insight:
  Only endpoints (degree=1) and junctions (degree=3+) become graph NODES.
  All degree-2 pixels in between are "walked" as EDGES (with pixel chain stored).

This reduces a 1M-pixel skeleton to a graph of hundreds of nodes.
"""

import numpy as np
import networkx as nx
from collections import deque
from scipy.spatial import cKDTree
from image_to_trajectory.config import GraphConfig
from image_to_trajectory.preprocessing.skeletonize import pixel_degree


# 8-connected neighbor offsets (row_delta, col_delta)
NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),           ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def classify_pixels(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify every skeleton pixel by its topological role.

    Returns:
        endpoints: bool mask — degree == 1
        junctions: bool mask — degree >= 3
        interior:  bool mask — degree == 2
    """
    degree = pixel_degree(skeleton)
    endpoints = (degree == 1) & skeleton
    junctions = (degree >= 3) & skeleton
    interior  = (degree == 2) & skeleton
    return endpoints, junctions, interior


def walk_edge(
    start: tuple[int, int],
    came_from: tuple[int, int],
    skeleton: np.ndarray,
    node_mask: np.ndarray
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """
    Walk along degree-2 pixels from `start` until reaching another node.

    Args:
        start:      pixel to begin walking from (a non-node interior pixel)
        came_from:  the node we just left (to avoid backtracking)
        skeleton:   full skeleton bool array
        node_mask:  True where pixels are nodes (endpoint or junction)

    Returns:
        pixel_chain: ordered list of pixels traversed (not including came_from)
        end_node:    the node pixel where walk terminated
    """
    chain = [start]
    prev = came_from
    current = start

    while True:
        # Find neighbors of current that are skeleton pixels and not where we came from
        neighbors = []
        for dr, dc in NEIGHBORS_8:
            nr, nc = current[0] + dr, current[1] + dc
            if (nr, nc) != prev and skeleton[nr, nc]:
                neighbors.append((nr, nc))

        if not neighbors:
            # Dead end (shouldn't happen in a properly skeletonized image)
            break

        # Take the next step
        nxt = neighbors[0]
        chain.append(nxt)
        prev = current
        current = nxt

        # Stop when we hit another node
        if node_mask[current]:
            break

    return chain[:-1], current  # chain excludes the terminal node pixel


def edge_weight(pixel_chain: list[tuple[int, int]], mode: str) -> float:
    """
    Compute edge weight from pixel chain.

    'pixel_count': number of pixels (Manhattan-ish length)
    'euclidean':   straight-line distance between endpoints
    """
    if mode == "pixel_count" or len(pixel_chain) < 2:
        return float(len(pixel_chain))
    # Euclidean between first and last pixel in chain
    p0, p1 = pixel_chain[0], pixel_chain[-1]
    return float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))


def extract_graph(
    skeleton: np.ndarray,
    cfg: GraphConfig | None = None
) -> nx.Graph:
    """
    Convert skeleton to NetworkX graph.

    Node attributes:
        'pos': (row, col) pixel position

    Edge attributes:
        'pixels': list of (row, col) tuples along the edge path
        'weight': edge length
        'is_air': True for pen-lift edges (added later in traversal)

    Returns:
        Undirected graph G
    """
    cfg = cfg or GraphConfig()

    endpoints, junctions, _ = classify_pixels(skeleton)
    node_mask = endpoints | junctions

    G = nx.Graph()

    # --- Add all node pixels ---
    node_positions = np.argwhere(node_mask)
    for r, c in node_positions:
        G.add_node((r, c), pos=(r, c))

    # Special case: if skeleton has NO nodes (one pure cycle with no endpoints
    # or junctions), inject an artificial node to allow traversal
    if G.number_of_nodes() == 0:
        skel_pixels = np.argwhere(skeleton)
        if len(skel_pixels) > 0:
            seed = tuple(skel_pixels[0])
            G.add_node(seed, pos=seed)
            node_mask[seed] = True

    # --- Walk edges ---
    visited_edges: set[frozenset] = set()

    for node in list(G.nodes()):
        r, c = node
        for dr, dc in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1]):
                continue
            neighbor = (nr, nc)
            if not skeleton[neighbor]:
                continue

            if node_mask[neighbor]:
                # Direct node-to-node edge (adjacent nodes)
                edge_key = frozenset([node, neighbor])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    chain = [node, neighbor]
                    w = edge_weight(chain, cfg.edge_weight_mode)
                    G.add_edge(node, neighbor, pixels=chain, weight=w, is_air=False)
            else:
                # Start walking an interior chain
                chain, end_node = walk_edge(neighbor, node, skeleton, node_mask)
                edge_key = frozenset([node, end_node])
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    full_chain = [node] + chain + [end_node]
                    w = edge_weight(full_chain, cfg.edge_weight_mode)
                    G.add_edge(node, end_node,
                               pixels=full_chain, weight=w, is_air=False)

    return G


def merge_nearby_junctions(G: nx.Graph, radius: float) -> nx.Graph:
    """
    Merge graph nodes that are within `radius` pixels of each other.

    Near-duplicate junction pixels arise when two 3+ degree pixels are
    diagonally adjacent — they should be treated as one junction.

    Uses union-find via connected components on a proximity graph.
    """
    nodes = list(G.nodes())
    if not nodes:
        return G

    positions = np.array([(r, c) for r, c in nodes])
    tree = cKDTree(positions)

    # Find all pairs within radius
    pairs = tree.query_pairs(radius)

    # Build proximity graph and find connected components
    prox = nx.Graph()
    prox.add_nodes_from(range(len(nodes)))
    for i, j in pairs:
        prox.add_edge(i, j)

    # Map each node to its cluster representative (lowest index in component)
    mapping = {}
    for component in nx.connected_components(prox):
        rep_idx = min(component)
        rep_node = nodes[rep_idx]
        for idx in component:
            mapping[nodes[idx]] = rep_node

    return nx.relabel_nodes(G, mapping)

if __name__ == "__main__":
    """
    Standalone test runner for graph extraction.

    Usage:
        python -m image_to_trajectory.graph.extractor
    """

    import cv2
    import matplotlib.pyplot as plt
    from pathlib import Path

    from image_to_trajectory.preprocessing.binarize import binarize
    from image_to_trajectory.preprocessing.skeletonize import skeletonize_image

    # Input image path
    TEST_IMAGE = Path(__file__).parent.parent / "input.png"

    try:
        print("[INFO] Running binarization...")
        binary = binarize(TEST_IMAGE)

        print("[INFO] Running skeletonization...")
        skeleton = skeletonize_image(binary)

        print("[INFO] Extracting graph...")
        G = extract_graph(skeleton)

        print(f"[INFO] Nodes: {G.number_of_nodes()}")
        print(f"[INFO] Edges: {G.number_of_edges()}")

        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(10, 10))

        # Show skeleton background
        ax.imshow(skeleton, cmap="gray")

        # Draw graph edges
        for u, v, data in G.edges(data=True):
            pixels = data["pixels"]
            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            ax.plot(xs, ys, linewidth=1)

        # Draw graph nodes
        node_y = [n[0] for n in G.nodes()]
        node_x = [n[1] for n in G.nodes()]

        ax.scatter(node_x, node_y, s=20)

        ax.set_title(
            f"Extracted Graph\n"
            f"Nodes: {G.number_of_nodes()} | "
            f"Edges: {G.number_of_edges()}"
        )

        ax.invert_yaxis()
        plt.tight_layout()

        # Save visualization
        out_path = "debug_graph.png"
        plt.savefig(out_path, dpi=300)

        print(f"[INFO] Graph visualization saved: {out_path}")

        plt.show()

    except Exception as e:
        print(f"[ERROR] {e}")