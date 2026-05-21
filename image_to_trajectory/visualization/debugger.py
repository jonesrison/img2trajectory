# visualization/debugger.py
"""
Debug visualizations for each pipeline stage.
All functions save images to disk and optionally display them.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from pathlib import Path


def save_binary(binary: np.ndarray, out_dir: Path, show: bool = False):
    path = out_dir / "01_binary.png"
    cv2.imwrite(str(path), binary)
    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(binary, cmap='gray')
        plt.title("Stage 1: Binarized Image")
        plt.axis('off')
        plt.show()
    print(f"[Debug] Binary → {path}")


def save_skeleton(skeleton: np.ndarray, out_dir: Path, show: bool = False):
    skel_img = (skeleton * 255).astype(np.uint8)
    path = out_dir / "02_skeleton.png"
    cv2.imwrite(str(path), skel_img)
    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(skel_img, cmap='gray')
        plt.title("Stage 2: Skeleton")
        plt.axis('off')
        plt.show()
    print(f"[Debug] Skeleton → {path}")


def save_graph(
    G: nx.Graph,
    image_shape: tuple[int, int],
    out_dir: Path,
    show: bool = False
):
    """Draw graph nodes and edges on a blank canvas."""
    h, w = image_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw edges
    for u, v, data in G.edges(data=True):
        pixels = data.get('pixels', [u, v])
        for i in range(len(pixels) - 1):
            r1, c1 = pixels[i]
            r2, c2 = pixels[i + 1]
            color = (0, 80, 200) if not data.get('is_air') else (80, 80, 80)
            cv2.line(canvas, (c1, r1), (c2, r2), color, 1)

    # Draw nodes
    for node in G.nodes():
        r, c = node
        deg = G.degree(node)
        if deg == 1:
            color = (0, 255, 0)   # green = endpoint
        elif deg >= 3:
            color = (0, 0, 255)   # red = junction
        else:
            color = (255, 255, 0) # yellow = other
        cv2.circle(canvas, (c, r), 3, color, -1)

    path = out_dir / "03_graph.png"
    cv2.imwrite(str(path), canvas)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title("Stage 3: Graph (green=endpoint, red=junction, blue=edge)")
        plt.axis('off')
        plt.show()
    print(f"[Debug] Graph → {path}")


def save_trajectory(
    trajectory: list[tuple[float, float]],
    image_shape: tuple[int, int],
    out_dir: Path,
    show: bool = False
):
    """Draw trajectory with color gradient showing draw order."""
    h, w = image_shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 240  # light grey background

    if not trajectory:
        return

    n = len(trajectory)
    colormap = cm.plasma

    for i in range(n - 1):
        x1, y1 = trajectory[i]
        x2, y2 = trajectory[i + 1]
        # Color by position in sequence (blue→red = start→end)
        t = i / max(n - 1, 1)
        r, g, b, _ = colormap(t)
        color = (int(b * 255), int(g * 255), int(r * 255))  # BGR
        cv2.line(canvas,
                 (int(x1), int(y1)),
                 (int(x2), int(y2)),
                 color, 1, cv2.LINE_AA)

    # Mark start (green circle) and end (red circle)
    sx, sy = int(trajectory[0][0]), int(trajectory[0][1])
    ex, ey = int(trajectory[-1][0]), int(trajectory[-1][1])
    cv2.circle(canvas, (sx, sy), 6, (0, 200, 0), -1)
    cv2.circle(canvas, (ex, ey), 6, (0, 0, 200), -1)

    path = out_dir / "05_trajectory.png"
    cv2.imwrite(str(path), canvas)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title("Final Trajectory (plasma color = draw order, green=start, red=end)")
        plt.axis('off')
        plt.colorbar(cm.ScalarMappable(cmap='plasma'), ax=plt.gca(),
                     label='Draw Order (0=start → 1=end)')
        plt.show()
    print(f"[Debug] Trajectory → {path}")


def save_all(
    binary, skeleton, G, trajectory, image_shape, out_dir_str: str, show: bool = False
):
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_binary(binary, out_dir, show)
    save_skeleton(skeleton, out_dir, show)
    save_graph(G, image_shape, out_dir, show)
    save_trajectory(trajectory, image_shape, out_dir, show)


if __name__ == "__main__":
    """
    Standalone debug visualization runner.

    Usage:
        python -m image_to_trajectory.visualization.debugger
    """

    from pathlib import Path

    from image_to_trajectory.preprocessing.binarize import binarize
    from image_to_trajectory.preprocessing.skeletonize import skeletonize_image
    from image_to_trajectory.graph.extractor import extract_graph
    from image_to_trajectory.traversal.eulerize import make_eulerian
    from image_to_trajectory.traversal.traverser import extract_trajectory
    from image_to_trajectory.smoothing.smoother import smooth_trajectory

    # Input image
    TEST_IMAGE = Path(__file__).parent.parent / "input.png"

    try:
        print("[INFO] Running full debug pipeline...")

        # Stage 1
        print("[INFO] Binarizing image...")
        binary = binarize(TEST_IMAGE)

        # Stage 2
        print("[INFO] Skeletonizing image...")
        skeleton = skeletonize_image(binary)

        # Stage 3
        print("[INFO] Extracting graph...")
        G = extract_graph(skeleton)

        # Stage 4a
        print("[INFO] Eulerizing graph...")
        MG = make_eulerian(G)

        # Stage 4b
        print("[INFO] Extracting trajectory...")
        raw_traj = extract_trajectory(MG)

        # Stage 5
        print("[INFO] Smoothing trajectory...")
        smooth_traj = smooth_trajectory(raw_traj)

        # Stage 6
        print("[INFO] Saving debug visualizations...")

        save_all(
            binary=binary,
            skeleton=skeleton,
            G=MG,
            trajectory=smooth_traj,
            image_shape=binary.shape,
            out_dir_str="debug_outputs",
            show=True
        )

        print("[SUCCESS] Full debug visualization complete.")
        print("[INFO] Outputs saved to: debug_outputs/")

    except Exception as e:
        print(f"[ERROR] {e}")