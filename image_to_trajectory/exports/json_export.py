# exports/json_export.py
"""
Stage 6: Export trajectory to JSON and compute path statistics.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def compute_statistics(
    trajectory: list[tuple[float, float]],
    image_shape: tuple[int, int]
) -> dict:
    """
    Compute path statistics useful for hardware planning.

    Args:
        trajectory: list of (x, y) points
        image_shape: (height, width) of original image

    Returns:
        Statistics dictionary
    """
    if not trajectory:
        return {}

    points = np.array(trajectory)
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    total_length = float(np.sum(segment_lengths))
    num_points = len(trajectory)

    # Bounding box
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    # Curvature estimate (direction changes per unit length)
    if len(diffs) > 1:
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        angle_changes = np.abs(np.diff(np.unwrap(angles)))
        mean_curvature = float(np.mean(angle_changes))
        max_curvature = float(np.max(angle_changes))
    else:
        mean_curvature = 0.0
        max_curvature = 0.0

    # Draw complexity: ratio of path length to image diagonal
    img_diagonal = float(np.hypot(*image_shape))
    complexity = total_length / max(img_diagonal, 1)

    return {
        "num_points": num_points,
        "total_length_px": round(total_length, 2),
        "image_coverage": {
            "x_range": [round(float(x_min), 2), round(float(x_max), 2)],
            "y_range": [round(float(y_min), 2), round(float(y_max), 2)],
        },
        "bounding_box": {
            "width": round(float(x_max - x_min), 2),
            "height": round(float(y_max - y_min), 2),
        },
        "curvature": {
            "mean_rad": round(mean_curvature, 4),
            "max_rad": round(max_curvature, 4),
        },
        "draw_complexity_ratio": round(complexity, 3),
        "image_shape": {"height": image_shape[0], "width": image_shape[1]},
    }


def export_json(
    trajectory: list[tuple[float, float]],
    image_shape: tuple[int, int],
    output_path: str | Path,
    metadata: dict | None = None
) -> dict:
    """
    Export trajectory to JSON file with statistics.

    Output format:
    {
        "metadata": { ... },
        "statistics": { ... },
        "path": [{"x": 120.5, "y": 50.2}, ...]
    }
    """
    stats = compute_statistics(trajectory, image_shape)

    output = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "coordinate_system": "image_pixels_xy",
            "origin": "top_left",
            **(metadata or {})
        },
        "statistics": stats,
        "path": [{"x": round(x, 3), "y": round(y, 3)} for x, y in trajectory]
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Export] Saved {len(trajectory)} points → {output_path}")
    print(
        f"[Export] Total length: "
        f"{stats.get('total_length_px', 0):.1f}px | "
        f"Complexity: "
        f"{stats.get('draw_complexity_ratio', 0):.3f}"
    )

    return output

if __name__ == "__main__":
    """
    Standalone test runner for JSON export pipeline.

    Usage:
        python -m image_to_trajectory.exports.json_export
    """

    import cv2
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
        print("[INFO] Running binarization...")
        binary = binarize(TEST_IMAGE)

        print("[INFO] Running skeletonization...")
        skeleton = skeletonize_image(binary)

        print("[INFO] Extracting graph...")
        G = extract_graph(skeleton)

        print("[INFO] Eulerizing graph...")
        MG = make_eulerian(G)

        print("[INFO] Extracting trajectory...")
        raw_traj = extract_trajectory(MG)

        print("[INFO] Smoothing trajectory...")
        smooth_traj = smooth_trajectory(raw_traj)

        # Output path
        output_path = Path("output/trajectory.json")

        print("[INFO] Exporting JSON...")

        result = export_json(
            trajectory=smooth_traj,
            image_shape=binary.shape,
            output_path=output_path,
            metadata={
                "source_image": str(TEST_IMAGE.name),
                "pipeline": "image_to_trajectory",
                "smoothed": True
            }
        )

        print(f"[Export] Saved {len(smooth_traj)} points → {output_path}")

        stats = result.get("statistics", {})

        print(
            f"[Export] Total length: "
            f"{stats.get('total_length_px', 0):.1f}px | "
            f"Complexity: "
            f"{stats.get('draw_complexity_ratio', 0):.3f}"
        )

        print("[INFO] Export complete.")

    except Exception as e:
        print(f"[ERROR] {e}")