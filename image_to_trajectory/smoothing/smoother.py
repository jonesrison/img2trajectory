# smoothing/smoother.py
"""
Stage 5: Smooth the jagged pixel-path into a curves suitable for robotic motion.

Pipeline:
  raw pixel coords
  → RDP simplification (reduce redundant collinear points)
  → detect and pin sharp corners (preserve intentional angles)
  → cubic B-spline interpolation
  → uniform arc-length resampling
"""

import numpy as np
from scipy.interpolate import splprep, splev
from image_to_trajectory.config import SmoothingConfig

def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker polyline simplification.

    Removes points that deviate less than epsilon from the simplified line.
    Recursive implementation — efficient for path lengths up to ~10k points.

    Args:
        points: (N, 2) array of [x, y]
        epsilon: max allowed deviation in pixels

    Returns:
        Simplified (M, 2) array, M <= N
    """
    if len(points) < 3:
        return points

    # Find point with maximum distance from line start→end
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        # Perpendicular distances from line
        t = np.dot(points - start, line_vec) / (line_len ** 2)
        projections = start + np.outer(t, line_vec)
        dists = np.linalg.norm(points - projections, axis=1)

    max_idx = np.argmax(dists)
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        # Recurse on both halves
        left  = rdp_simplify(points[:max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def compute_turn_angles(points: np.ndarray) -> np.ndarray:
    """
    Compute turning angle at each interior point (degrees).
    Returns array of length len(points), with 0 at endpoints.
    """
    angles = np.zeros(len(points))
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        angles[i] = np.degrees(np.arccos(cos_a))
    return angles


def find_corner_indices(points: np.ndarray, angle_threshold: float) -> list[int]:
    """
    Find indices of sharp corners that should be pinned during spline fitting.
    A corner is where the turning angle exceeds angle_threshold degrees.
    """
    angles = compute_turn_angles(points)
    # Turning angle > threshold means a sharp bend
    # (note: straight = 0°, U-turn = 180°)
    corners = [0]  # always pin start
    for i in range(1, len(points) - 1):
        if angles[i] > angle_threshold:
            corners.append(i)
    corners.append(len(points) - 1)  # always pin end
    return corners


def fit_spline_segment(
    points: np.ndarray,
    smoothing: float,
    degree: int
) -> tuple:
    """
    Fit a parametric B-spline through a segment of points.

    Returns (tck, u) — scipy spline representation.
    None if segment is too short to fit.
    """
    if len(points) < degree + 1:
        return None

    # Remove duplicate consecutive points (causes splprep to fail)
    diffs = np.diff(points, axis=0)
    mask = np.any(diffs != 0, axis=1)
    mask = np.concatenate([[True], mask])
    points = points[mask]

    if len(points) < degree + 1:
        return None

    try:
        tck, u = splprep(
            [points[:, 0], points[:, 1]],
            s=smoothing,
            k=degree,
            per=False
        )
        return tck, u
    except Exception:
        return None


def resample_uniform(coords: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a polyline at uniform arc-length intervals.

    Equal spacing ensures constant motor velocity for the robot.

    Args:
        coords:  (N, 2) array
        spacing: distance between output points in pixels

    Returns:
        (M, 2) array of uniformly spaced points
    """
    if len(coords) < 2:
        return coords

    # Compute cumulative arc length
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cum_lengths[-1]

    if total_length == 0:
        return coords[:1]

    # Create uniform parameter values
    num_points = max(2, int(total_length / spacing))
    uniform_t = np.linspace(0, total_length, num_points)

    # Interpolate x and y independently at uniform t values
    x_resampled = np.interp(uniform_t, cum_lengths, coords[:, 0])
    y_resampled = np.interp(uniform_t, cum_lengths, coords[:, 1])

    return np.column_stack([x_resampled, y_resampled])


def smooth_trajectory(
    raw_coords: list[tuple[float, float]],
    cfg: SmoothingConfig | None = None
) -> list[tuple[float, float]]:
    """
    Full smoothing pipeline.

    Args:
        raw_coords: List of (x, y) pixel coordinates from traversal

    Returns:
        Smoothed and resampled list of (x, y) coordinates
    """
    cfg = cfg or SmoothingConfig()

    if len(raw_coords) < 2:
        return raw_coords

    points = np.array(raw_coords, dtype=float)

    # Step 1: RDP simplification — remove redundant collinear pixels
    simplified = rdp_simplify(points, cfg.rdp_epsilon)

    if len(simplified) < 2:
        return raw_coords

    # Step 2: Find corners to preserve during spline fitting
    corner_idxs = find_corner_indices(simplified, cfg.corner_pin_angle)

    # Step 3: Fit splines segment-by-segment between corners
    all_smoothed = []

    for seg_start, seg_end in zip(corner_idxs[:-1], corner_idxs[1:]):
        segment = simplified[seg_start:seg_end + 1]

        if len(segment) < 2:
            all_smoothed.extend(segment.tolist())
            continue

        result = fit_spline_segment(segment, cfg.spline_smoothing, cfg.spline_degree)

        if result is None:
            all_smoothed.extend(segment.tolist())
            continue

        tck, u = result
        # Evaluate spline at dense parameter values
        n_eval = max(len(segment), int(
            np.linalg.norm(segment[-1] - segment[0]) * 2
        ))
        u_new = np.linspace(0, 1, n_eval)
        x_new, y_new = splev(u_new, tck)
        seg_points = np.column_stack([x_new, y_new])
        all_smoothed.extend(seg_points.tolist())

    if not all_smoothed:
        return raw_coords

    smoothed = np.array(all_smoothed)

    # Step 4: Uniform arc-length resampling for constant robot velocity
    resampled = resample_uniform(smoothed, cfg.resample_spacing)

    return [(float(x), float(y)) for x, y in resampled]

if __name__ == "__main__":
    """
    Standalone test runner for trajectory smoothing.

    Usage:
        python -m image_to_trajectory.smoothing.smoother
    """

    import matplotlib.pyplot as plt
    from pathlib import Path

    from image_to_trajectory.preprocessing.binarize import binarize
    from image_to_trajectory.preprocessing.skeletonize import skeletonize_image
    from image_to_trajectory.graph.extractor import extract_graph
    from image_to_trajectory.traversal.eulerize import make_eulerian
    from image_to_trajectory.traversal.traverser import extract_trajectory

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

        print("[INFO] Extracting raw trajectory...")
        raw_traj = extract_trajectory(MG)

        print(f"[INFO] Raw trajectory points: {len(raw_traj)}")

        print("[INFO] Smoothing trajectory...")
        smooth_traj = smooth_trajectory(raw_traj)

        print(f"[INFO] Smoothed trajectory points: {len(smooth_traj)}")

        # Convert to arrays for plotting
        raw = np.array(raw_traj)
        smooth = np.array(smooth_traj)

        # --- Visualization ---
        fig, ax = plt.subplots(figsize=(12, 12))

        # Background skeleton
        ax.imshow(skeleton, cmap="gray")

        # Raw trajectory
        if len(raw) > 0:
            ax.plot(
                raw[:, 0],
                raw[:, 1],
                linewidth=0.7,
                alpha=0.4,
                label="Raw Trajectory"
            )

        # Smoothed trajectory
        if len(smooth) > 0:
            ax.plot(
                smooth[:, 0],
                smooth[:, 1],
                linewidth=1.5,
                label="Smoothed Trajectory"
            )

            # Start / end markers
            ax.scatter(
                smooth[0, 0],
                smooth[0, 1],
                s=80
            )

            ax.scatter(
                smooth[-1, 0],
                smooth[-1, 1],
                s=80
            )

        ax.set_title(
            "Smoothed Robotic Trajectory\n"
            f"Raw: {len(raw_traj)} pts | "
            f"Smoothed: {len(smooth_traj)} pts"
        )

        ax.legend()

        ax.invert_yaxis()
        plt.tight_layout()

        # Save output
        out_path = "debug_smoothed_trajectory.png"
        plt.savefig(out_path, dpi=300)

        print(f"[INFO] Smoothed trajectory saved: {out_path}")

        plt.show()

    except Exception as e:
        print(f"[ERROR] {e}")