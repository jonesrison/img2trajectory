# pipeline.py
"""
Main orchestrator — runs all stages in sequence.

Usage:
    python pipeline.py --input art.png --output output/trajectory.json

Or import and use programmatically:
    from pipeline import run_pipeline
    trajectory = run_pipeline("art.png")
"""

import argparse
import time
from pathlib import Path

from image_to_trajectory.config import PipelineConfig
from image_to_trajectory.preprocessing.binarize import binarize
from image_to_trajectory.preprocessing.skeletonize import skeletonize_image
from image_to_trajectory.graph.extractor import extract_graph, merge_nearby_junctions
from image_to_trajectory.traversal.eulerize import make_eulerian
from image_to_trajectory.traversal.traverser import extract_trajectory
from image_to_trajectory.smoothing.smoother import smooth_trajectory
from image_to_trajectory.exports.json_export import export_json
from image_to_trajectory.visualization.debugger import save_all


def run_pipeline(
    input_path: str,
    output_path: str = "output/trajectory.json",
    cfg: PipelineConfig | None = None,
    debug: bool = False
) -> list[tuple[float, float]]:
    """
    Execute the full image-to-trajectory pipeline.

    Args:
        input_path:  Path to input image (PNG, JPG, etc.)
        output_path: Path for JSON output
        cfg:         PipelineConfig (defaults used if None)
        debug:       Save debug images at each stage

    Returns:
        Ordered list of (x, y) coordinate tuples
    """
    cfg = cfg or PipelineConfig()
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Sand Art Trajectory Pipeline")
    print(f"  Input: {input_path}")
    print(f"{'='*60}\n")

    # ── Stage 1: Binarization ───────────────────────────────────────
    print("[1/6] Binarizing image...")
    t = time.time()
    binary = binarize(input_path, cfg.binarization)
    print(f"      {binary.shape[1]}×{binary.shape[0]}px | "
          f"{time.time()-t:.2f}s")

    # ── Stage 2: Skeletonization ────────────────────────────────────
    print("[2/6] Skeletonizing...")
    t = time.time()
    skeleton = skeletonize_image(binary, cfg.skeleton)
    skel_px = int(skeleton.sum())
    print(f"      {skel_px:,} skeleton pixels | {time.time()-t:.2f}s")

    # ── Stage 3: Graph Extraction ───────────────────────────────────
    print("[3/6] Extracting graph...")
    t = time.time()
    G = extract_graph(skeleton, cfg.graph)
    G = merge_nearby_junctions(G, cfg.graph.junction_merge_radius)
    print(f"      {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | "
          f"{time.time()-t:.2f}s")

    # ── Stage 4: Eulerization + Traversal ──────────────────────────
    print("[4/6] Eulerizing + traversing...")
    t = time.time()
    euler_graph = make_eulerian(G, cfg.traversal)
    raw_trajectory = extract_trajectory(euler_graph)
    print(f"      {len(raw_trajectory):,} raw waypoints | {time.time()-t:.2f}s")

    # ── Stage 5: Smoothing ──────────────────────────────────────────
    print("[5/6] Smoothing trajectory...")
    t = time.time()
    smooth = smooth_trajectory(raw_trajectory, cfg.smoothing)
    print(f"      {len(smooth):,} smoothed waypoints | {time.time()-t:.2f}s")

    # ── Stage 6: Export ─────────────────────────────────────────────
    print("[6/6] Exporting...")
    t = time.time()
    export_json(
        smooth,
        image_shape=binary.shape,
        output_path=output_path,
        metadata={"source_image": str(input_path)}
    )
    print(f"      Done | {time.time()-t:.2f}s")

    # ── Debug Visualizations ────────────────────────────────────────
    if debug or cfg.debug:
        print("\n[Debug] Saving visualizations...")
        save_all(binary, skeleton, G, smooth,
                 binary.shape, cfg.debug_dir, show=False)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Complete in {total:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    return smooth


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert image to sand art robot trajectory"
    )
    parser.add_argument("--input",  required=True, help="Input image path")
    parser.add_argument("--output", default="output/trajectory.json",
                        help="Output JSON path")
    parser.add_argument("--debug",  action="store_true",
                        help="Save debug images")
    parser.add_argument("--min-branch", type=int, default=8,
                        help="Minimum skeleton branch length (px)")
    parser.add_argument("--rdp-epsilon", type=float, default=1.2,
                        help="RDP simplification epsilon (px)")
    parser.add_argument("--spacing", type=float, default=2.0,
                        help="Output point spacing (px)")
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.skeleton.min_branch_length = args.min_branch
    cfg.smoothing.rdp_epsilon = args.rdp_epsilon
    cfg.smoothing.resample_spacing = args.spacing
    cfg.debug = args.debug

    run_pipeline(args.input, args.output, cfg, args.debug)