# config.py
"""
Central configuration for all pipeline stages.
All magic numbers live here — never hardcoded in modules.
"""
from dataclasses import dataclass, field


@dataclass
class BinarizationConfig:
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    # Gaussian blur before threshold (kernel must be odd)
    blur_kernel_size: int = 3

    # Otsu threshold: invert so lines = white (255), background = black (0)
    invert: bool = True

    # Morphological cleanup after threshold
    morph_close_kernel: int = 2     # closes tiny gaps in lines
    morph_open_kernel: int = 1      # removes isolated noise dots


@dataclass
class SkeletonConfig:
    # Prune skeleton branches shorter than this (pixels)
    min_branch_length: int = 8

    # Remove connected components smaller than this (pixels)
    min_component_size: int = 20


@dataclass
class GraphConfig:
    # Merge junction nodes within this radius (pixels)
    junction_merge_radius: float = 3.0

    # Edge weights: 'euclidean' or 'pixel_count'
    edge_weight_mode: str = "euclidean"


@dataclass
class TraversalConfig:
    # Chinese Postman: max nearest-neighbors to consider per odd node
    # Lower = faster but suboptimal; higher = better but slower
    postman_k_neighbors: int = 20

    # TSP stitching between disconnected components: 'nearest' or 'greedy'
    component_stitch_mode: str = "nearest"


@dataclass
class SmoothingConfig:
    # RDP simplification epsilon (pixels) — higher = more simplification
    rdp_epsilon: float = 1.2

    # B-spline smoothing factor (0 = interpolating, higher = smoother)
    spline_smoothing: float = 0.0

    # Spline degree (3 = cubic)
    spline_degree: int = 3

    # Final resampling: target spacing between output points (pixels)
    resample_spacing: float = 2.0

    # Corners sharper than this angle (degrees) are pinned as hard knots
    corner_pin_angle: float = 80.0


@dataclass
class PipelineConfig:
    binarization: BinarizationConfig = field(default_factory=BinarizationConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    traversal: TraversalConfig = field(default_factory=TraversalConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)

    # Output resolution scaling (1.0 = original size)
    output_scale: float = 1.0

    # Save debug images at each stage
    debug: bool = False
    debug_dir: str = "debug_output"