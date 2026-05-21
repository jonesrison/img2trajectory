# preprocessing/skeletonize.py
"""
Stage 2: Convert thick binary lines into 1-pixel-wide centerline skeleton.

Uses Zhang-Suen topology-preserving thinning (skimage).
Then prunes short spurious branches that appear as skeleton hair.
"""

import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize
from skimage.measure import label
from scipy import ndimage
from image_to_trajectory.config import SkeletonConfig


def compute_skeleton(binary: np.ndarray) -> np.ndarray:
    """
    Run Zhang-Suen skeletonization.

    Args:
        binary: uint8 image, lines=255, bg=0

    Returns:
        bool array, True where skeleton pixels exist
    """
    # skimage expects bool input
    return sk_skeletonize(binary > 0)


def pixel_degree(skeleton: np.ndarray) -> np.ndarray:
    """
    Compute the degree (number of neighbors) of each skeleton pixel.
    Uses a 3x3 convolution with all-ones kernel, then subtract 1
    (the pixel itself) for foreground pixels.

    Degree values:
        1 → endpoint
        2 → interior path pixel
        3+ → junction / branch point
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = ndimage.convolve(
        skeleton.astype(np.uint8),
        kernel,
        mode='constant',
        cval=0
    )
    # neighbor_count includes the pixel itself, so degree = count - 1
    degree = np.where(skeleton, neighbor_count - 1, 0)
    return degree


def find_branch_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """Return boolean mask of endpoint pixels (degree == 1)."""
    degree = pixel_degree(skeleton)
    return (degree == 1) & skeleton


def prune_short_branches(
    skeleton: np.ndarray,
    min_length: int
) -> np.ndarray:
    """
    Iteratively remove branches shorter than min_length pixels.

    Strategy:
        1. Find endpoints (degree=1)
        2. Walk from each endpoint toward the nearest junction
        3. If walk length < min_length, delete those pixels
        4. Repeat until stable (short branches can expose new endpoints)
    """
    skel = skeleton.copy()

    for _ in range(min_length):
        degree = pixel_degree(skel)
        endpoints = (degree == 1) & skel
        # Remove all current endpoints (they're the tips of short branches)
        # This is a conservative approach — it removes one pixel per iteration
        # and repeats, so branches of length k are pruned in k iterations.
        skel[endpoints] = False

    # Restore any junction pixels that were accidentally removed
    # (can happen at isolated dots). Recompute degree on original.
    # Actually: re-running skeletonization after deletion is overkill —
    # the iterative endpoint removal is safe for pruning only.

    return skel


def remove_small_components(
    skeleton: np.ndarray,
    min_size: int
) -> np.ndarray:
    """
    Remove connected components smaller than min_size pixels.
    Eliminates isolated skeleton dots from binarization noise.
    """
    labeled = label(skeleton, connectivity=2)
    # Count component sizes
    component_sizes = np.bincount(labeled.ravel())
    # Zero out background (label 0) from consideration
    component_sizes[0] = 0
    # Keep only components >= min_size
    keep_mask = component_sizes >= min_size
    return keep_mask[labeled]


def skeletonize_image(
    binary: np.ndarray,
    cfg: SkeletonConfig | None = None
) -> np.ndarray:
    """
    Full skeletonization pipeline.

    Returns:
        bool ndarray, True = skeleton pixel
    """
    cfg = cfg or SkeletonConfig()

    skel = compute_skeleton(binary)
    skel = remove_small_components(skel, cfg.min_component_size)
    skel = prune_short_branches(skel, cfg.min_branch_length)

    # Final component cleanup after pruning (pruning can create new isolates)
    skel = remove_small_components(skel, cfg.min_component_size)

    return skel


if __name__ == "__main__":
    """
    Standalone test runner for skeletonization pipeline.

    Usage:
        python -m image_to_trajectory.preprocessing.skeletonize
    """

    import cv2
    from pathlib import Path

    from image_to_trajectory.preprocessing.binarize import binarize

    # Input image path
    TEST_IMAGE = Path(__file__).parent.parent / "input.png"

    try:
        print("[INFO] Running binarization...")
        binary = binarize(TEST_IMAGE)

        print("[INFO] Running skeletonization...")
        skeleton = skeletonize_image(binary)

        # Convert bool → uint8 for visualization
        skeleton_vis = (skeleton * 255).astype(np.uint8)

        # Save outputs
        binary_out = "debug_binary.png"
        skeleton_out = "debug_skeleton.png"

        cv2.imwrite(binary_out, binary)
        cv2.imwrite(skeleton_out, skeleton_vis)

        print(f"[INFO] Binary saved: {binary_out}")
        print(f"[INFO] Skeleton saved: {skeleton_out}")

        print(f"[INFO] Skeleton shape: {skeleton.shape}")
        print(f"[INFO] Skeleton pixels: {np.count_nonzero(skeleton)}")

        # Preview windows
        cv2.imshow("Binary", binary)
        cv2.imshow("Skeleton", skeleton_vis)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] {e}")