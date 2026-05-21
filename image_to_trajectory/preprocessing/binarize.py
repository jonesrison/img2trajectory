# preprocessing/binarize.py
"""
Stage 1: Convert input image to a clean binary image.

Pipeline:
  RGB/RGBA → grayscale → CLAHE → Gaussian blur → Otsu threshold
  → morphological cleanup → binary image (lines=255, bg=0)
"""

import cv2
import numpy as np
from pathlib import Path
from image_to_trajectory.config import BinarizationConfig

def load_image(path: str | Path) -> np.ndarray:
    """Load image from disk, handling RGB and RGBA."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert any channel format to single-channel grayscale."""
    if img.ndim == 2:
        return img  # already grayscale
    if img.shape[2] == 4:
        # RGBA: composite alpha onto white background first
        alpha = img[:, :, 3:4].astype(float) / 255.0
        rgb = img[:, :, :3].astype(float)
        white = np.ones_like(rgb) * 255.0
        composited = (alpha * rgb + (1 - alpha) * white).astype(np.uint8)
        return cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, cfg: BinarizationConfig) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Normalizes local brightness — essential for AI-generated art with
    gradient backgrounds or uneven lighting.
    """
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid_size
    )
    return clahe.apply(gray)


def threshold_otsu(blurred: np.ndarray, invert: bool) -> np.ndarray:
    """
    Otsu's method finds the globally optimal threshold automatically.
    No manual tuning needed for different image brightness levels.

    invert=True: dark lines on light bg → after invert, lines become white
    """
    flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if invert \
            else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, binary = cv2.threshold(blurred, 0, 255, flags)
    return binary


def morphological_cleanup(
    binary: np.ndarray,
    cfg: BinarizationConfig
) -> np.ndarray:
    """
    Close small gaps in lines, then open (remove) isolated noise pixels.
    Order matters: close first to preserve line continuity.
    """
    if cfg.morph_close_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.morph_close_kernel * 2 + 1,) * 2
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    if cfg.morph_open_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.morph_open_kernel * 2 + 1,) * 2
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    return binary


def binarize(
    path: str | Path,
    cfg: BinarizationConfig | None = None
) -> np.ndarray:
    """
    Full binarization pipeline.

    Returns:
        Binary image, dtype uint8, lines=255, background=0.
    """
    cfg = cfg or BinarizationConfig()

    raw = load_image(path)
    gray = to_grayscale(raw)

    equalized = apply_clahe(gray, cfg)

    # Gaussian blur smooths anti-aliasing artifacts before threshold
    ksize = cfg.blur_kernel_size | 1  # ensure odd
    blurred = cv2.GaussianBlur(equalized, (ksize, ksize), 0)

    binary = threshold_otsu(blurred, cfg.invert)
    binary = morphological_cleanup(binary, cfg)

    return binary

if __name__ == "__main__":
    """
    Standalone test runner for binarization pipeline.

    Usage:
        python preprocessing/binarize.py
    """

    TEST_IMAGE = Path(__file__).parent.parent / "input.png"
    try:
        binary = binarize(TEST_IMAGE)

        # Save result
        output_path = Path(__file__).parent.parent / "output_binary.png"
        cv2.imwrite(output_path, binary)

        print(f"[INFO] Binary image saved to: {output_path}")
        print(f"[INFO] Shape: {binary.shape}")
        print(f"[INFO] Dtype: {binary.dtype}")
        print(f"[INFO] Pixel range: {binary.min()} -> {binary.max()}")

        # Preview window
        cv2.imshow("Binary Output", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] {e}")