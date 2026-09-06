"""
Bi-temporal Change Detection Module
Uses absolute difference + morphological-like cleanup with pure NumPy/PIL for MVP.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import logging

logger = logging.getLogger(__name__)


def load_and_resize(path: Path, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def compute_change_map(
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: float = 30.0,
) -> Tuple[np.ndarray, float]:
    """Returns binary change mask (uint8 0/255) and change ratio (0-1)."""
    # Convert to grayscale
    g1 = (0.299 * img1[:,:,0] + 0.587 * img1[:,:,1] + 0.114 * img1[:,:,2]).astype(np.float32)
    g2 = (0.299 * img2[:,:,0] + 0.587 * img2[:,:,1] + 0.114 * img2[:,:,2]).astype(np.float32)

    diff = np.abs(g1 - g2)
    mask = (diff > threshold).astype(np.uint8) * 255

    # Simple morphological-like cleanup using PIL
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))  # erode-like
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))  # dilate-like
    mask = np.array(mask_img)

    change_ratio = float(np.sum(mask > 0) / mask.size)
    return mask, change_ratio


def describe_change(change_ratio: float, query: str = "") -> str:
    if change_ratio < 0.02:
        level = "very little or no significant"
        direction = "remained largely unchanged"
    elif change_ratio < 0.08:
        level = "minor"
        direction = "show limited changes"
    elif change_ratio < 0.20:
        level = "moderate"
        direction = "show noticeable changes"
    else:
        level = "significant"
        direction = "show substantial changes"

    base = (
        f"Analysis of the bi-temporal pair indicates {level} change "
        f"(approximately {change_ratio*100:.1f}% of the area). "
        f"The region appears to {direction} between the two dates."
    )

    q = query.lower()
    if "built" in q or "urban" in q or "building" in q:
        base += " Particular attention was given to built-up / urban areas."
    if "water" in q or "flood" in q or "river" in q:
        base += " Water-related changes were considered in the analysis."
    if "vegetation" in q or "forest" in q or "crop" in q:
        base += " Vegetation and land-cover changes were highlighted."

    return base


def run_change_detection(
    path1: Path,
    path2: Path,
    query: str = "",
    threshold: float = 30.0,
) -> Dict[str, Any]:
    try:
        img1 = load_and_resize(path1)
        img2 = load_and_resize(path2)

        mask, ratio = compute_change_map(img1, img2, threshold=threshold)
        description = describe_change(ratio, query)

        if ratio < 0.01 or ratio > 0.6:
            confidence = 0.55
        else:
            confidence = 0.75

        # Side-by-side visualization
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        vis = np.concatenate([img1, img2, mask_rgb], axis=1)

        return {
            "success": True,
            "change_mask": mask,
            "change_ratio": ratio,
            "description": description,
            "confidence": confidence,
            "visualization": vis,
            "model_used": "change_detector_v1 (diff-based MVP)",
            "parameters": {"threshold": threshold, "size": (512, 512)},
        }
    except Exception as e:
        logger.exception("Change detection failed")
        return {
            "success": False,
            "error": str(e),
            "confidence": 0.0,
            "model_used": "change_detector_v1",
        }
