"""
Optical + SAR Cross-modal Analysis
MVP: Side-by-side complementary information extraction (pure NumPy/PIL).
"""

from typing import Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_gray(path: Path, size=(512, 512)) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def load_rgb(path: Path, size=(512, 512)) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def analyze_optical_sar(
    optical_path: Path,
    sar_path: Path,
    query: str = "",
) -> Dict[str, Any]:
    try:
        opt = load_rgb(optical_path)
        sar = load_gray(sar_path)

        opt_mean = opt.mean(axis=(0, 1))
        sar_mean = float(sar.mean())
        sar_std = float(sar.std())

        insights = []
        insights.append(
            "Built-up / urban areas often appear bright in optical imagery and "
            "show strong backscatter (bright) in SAR due to double-bounce effects."
        )
        insights.append(
            "Water bodies typically appear dark in optical true-color images and "
            "also dark in SAR (low backscatter) under calm conditions."
        )
        insights.append(
            "Vegetation can be identified spectrally in optical data while SAR provides structural information."
        )

        q = query.lower()
        focused = []
        if any(k in q for k in ["built", "urban", "building", "settlement"]):
            focused.append(
                "For built-up detection: combine optical spectral/textural cues with SAR backscatter. "
                "High optical brightness + high SAR return is a strong indicator of built-up."
            )
        if any(k in q for k in ["water", "flood", "river", "lake"]):
            focused.append(
                "For water mapping: optical dark regions + consistently dark SAR returns "
                "improve reliability, especially under cloud cover."
            )
        if any(k in q for k in ["vegetation", "forest", "crop", "agriculture"]):
            focused.append(
                "For vegetation: optical provides condition cues while SAR adds structure and cloud penetration."
            )

        if not focused:
            focused = insights

        answer = (
            "Cross-modal Optical + SAR analysis:\n\n"
            + "\n".join(f"• {f}" for f in focused) +
            f"\n\nOptical mean RGB ≈ ({opt_mean[0]:.0f}, {opt_mean[1]:.0f}, {opt_mean[2]:.0f}). "
            f"SAR mean intensity ≈ {sar_mean:.1f} (std {sar_std:.1f}). "
            "These complementary signals together provide more reliable land-cover information "
            "than either modality alone."
        )

        sar_rgb = np.stack([sar, sar, sar], axis=-1)
        vis = np.concatenate([opt, sar_rgb], axis=1)

        return {
            "success": True,
            "answer": answer,
            "confidence": 0.72,
            "model_used": "optical_sar_analyzer",
            "visualization": vis,
            "parameters": {"size": (512, 512)},
            "stats": {
                "optical_mean_rgb": opt_mean.tolist(),
                "sar_mean": sar_mean,
                "sar_std": sar_std,
            },
        }
    except Exception as e:
        logger.exception("Optical-SAR analysis failed")
        return {
            "success": False,
            "error": str(e),
            "confidence": 0.0,
            "model_used": "optical_sar_analyzer",
        }
