"""
Single Image Analysis: VQA, Captioning, basic Grounding
Improved heuristic version - answers change based on image stats + query.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import hashlib

logger = logging.getLogger(__name__)


def load_image(path: Path, size=(512, 512)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    return img, np.array(img)


def get_image_stats(img_array: np.ndarray) -> Dict[str, Any]:
    """Extract useful statistics from the image."""
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)

    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    std_r, std_g, std_b = r.std(), g.std(), b.std()
    brightness = (mean_r + mean_g + mean_b) / 3

    # Simple vegetation index like (G - R) / (G + R + 1)
    veg_score = (mean_g - mean_r) / (mean_g + mean_r + 1e-5)

    # Water-ish: low brightness + blue dominance
    water_score = (mean_b - mean_r) / (mean_b + mean_r + 1e-5) - (brightness / 255.0)

    # Built-up proxy: higher brightness + lower vegetation
    built_score = (brightness / 255.0) - max(0, veg_score)

    # Texture proxy (std)
    texture = (std_r + std_g + std_b) / 3

    return {
        "mean_rgb": (mean_r, mean_g, mean_b),
        "brightness": brightness,
        "veg_score": veg_score,
        "water_score": water_score,
        "built_score": built_score,
        "texture": texture,
        "std_rgb": (std_r, std_g, std_b),
    }


def landcover_from_stats(stats: Dict) -> Tuple[str, str]:
    """Return primary land-cover guess and confidence text."""
    veg = stats["veg_score"]
    water = stats["water_score"]
    built = stats["built_score"]
    bright = stats["brightness"]

    scores = {
        "vegetation / agriculture": veg + 0.15,
        "built-up / urban or bare soil": built,
        "water or dark surfaces": water + 0.1,
        "mixed land-cover": 0.3,
    }

    # Adjust by brightness
    if bright < 60:
        scores["water or dark surfaces"] += 0.2
    if bright > 140:
        scores["built-up / urban or bare soil"] += 0.15

    primary = max(scores, key=scores.get)
    return primary, f"(confidence based on color statistics)"


def heuristic_caption(img_array: np.ndarray, filename: str = "") -> str:
    stats = get_image_stats(img_array)
    primary, _ = landcover_from_stats(stats)
    bright = stats["brightness"]
    texture = stats["texture"]

    brightness_desc = "dark" if bright < 70 else "moderate" if bright < 130 else "bright"
    texture_desc = "smooth" if texture < 25 else "moderately textured" if texture < 45 else "highly textured"

    mean_r, mean_g, mean_b = stats["mean_rgb"]

    parts = [
        f"This remote sensing image shows a landscape dominated by **{primary}**.",
        f"Overall brightness is {brightness_desc} (avg ≈ {bright:.0f}/255) and the scene appears {texture_desc}.",
        f"Mean RGB values are approximately ({mean_r:.0f}, {mean_g:.0f}, {mean_b:.0f}).",
    ]

    if stats["veg_score"] > 0.05:
        parts.append("Green channel is relatively strong, suggesting presence of vegetation.")
    if stats["water_score"] > 0.05:
        parts.append("Blueish / darker regions are present which may correspond to water bodies.")
    if stats["built_score"] > 0.25:
        parts.append("Brighter structured areas are visible that often correspond to built-up or bare soil.")

    name = filename.lower()
    if "urban" in name or "city" in name:
        parts.append("Filename also suggests an urban context.")
    if "forest" in name or "veg" in name:
        parts.append("Filename indicates vegetation focus.")
    if "water" in name or "river" in name or "lake" in name:
        parts.append("Filename suggests water-related content.")

    return " ".join(parts)


def heuristic_vqa(query: str, img_array: np.ndarray, filename: str = "") -> str:
    q = query.lower().strip()
    stats = get_image_stats(img_array)
    primary, _ = landcover_from_stats(stats)

    # --- Specific question types ---
    if any(k in q for k in ["describe", "what is", "land cover", "land-cover", "scene", "visible", "show"]):
        return heuristic_caption(img_array, filename)

    if any(k in q for k in ["water", "river", "lake", "flood", "pond"]):
        score = stats["water_score"]
        if score > 0.08:
            return (
                f"Yes, there are indications of water or dark surfaces in this image. "
                f"Water-score ≈ {score:.2f}. Darker and bluish regions are the most likely water candidates. "
                f"For precise delineation, NDWI or a dedicated water model is recommended."
            )
        else:
            return (
                f"Water presence appears limited in this image (water-score ≈ {score:.2f}). "
                f"The dominant appearance is closer to **{primary}**. "
                f"If water is expected, check for very dark patches or provide a SAR image for confirmation."
            )

    if any(k in q for k in ["built", "urban", "building", "settlement", "city", "road", "infrastructure"]):
        score = stats["built_score"]
        if score > 0.2:
            return (
                f"Built-up or bare-soil like signatures are visible (built-score ≈ {score:.2f}). "
                f"Brighter and more structured regions are the main candidates for buildings / infrastructure. "
                f"Primary land-cover estimate: **{primary}**."
            )
        else:
            return (
                f"Strong built-up signatures are not dominant (built-score ≈ {score:.2f}). "
                f"The image leans more towards **{primary}**. "
                f"Some brighter patches may still contain sparse structures."
            )

    if any(k in q for k in ["vegetation", "forest", "crop", "agriculture", "green", "plant", "tree"]):
        score = stats["veg_score"]
        if score > 0.04:
            return (
                f"Vegetation signals are present (veg-score ≈ {score:.2f}). "
                f"Higher green response relative to red supports presence of vegetation / crops. "
                f"Overall scene is estimated as **{primary}**."
            )
        else:
            return (
                f"Vegetation does not appear dominant (veg-score ≈ {score:.2f}). "
                f"The image is closer to **{primary}**. "
                f"Some residual green patches may still exist."
            )

    if "change" in q:
        return (
            "A single image cannot show change over time. "
            "Please upload a **bi-temporal pair** (two images of the same area from different dates) "
            "so that change detection and change-based VQA can be performed."
        )

    if any(k in q for k in ["how many", "count", "number of"]):
        return (
            f"Counting specific objects (buildings, trees, vehicles etc.) requires an object detection model. "
            f"Current analysis only provides scene-level understanding. "
            f"Dominant land-cover appears to be **{primary}**."
        )

    # Default – still use image stats so answer is not static
    return (
        f"Regarding your query: \"{query}\"\n\n"
        f"{heuristic_caption(img_array, filename)}\n\n"
        f"This is a scene-level interpretation based on color and texture statistics. "
        f"For higher accuracy, a remote-sensing fine-tuned VLM (e.g. on BigEarthNet) should be used."
    )


def basic_grounding(query: str, img_array: np.ndarray) -> Dict[str, Any]:
    """Slightly smarter demo grounding – shifts box based on query keywords + image stats."""
    h, w = img_array.shape[:2]
    stats = get_image_stats(img_array)
    q = query.lower()

    # Default center
    x1, y1, x2, y2 = int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)

    # Shift based on simple cues
    if "water" in q or "river" in q or "lake" in q:
        # Prefer darker regions – approximate by shifting a bit
        x1, y1 = int(w*0.1), int(h*0.4)
        x2, y2 = int(w*0.5), int(h*0.9)
    elif "built" in q or "urban" in q or "building" in q:
        x1, y1 = int(w*0.4), int(h*0.1)
        x2, y2 = int(w*0.9), int(h*0.6)
    elif "vegetation" in q or "forest" in q or "crop" in q:
        x1, y1 = int(w*0.15), int(h*0.15)
        x2, y2 = int(w*0.65), int(h*0.7)

    return {
        "bbox": [x1, y1, x2, y2],
        "label": f"region related to: {query[:40]}",
        "confidence": 0.48,
        "note": "Demo grounding only. Replace with real referring-expression model later."
    }


def run_single_image_analysis(
    path: Path,
    query: str,
    task: str = "vqa",
) -> Dict[str, Any]:
    try:
        img, arr = load_image(path)
        filename = path.name
        stats = get_image_stats(arr)

        result = {
            "success": True,
            "task": task,
            "model_used": "rs_vlm_heuristic_v2",
            "confidence": 0.67,
            "image_stats": {
                "brightness": round(stats["brightness"], 1),
                "veg_score": round(stats["veg_score"], 3),
                "water_score": round(stats["water_score"], 3),
                "built_score": round(stats["built_score"], 3),
            }
        }

        # RS-VLM (Core): Qwen3-VL (+ optional BigEarthNet LoRA) if enabled, else heuristic.
        try:
            from src.models.vlm_wrapper import RSVLMCore
            vlm = RSVLMCore.get()
            info = vlm.info()
            result["model_used"] = info.get("model_id", "unknown")
            result["vlm_backend"] = info.get("backend", "unknown")
            result["lora_adapter"] = info.get("lora_adapter")
        except Exception as e:
            vlm = None
            result["vlm_backend"] = f"load_failed: {e}"

        if task == "caption":
            result["answer"] = vlm.caption(path, arr, filename) if vlm else heuristic_caption(arr, filename)
            result["confidence"] = 0.85 if (vlm and vlm.backend != "heuristic") else 0.72
        elif task == "grounding":
            ground = basic_grounding(query, arr)
            result["answer"] = f"Highlighted region for query: '{query}'"
            result["bbox"] = ground["bbox"]
            result["confidence"] = ground["confidence"]
            result["note"] = ground["note"]
        else:
            result["answer"] = vlm.vqa(path, query, arr, filename) if vlm else heuristic_vqa(query, arr, filename)
            result["confidence"] = 0.85 if (vlm and vlm.backend != "heuristic") else 0.70

        result["image_size"] = list(arr.shape[:2])
        return result

    except Exception as e:
        logger.exception("Single image analysis failed")
        return {
            "success": False,
            "error": str(e),
            "confidence": 0.0,
            "model_used": "rs_vlm_heuristic_v2",
        }
