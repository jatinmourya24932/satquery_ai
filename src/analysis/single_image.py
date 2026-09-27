"""
Single Image Analysis: VQA, Captioning, Multi-region Grounding
- Real Qwen3-VL used for caption & VQA
- Improved multi-region heuristic grounding for localization queries
"""

from typing import Dict, Any, Tuple, List
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


# def load_image(path: Path, size=(512, 512)):
#     img = Image.open(path).convert("RGB")
#     img = img.resize(size, Image.BILINEAR)
#     return img, np.array(img)
def load_image(path: Path, size=(512, 512)):
    """
    Load image supporting both normal images (PNG/JPG) and GeoTIFF.
    Returns (PIL Image, numpy array)
    """
    import numpy as np
    from PIL import Image

    path = Path(path)
    suffix = path.suffix.lower()

    # ----- Try GeoTIFF / multi-band with rasterio first -----
    if suffix in {".tif", ".tiff", ".geotiff"}:
        try:
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(path) as src:
                # Read first 3 bands (or less)
                count = min(src.count, 3)
                data = src.read(
                    list(range(1, count + 1)),
                    out_shape=(count, size[1], size[0]),
                    resampling=Resampling.bilinear,
                )

                # Handle different band counts
                if data.shape[0] == 1:
                    # Single band → grayscale to RGB
                    arr = np.stack([data[0]] * 3, axis=-1)
                elif data.shape[0] == 2:
                    arr = np.stack([data[0], data[1], data[0]], axis=-1)
                else:
                    arr = np.transpose(data, (1, 2, 0))  # CHW → HWC

                # Normalize to 0-255 if needed
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float32)
                    for i in range(arr.shape[-1]):
                        band = arr[:, :, i]
                        mn, mx = band.min(), band.max()
                        if mx > mn:
                            arr[:, :, i] = (band - mn) / (mx - mn) * 255
                    arr = arr.astype(np.uint8)

                img = Image.fromarray(arr)
                return img, arr
        except Exception as e:
            logger.warning(f"rasterio failed for {path.name}: {e}. Trying PIL...")

    # ----- Fallback to PIL (PNG/JPG and some TIFFs) -----
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        return img, np.array(img)
    except Exception as e:
        raise ValueError(f"Cannot open image {path.name}: {e}")


def get_image_stats(img_array: np.ndarray) -> Dict[str, Any]:
    """Extract useful statistics from the image."""
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)

    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    std_r, std_g, std_b = r.std(), g.std(), b.std()
    brightness = (mean_r + mean_g + mean_b) / 3

    veg_score = (mean_g - mean_r) / (mean_g + mean_r + 1e-5)
    water_score = (mean_b - mean_r) / (mean_b + mean_r + 1e-5) - (brightness / 255.0)
    built_score = (brightness / 255.0) - max(0, veg_score)
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

    if bright < 60:
        scores["water or dark surfaces"] += 0.2
    if bright > 140:
        scores["built-up / urban or bare soil"] += 0.15

    primary = max(scores, key=scores.get)
    return primary, "(confidence based on color statistics)"


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

    if any(k in q for k in ["describe", "what is", "land cover", "land-cover", "scene", "visible"]):
        return heuristic_caption(img_array, filename)

    if any(k in q for k in ["water", "river", "lake", "flood", "pond"]):
        score = stats["water_score"]
        if score > 0.08:
            return (
                f"Yes, there are indications of water or dark surfaces in this image. "
                f"Water-score ≈ {score:.2f}. Darker and bluish regions are the most likely water candidates."
            )
        return (
            f"Water presence appears limited (water-score ≈ {score:.2f}). "
            f"Dominant appearance is closer to **{primary}**."
        )

    if any(k in q for k in ["built", "urban", "building", "settlement", "city", "road"]):
        score = stats["built_score"]
        if score > 0.2:
            return (
                f"Built-up or bare-soil signatures are visible (built-score ≈ {score:.2f}). "
                f"Primary land-cover estimate: **{primary}**."
            )
        return (
            f"Strong built-up signatures are not dominant (built-score ≈ {score:.2f}). "
            f"Image leans more towards **{primary}**."
        )

    if any(k in q for k in ["vegetation", "forest", "crop", "agriculture", "green", "tree"]):
        score = stats["veg_score"]
        if score > 0.04:
            return (
                f"Vegetation signals are present (veg-score ≈ {score:.2f}). "
                f"Overall scene estimated as **{primary}**."
            )
        return (
            f"Vegetation does not appear dominant (veg-score ≈ {score:.2f}). "
            f"Image is closer to **{primary}**."
        )

    if "change" in q:
        return (
            "A single image cannot show change over time. "
            "Please upload a bi-temporal pair for change detection."
        )

    return (
        f"Regarding your query: \"{query}\"\n\n"
        f"{heuristic_caption(img_array, filename)}\n\n"
        f"This is a scene-level interpretation based on color and texture statistics."
    )


def basic_grounding(query: str, img_array: np.ndarray, max_regions: int = 6) -> List[Dict[str, Any]]:
    """
    Multi-region detection using color + contour analysis.
    Returns list of boxes for water / urban / vegetation.
    Note: This is heuristic because Qwen3-VL does not output bounding boxes.
    """
    h, w = img_array.shape[:2]
    q = query.lower()
    boxes = []

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # ----- WATER -----
    if any(k in q for k in ["water", "river", "lake", "pond", "reservoir", "stream", "canal",
                            "water body", "waterbodies", "flood", "where is the water", "detect water"]):
        lower = np.array([90, 25, 25])
        upper = np.array([140, 255, 210])
        mask = cv2.inRange(hsv, lower, upper)
        _, dark = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_or(mask, dark)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad = 8
            x, y = max(0, x - pad), max(0, y - pad)
            bw, bh = min(w - x, bw + 2 * pad), min(h - y, bh + 2 * pad)
            boxes.append({
                "label": "water",
                "bbox": [x, y, x + bw, y + bh],
                "confidence": round(min(0.92, 0.55 + (area / (w * h)) * 3), 3),
                "area": int(area),
            })

    # ----- URBAN -----
    if any(k in q for k in ["urban", "built", "building", "buildings", "city", "settlement",
                            "road", "concrete", "built-up", "infrastructure", "highlight built"]):
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        l_ch = lab[:, :, 0]
        _, bright = cv2.threshold(l_ch, 135, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(edges, bright)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]:
            area = cv2.contourArea(cnt)
            if area < 1200:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad = 6
            x, y = max(0, x - pad), max(0, y - pad)
            bw, bh = min(w - x, bw + 2 * pad), min(h - y, bh + 2 * pad)
            boxes.append({
                "label": "urban",
                "bbox": [x, y, x + bw, y + bh],
                "confidence": round(min(0.88, 0.50 + (area / (w * h)) * 2.5), 3),
                "area": int(area),
            })

    # ----- VEGETATION -----
    if any(k in q for k in ["vegetation", "forest", "tree", "green", "crop", "agriculture",
                            "veg", "plants", "grass", "locate vegetation"]):
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad = 6
            x, y = max(0, x - pad), max(0, y - pad)
            bw, bh = min(w - x, bw + 2 * pad), min(h - y, bh + 2 * pad)
            boxes.append({
                "label": "vegetation",
                "bbox": [x, y, x + bw, y + bh],
                "confidence": round(min(0.90, 0.52 + (area / (w * h)) * 2.8), 3),
                "area": int(area),
            })

    boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    return boxes[:max_regions]


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
            },
        }

        # Try real VLM
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
            result["confidence"] = 0.85 if (vlm and getattr(vlm, "backend", "") != "heuristic") else 0.72

        elif task == "grounding":
            boxes = basic_grounding(query, arr)
            result["bboxes"] = boxes
            result["bbox"] = boxes[0]["bbox"] if boxes else None
            result["answer"] = (
                f"Found {len(boxes)} region(s) matching '{query}'."
                if boxes else
                f"No clear region found for '{query}'."
            )
            result["confidence"] = boxes[0]["confidence"] if boxes else 0.3
            result["note"] = "Multi-region heuristic grounding (Qwen3-VL does not output boxes)."

        else:  # vqa
            result["answer"] = vlm.vqa(path, query, arr, filename) if vlm else heuristic_vqa(query, arr, filename)
            result["confidence"] = 0.85 if (vlm and getattr(vlm, "backend", "") != "heuristic") else 0.70

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
