"""
Visualization utilities for SatQuery AI.
Supports single + multiple bounding boxes, change heatmaps, evidence panels.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


def _get_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_bbox_on_image(
    img_array: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (255, 60, 60),
) -> np.ndarray:
    """Draw a single labeled bounding box (backward compatible)."""
    img = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

    if label:
        font = _get_font(14)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2), label, fill=(255, 255, 255), font=font)

    return np.array(img)


def draw_bboxes_on_image(
    img_array: np.ndarray,
    boxes: List[Dict[str, Any]],
    thickness: int = 3,
) -> np.ndarray:
    """Draw multiple bounding boxes with labels and confidence."""
    if img_array is None or not boxes:
        return img_array

    img = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(13)

    color_map = {
        "water": (0, 140, 255),
        "urban": (255, 40, 40),
        "vegetation": (40, 200, 40),
        "default": (255, 60, 60),
    }

    for box in boxes:
        if "bbox" not in box:
            continue
        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
        label = str(box.get("label", "object"))
        conf = box.get("confidence", None)
        color = color_map.get(label.lower(), color_map["default"])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        text = f"{label} {conf:.2f}" if conf is not None else label
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        ty = max(0, y1 - th - 6)

        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2), text, fill=(255, 255, 255), font=font)

    return np.array(img)


def make_change_heatmap(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 0:
        m = m / m.max()
    r = np.clip(m * 3.0, 0, 1)
    g = np.clip(m * 3.0 - 1.0, 0, 1)
    b = np.clip(m * 3.0 - 2.0, 0, 1)
    heat = np.stack([r, g, b], axis=-1)
    return (heat * 255).astype(np.uint8)


def overlay_change_on_image(
    base_img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    base = base_img.astype(np.float32)
    heat = make_change_heatmap(mask).astype(np.float32)
    mask_norm = (mask > 0).astype(np.float32)[..., None]
    blended = base * (1 - mask_norm * alpha) + heat * (mask_norm * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def side_by_side(*images: np.ndarray, labels: Optional[List[str]] = None, pad: int = 6) -> np.ndarray:
    imgs = [Image.fromarray(im.astype(np.uint8)).convert("RGB") for im in images]
    h = max(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]

    caption_h = 24 if labels else 0
    total_w = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (total_w, h + caption_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(14)

    x = 0
    for i, im in enumerate(imgs):
        canvas.paste(im, (x, caption_h))
        if labels and i < len(labels):
            draw.text((x + 4, 4), labels[i], fill=(255, 255, 255), font=font)
        x += im.width + pad

    return np.array(canvas)


def build_evidence_panel(
    result_type: str,
    images: List[np.ndarray],
    labels: List[str],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    bboxes: Optional[List[Dict[str, Any]]] = None,
    change_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    panels = list(images)
    panel_labels = list(labels)

    if bboxes and len(bboxes) > 0 and len(panels) >= 1:
        panels[0] = draw_bboxes_on_image(panels[0], bboxes)
    elif bbox is not None and len(panels) >= 1:
        panels[0] = draw_bbox_on_image(panels[0], bbox, label="Region of interest")

    if change_mask is not None and len(panels) >= 2:
        overlay = overlay_change_on_image(panels[-1], change_mask)
        panels.append(overlay)
        panel_labels.append("Change Overlay")
        heat = make_change_heatmap(change_mask)
        panels.append(heat)
        panel_labels.append("Change Heatmap")

    return side_by_side(*panels, labels=panel_labels)
"""
Visualization utilities for SatQuery AI.
Supports single + multiple bounding boxes, change heatmaps, evidence panels.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


def _get_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_bbox_on_image(
    img_array: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = (255, 60, 60),
) -> np.ndarray:
    """Draw a single labeled bounding box (backward compatible)."""
    img = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

    if label:
        font = _get_font(14)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2), label, fill=(255, 255, 255), font=font)

    return np.array(img)


def draw_bboxes_on_image(
    img_array: np.ndarray,
    boxes: List[Dict[str, Any]],
    thickness: int = 3,
) -> np.ndarray:
    """Draw multiple bounding boxes with labels and confidence."""
    if img_array is None or not boxes:
        return img_array

    img = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(13)

    color_map = {
        "water": (0, 140, 255),
        "urban": (255, 40, 40),
        "vegetation": (40, 200, 40),
        "default": (255, 60, 60),
    }

    for box in boxes:
        if "bbox" not in box:
            continue
        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
        label = str(box.get("label", "object"))
        conf = box.get("confidence", None)
        color = color_map.get(label.lower(), color_map["default"])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        text = f"{label} {conf:.2f}" if conf is not None else label
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        ty = max(0, y1 - th - 6)

        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2), text, fill=(255, 255, 255), font=font)

    return np.array(img)


def make_change_heatmap(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 0:
        m = m / m.max()
    r = np.clip(m * 3.0, 0, 1)
    g = np.clip(m * 3.0 - 1.0, 0, 1)
    b = np.clip(m * 3.0 - 2.0, 0, 1)
    heat = np.stack([r, g, b], axis=-1)
    return (heat * 255).astype(np.uint8)


def overlay_change_on_image(
    base_img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    base = base_img.astype(np.float32)
    heat = make_change_heatmap(mask).astype(np.float32)
    mask_norm = (mask > 0).astype(np.float32)[..., None]
    blended = base * (1 - mask_norm * alpha) + heat * (mask_norm * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def side_by_side(*images: np.ndarray, labels: Optional[List[str]] = None, pad: int = 6) -> np.ndarray:
    imgs = [Image.fromarray(im.astype(np.uint8)).convert("RGB") for im in images]
    h = max(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]

    caption_h = 24 if labels else 0
    total_w = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (total_w, h + caption_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = _get_font(14)

    x = 0
    for i, im in enumerate(imgs):
        canvas.paste(im, (x, caption_h))
        if labels and i < len(labels):
            draw.text((x + 4, 4), labels[i], fill=(255, 255, 255), font=font)
        x += im.width + pad

    return np.array(canvas)


def build_evidence_panel(
    result_type: str,
    images: List[np.ndarray],
    labels: List[str],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    bboxes: Optional[List[Dict[str, Any]]] = None,
    change_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    panels = list(images)
    panel_labels = list(labels)

    if bboxes and len(bboxes) > 0 and len(panels) >= 1:
        panels[0] = draw_bboxes_on_image(panels[0], bboxes)
    elif bbox is not None and len(panels) >= 1:
        panels[0] = draw_bbox_on_image(panels[0], bbox, label="Region of interest")

    if change_mask is not None and len(panels) >= 2:
        overlay = overlay_change_on_image(panels[-1], change_mask)
        panels.append(overlay)
        panel_labels.append("Change Overlay")
        heat = make_change_heatmap(change_mask)
        panels.append(heat)
        panel_labels.append("Change Heatmap")

    return side_by_side(*panels, labels=panel_labels)
