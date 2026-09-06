"""
Input Validation Module for SatQuery AI
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"}
GEOTIFF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class ValidationResult:
    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.image_count: int = 0
        self.modalities: List[str] = []
        self.input_type: str = "unknown"

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "image_count": self.image_count,
            "modalities": self.modalities,
            "input_type": self.input_type,
        }


def detect_modality_from_filename(filename: str) -> str:
    name = filename.lower()
    if any(k in name for k in ["sar", "sentinel1", "s1", "risat", "radar"]):
        return "SAR"
    if any(k in name for k in ["optical", "sentinel2", "s2", "cartosat", "multispectral", "rgb"]):
        return "Optical"
    return "Unknown"


def validate_image_file(file_path: Path) -> Tuple[bool, str, Optional[Dict]]:
    try:
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported format: {suffix}", None

        meta = {
            "path": str(file_path),
            "filename": file_path.name,
            "extension": suffix,
            "is_geotiff": suffix in GEOTIFF_EXTENSIONS,
            "modality": detect_modality_from_filename(file_path.name),
        }

        try:
            with Image.open(file_path) as img:
                meta["width"] = img.width
                meta["height"] = img.height
                meta["mode"] = img.mode
                meta["format"] = img.format
        except Exception:
            try:
                import rasterio
                with rasterio.open(file_path) as src:
                    meta["width"] = src.width
                    meta["height"] = src.height
                    meta["crs"] = str(src.crs) if src.crs else None
                    meta["count"] = src.count
                    meta["is_geotiff"] = True
            except Exception as e2:
                return False, f"Cannot open image: {str(e2)}", None

        return True, "OK", meta
    except Exception as e:
        return False, f"Validation error: {str(e)}", None


def validate_inputs(
    file_paths: List[Path],
    query: str = "",
    user_modality_hints: Optional[List[str]] = None,
) -> ValidationResult:
    result = ValidationResult()

    if not file_paths:
        result.add_error("No images provided.")
        return result

    if len(file_paths) > 2:
        result.add_error("Maximum 2 images supported in current MVP.")
        return result

    result.image_count = len(file_paths)
    all_meta = []

    for fp in file_paths:
        ok, msg, meta = validate_image_file(fp)
        if not ok:
            result.add_error(f"{fp.name}: {msg}")
        else:
            all_meta.append(meta)
            result.modalities.append(meta.get("modality", "Unknown"))

    if not result.is_valid:
        return result

    result.metadata["images"] = all_meta

    if result.image_count == 1:
        result.input_type = "single"
    elif result.image_count == 2:
        mods = result.modalities
        if "SAR" in mods and "Optical" in mods:
            result.input_type = "optical_sar"
        else:
            result.input_type = "bi_temporal"

    if query and len(query.strip()) < 3:
        result.add_warning("Query is very short.")

    if user_modality_hints and len(user_modality_hints) == result.image_count:
        result.modalities = user_modality_hints

    for m in all_meta:
        if not m.get("is_geotiff") and m.get("extension") in IMAGE_EXTENSIONS:
            result.add_warning(
                f"{m['filename']}: PNG/JPEG accepted mainly for benchmarks. Prefer GeoTIFF."
            )

    return result


def get_input_summary(validation: ValidationResult) -> str:
    if not validation.is_valid:
        return "Validation Failed:\n" + "\n".join(f"- {e}" for e in validation.errors)

    lines = [
        f"Input Valid",
        f"• Image count : {validation.image_count}",
        f"• Input type  : {validation.input_type}",
        f"• Modalities  : {', '.join(validation.modalities)}",
    ]
    if validation.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in validation.warnings)
    return "\n".join(lines)
