"""
GeoTIFF Utilities for SatQuery AI
Handles geospatial metadata extraction, coordinate transforms, and
writing georeferenced outputs (change maps, evidence rasters) back to disk.

Falls back gracefully to a "pseudo-georeferenced" mode (identity transform,
EPSG:3857) when the input is a plain PNG/JPEG with no CRS, so the rest of
the pipeline (Geospatial Processing Layer) always has a transform/CRS to work with.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available - GeoTIFF features will be limited.")


GEOTIFF_EXTENSIONS = {".tif", ".tiff", ".geotiff"}


def is_geotiff(path: Path) -> bool:
    return Path(path).suffix.lower() in GEOTIFF_EXTENSIONS


def read_geospatial_metadata(path: Path) -> Dict[str, Any]:
    """
    Extract CRS, transform, bounds, pixel resolution for a raster.
    Works for real GeoTIFFs. For non-georeferenced images (PNG/JPEG),
    returns a synthetic/identity georeferencing so downstream code
    (area computation, coordinate transforms, map rendering) never breaks.
    """
    path = Path(path)
    meta: Dict[str, Any] = {
        "has_crs": False,
        "crs": None,
        "transform": None,
        "bounds": None,
        "pixel_size_x": None,
        "pixel_size_y": None,
        "width": None,
        "height": None,
        "count": None,
        "georeferencing_source": "none",
    }

    if RASTERIO_AVAILABLE and is_geotiff(path):
        try:
            with rasterio.open(path) as src:
                meta["width"] = src.width
                meta["height"] = src.height
                meta["count"] = src.count
                meta["transform"] = src.transform
                meta["crs"] = str(src.crs) if src.crs else None
                meta["has_crs"] = src.crs is not None
                meta["bounds"] = tuple(src.bounds) if src.bounds else None
                if src.transform:
                    meta["pixel_size_x"] = abs(src.transform.a)
                    meta["pixel_size_y"] = abs(src.transform.e)
                meta["georeferencing_source"] = "file_crs" if src.crs else "file_no_crs"
                return meta
        except Exception as e:
            logger.warning(f"Could not read GeoTIFF metadata for {path}: {e}")

    # Fallback: synthetic georeferencing so area/coordinate stats still work.
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.width, img.height
    except Exception:
        w, h = 512, 512

    meta["width"], meta["height"], meta["count"] = w, h, 3
    meta["georeferencing_source"] = "synthetic"
    if RASTERIO_AVAILABLE:
        # Assume a nominal 10m/pixel (Sentinel-2-like) scene centered at (0,0)
        # purely so downstream "area changed" stats are demonstrable.
        pixel_size = 10.0  # meters
        meta["transform"] = Affine(pixel_size, 0.0, 0.0, 0.0, -pixel_size, 0.0)
        meta["crs"] = "EPSG:3857"
        meta["pixel_size_x"] = pixel_size
        meta["pixel_size_y"] = pixel_size
        meta["bounds"] = (0.0, -h * pixel_size, w * pixel_size, 0.0)
    return meta


def pixel_area_m2(meta: Dict[str, Any]) -> float:
    """Approximate area (m^2) of a single pixel given metadata."""
    px = meta.get("pixel_size_x")
    py = meta.get("pixel_size_y")
    if px and py:
        return float(px) * float(py)
    return 100.0  # default assumption: 10m x 10m


def changed_area_stats(change_mask: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a binary change mask into human-friendly area statistics."""
    changed_pixels = int(np.sum(change_mask > 0))
    total_pixels = int(change_mask.size)
    area_per_px = pixel_area_m2(meta)

    changed_m2 = changed_pixels * area_per_px
    total_m2 = total_pixels * area_per_px

    return {
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_pct": round(100.0 * changed_pixels / total_pixels, 2) if total_pixels else 0.0,
        "changed_area_m2": round(changed_m2, 1),
        "changed_area_ha": round(changed_m2 / 10000.0, 3),
        "changed_area_km2": round(changed_m2 / 1_000_000.0, 5),
        "total_area_km2": round(total_m2 / 1_000_000.0, 5),
        "pixel_size_m": round(area_per_px ** 0.5, 2),
        "georeferencing_source": meta.get("georeferencing_source", "unknown"),
    }


def save_array_as_geotiff(
    array: np.ndarray,
    out_path: Path,
    reference_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Persist a 2D or 3D numpy array as a GeoTIFF, using CRS/transform from
    reference_meta when available (from read_geospatial_metadata).
    Returns the output path, or None if rasterio is unavailable.
    """
    if not RASTERIO_AVAILABLE:
        logger.warning("rasterio unavailable - cannot write GeoTIFF.")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = array
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        # (H, W, C) -> (C, H, W)
        arr = np.moveaxis(arr, -1, 0)

    count, height, width = arr.shape
    dtype = arr.dtype.name if arr.dtype.name in ("uint8", "uint16", "float32") else "uint8"
    arr = arr.astype(dtype)

    crs = None
    transform = None
    if reference_meta:
        transform = reference_meta.get("transform")
        crs_str = reference_meta.get("crs")
        if crs_str:
            try:
                crs = CRS.from_string(crs_str)
            except Exception:
                crs = None

    if transform is None:
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    if crs is None:
        crs = CRS.from_epsg(3857)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(arr)

    return out_path


def bbox_pixel_to_geo(bbox: Tuple[int, int, int, int], meta: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Convert a pixel-space bbox (x1,y1,x2,y2) to geographic coordinates using the transform."""
    transform = meta.get("transform")
    if transform is None:
        return None
    x1, y1, x2, y2 = bbox
    gx1, gy1 = transform * (x1, y1)
    gx2, gy2 = transform * (x2, y2)
    return (gx1, gy1, gx2, gy2)
