"""
Report / Downloadables generator - implements the "Downloadables:
GeoTIFF, PDF, JSON" box of the Output Layer.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from io import BytesIO
from datetime import datetime
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_json_report(result: Dict[str, Any]) -> bytes:
    """Serialize the final result (minus large binary/array fields) to JSON."""
    safe = {}
    for k, v in result.items():
        if k in ("visual_evidence", "change_mask", "raw_results"):
            continue
        if isinstance(v, np.ndarray):
            continue
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)

    safe["generated_at"] = datetime.now().isoformat()
    safe["report_type"] = "SatQuery AI - Analysis Report"
    return json.dumps(safe, indent=2, default=str).encode("utf-8")


def generate_pdf_report(result: Dict[str, Any]) -> Optional[bytes]:
    """Render a simple, self-contained PDF summary report using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except Exception as e:
        logger.warning(f"reportlab unavailable, cannot build PDF: {e}")
        return None

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#0B3D91"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#0B3D91"))
    body = styles["BodyText"]

    story = [
        Paragraph("SatQuery AI - Analysis Report", title_style),
        Paragraph("ISRO | Smart India Hackathon 2026 | PS-26167", body),
        Spacer(1, 12),
    ]

    story.append(Paragraph("Query", h2))
    story.append(Paragraph(str(result.get("query", "-")), body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Answer", h2))
    for para in str(result.get("answer", "-")).split("\n\n"):
        story.append(Paragraph(para.replace("**", ""), body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Key Metrics", h2))
    meta_rows = [
        ["Primary Task", str(result.get("primary_task", "-")).upper()],
        ["Input Type", str(result.get("input_type", "-"))],
        ["Confidence", f"{result.get('confidence', 0):.0%}"],
        ["Models Used", ", ".join(result.get("models_used", []))],
    ]
    table = Table(meta_rows, colWidths=[5 * cm, 11 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    quant = result.get("quantitative_summary", {})
    if quant.get("metrics"):
        story.append(Paragraph("Quantitative Summary", h2))
        rows = [["Metric", "Value"]] + [[k, str(v)] for k, v in quant["metrics"].items() if v is not None]
        qtable = Table(rows, colWidths=[8 * cm, 8 * cm])
        qtable.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(qtable)
        story.append(Spacer(1, 8))

    if quant.get("area_analysis"):
        story.append(Paragraph("Area Analysis (Geospatial)", h2))
        area = quant["area_analysis"]
        rows = [["Metric", "Value"]] + [[k, str(v)] for k, v in area.items()]
        atable = Table(rows, colWidths=[8 * cm, 8 * cm])
        atable.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(atable)
        story.append(Spacer(1, 8))

    vis = result.get("visual_evidence")
    if isinstance(vis, np.ndarray):
        try:
            from PIL import Image as PILImage
            img_buf = BytesIO()
            PILImage.fromarray(vis.astype("uint8")).save(img_buf, format="PNG")
            img_buf.seek(0)
            story.append(Paragraph("Visual Evidence", h2))
            story.append(RLImage(img_buf, width=17 * cm, height=17 * cm * vis.shape[0] / vis.shape[1]))
        except Exception as e:
            logger.warning(f"Could not embed visual evidence in PDF: {e}")

    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().isoformat()}", styles["Italic"]))

    doc.build(story)
    return buf.getvalue()


def generate_geotiff_bytes(result: Dict[str, Any], reference_meta: Optional[Dict[str, Any]] = None) -> Optional[bytes]:
    """Export the change mask (or visual evidence) as a GeoTIFF and return raw bytes."""
    from src.utils.geotiff_utils import save_array_as_geotiff, RASTERIO_AVAILABLE

    if not RASTERIO_AVAILABLE:
        return None

    array = result.get("change_mask")
    if array is None:
        array = result.get("visual_evidence")
    if array is None:
        return None

    tmp_path = Path("/tmp/satquery_export.tif")
    out = save_array_as_geotiff(np.asarray(array), tmp_path, reference_meta)
    if out is None:
        return None
    data = out.read_bytes()
    try:
        out.unlink()
    except Exception:
        pass
    return data
