"""
Reusable Streamlit UI components for the SatQuery AI Output Layer:
metrics row, quantitative summary table, execution-summary badge,
and the three downloadable artifacts (JSON / PDF / GeoTIFF).
"""

from typing import Dict, Any
import streamlit as st

from src.utils.report_generation import (
    generate_json_report,
    generate_pdf_report,
    generate_geotiff_bytes,
)


def render_metrics_row(result: Dict[str, Any]):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Primary Task", result.get("primary_task", "—").upper())
    m2.metric("Input Type", result.get("input_type", "—"))
    m3.metric("Confidence", f"{result.get('confidence', 0):.0%}")
    m4.metric("Models Used", len(result.get("models_used", [])))


def render_quantitative_summary(result: Dict[str, Any]):
    """Renders the 'Quantitative Summary: Metrics, statistics and insights' block."""
    quant = result.get("quantitative_summary") or {}
    metrics = quant.get("metrics", {})
    area = quant.get("area_analysis")

    if not metrics and not area:
        st.caption("No additional quantitative metrics for this query type.")
        return

    if metrics:
        cols = st.columns(min(4, max(1, len(metrics))))
        for i, (k, v) in enumerate(metrics.items()):
            if v is None:
                continue
            label = k.replace("_", " ").title()
            display_val = f"{v:.2f}" if isinstance(v, float) else str(v)
            cols[i % len(cols)].metric(label, display_val)

    if area:
        st.markdown("**Area / Geospatial Analysis**")
        st.table({
            "Metric": list(area.keys()),
            "Value": list(area.values()),
        })


def render_downloads(result: Dict[str, Any], reference_meta=None):
    """Renders the 'Downloadables: GeoTIFF, PDF, JSON' block of the Output Layer.

    `reference_meta` should be the *raw* metadata dict (including the
    rasterio `transform` object) from `read_geospatial_metadata`, e.g.
    computed fresh from the first input path - NOT the JSON-safe copy
    stored inside `result["geospatial_metadata"]`.
    """
    st.markdown("**⬇️ Downloadables**")
    d1, d2, d3 = st.columns(3)

    json_bytes = generate_json_report(result)
    d1.download_button(
        "📄 JSON Report", data=json_bytes, file_name="satquery_report.json",
        mime="application/json", use_container_width=True,
    )

    pdf_bytes = generate_pdf_report(result)
    if pdf_bytes:
        d2.download_button(
            "📕 PDF Report", data=pdf_bytes, file_name="satquery_report.pdf",
            mime="application/pdf", use_container_width=True,
        )
    else:
        d2.button("📕 PDF unavailable", disabled=True, use_container_width=True)

    tif_bytes = generate_geotiff_bytes(result, reference_meta)
    if tif_bytes:
        d3.download_button(
            "🗺️ GeoTIFF", data=tif_bytes, file_name="satquery_evidence.tif",
            mime="image/tiff", use_container_width=True,
        )
    else:
        d3.button("🗺️ GeoTIFF unavailable", disabled=True, use_container_width=True)
