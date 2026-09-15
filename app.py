"""
SatQuery AI - Main Streamlit Application
Interactive Vision-Language Assistant for Multimodal Remote Sensing
"""

import streamlit as st
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import sys
import os

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

# MUST load .env before importing VLM wrapper (it reads env at import time)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

from src.agent.orchestrator import SatQueryOrchestrator
from src.utils.validation import get_input_summary
from src.utils.geotiff_utils import read_geospatial_metadata
from src.ui.components import render_metrics_row, render_quantitative_summary, render_downloads

st.set_page_config(
    page_title="SatQuery AI | ISRO SIH 2026",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0B3D91;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0B3D91;
    }
    .success-box {
        background: #e6f4ea;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #34a853;
    }
    .warning-box {
        background: #fef7e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #fbbc04;
    }
</style>
""", unsafe_allow_html=True)


def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to temp and return path."""
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return Path(tmp.name)


def render_results(result, paths):
    """Renders the full Output Layer for a given orchestrator result.
    Pulled into its own function so results persist across tab switches /
    download-button clicks (Streamlit reruns the whole script on each interaction)."""
    if not result.get("success"):
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        if "validation" in result:
            st.json(result["validation"])
        return

    st.markdown("### ✅ Analysis Complete")

    render_metrics_row(result)
    if result.get("plan"):
        st.caption(f"🧠 Agent plan: {result['plan']}")

    st.markdown("---")

    # Answer
    st.subheader("📝 Answer")
    st.markdown(f'<div class="success-box">{result.get("answer", "No answer generated.")}</div>', unsafe_allow_html=True)

    # Visual Evidence
    st.subheader("🖼️ Visual Evidence")
    if result.get("visual_evidence") is not None:
        vis = result["visual_evidence"]
        if isinstance(vis, np.ndarray):
            st.image(vis, caption="Analysis Visualization (Input(s) + Evidence)", use_container_width=True)
    elif result.get("change_mask") is not None:
        st.image(result["change_mask"], caption="Change Mask", use_container_width=True, clamp=True)
    else:
        cols = st.columns(len(paths))
        for i, p in enumerate(paths):
            with cols[i]:
                st.image(str(p), caption=f"Input {i+1}", use_container_width=True)

    if result.get("bbox"):
        st.info(f"Grounding BBox (demo): {result['bbox']}")

    # Quantitative Summary (Output Layer)
    st.markdown("---")
    st.subheader("📊 Quantitative Summary")
    render_quantitative_summary(result)

    # Models + Validation
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🤖 Models Used")
        for m in result.get("models_used", []):
            st.markdown(f"- `{m}`")
    with c2:
        st.subheader("✅ Validation Summary")
        st.text(result.get("validation_summary", ""))

    # Downloadables (Output Layer)
    st.markdown("---")
    render_downloads(result, reference_meta=st.session_state.get("last_geo_meta_raw"))


def main():
    st.markdown('<div class="main-header">🛰️ SatQuery AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Agentic Vision-Language Assistant for Multimodal Remote Sensing Image Analysis<br>'
        '<b>ISRO | Smart India Hackathon 2026 | PS-26167</b></div>',
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("**Input Type Hints (optional)**")
        modality_1 = st.selectbox("Image 1 Modality", ["Auto", "Optical", "SAR"], index=0)
        modality_2 = st.selectbox("Image 2 Modality", ["Auto", "Optical", "SAR"], index=0)

        st.markdown("---")
        st.markdown("**About this MVP**")
        st.info(
            "This is a 1-2 day prototype demonstrating:\n"
            "- Agentic orchestration\n"
            "- Input validation\n"
            "- Single-image VQA / Caption / Grounding\n"
            "- Bi-temporal Change Analysis\n"
            "- Optical + SAR Cross-modal Analysis\n"
            "- Evidence + Confidence + Execution Trace\n\n"
            "Real BigEarthNet fine-tuned models can be plugged in later."
        )

        st.markdown("---")
        st.markdown("**Representative Queries**")
        st.code("Describe the land-cover and major objects visible in this image.", language=None)
        st.code("What changed between these two dates, and where did the change occur?", language=None)
        st.code("Use the optical and SAR images together to identify built-up and water regions.", language=None)
        st.code("Has the built-up area increased, decreased, or remained unchanged?", language=None)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Query & Analyze", "🕵️ Execution Trace", "📋 System Info"])

    with tab1:
        col_upload, col_query = st.columns([1, 1])

        with col_upload:
            st.subheader("📤 Upload Satellite Image(s)")
            uploaded_files = st.file_uploader(
                "Upload 1 or 2 images (GeoTIFF / TIFF / PNG / JPEG)",
                type=["tif", "tiff", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )

            if uploaded_files:
                st.success(f"{len(uploaded_files)} file(s) uploaded")
                for f in uploaded_files:
                    st.caption(f"• {f.name} ({f.size / 1024:.1f} KB)")

        with col_query:
            st.subheader("💬 Natural Language Query")
            query = st.text_area(
                "Enter your question about the image(s)",
                height=120,
                placeholder="e.g. Describe the land-cover... / What changed between these two dates? / Identify built-up and water using both optical and SAR..."
            )

            run_btn = st.button("🚀 Run SatQuery Analysis", type="primary", use_container_width=True)

        if run_btn:
            if not uploaded_files:
                st.error("Please upload at least one image.")
            elif not query or len(query.strip()) < 3:
                st.error("Please enter a meaningful query.")
            else:
                with st.spinner("Agent is thinking... Validating → Classifying → Selecting models → Executing → Fusing..."):
                    # Save files
                    paths = [save_uploaded_file(f) for f in uploaded_files]

                    # Modality hints
                    hints = []
                    if modality_1 != "Auto":
                        hints.append(modality_1)
                    if len(uploaded_files) > 1 and modality_2 != "Auto":
                        hints.append(modality_2)
                    if len(hints) != len(paths):
                        hints = None

                    # Run orchestrator
                    orchestrator = SatQueryOrchestrator()
                    result = orchestrator.run(paths, query, user_modality_hints=hints)

                    # Store in session (keep a *raw* geo-metadata copy with the
                    # rasterio transform intact, for GeoTIFF export downstream)
                    st.session_state["last_result"] = result
                    st.session_state["last_paths"] = paths
                    st.session_state["last_geo_meta_raw"] = read_geospatial_metadata(paths[0])

                render_results(result, paths)
        elif "last_result" in st.session_state:
            # Persist last analysis across reruns (tab switches, download clicks, etc.)
            render_results(st.session_state["last_result"], st.session_state.get("last_paths", []))

    with tab2:
        st.subheader("🕵️ Auditable Execution Trace")
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            summary = result.get("execution_summary", {})
            st.json(summary)

            st.markdown("#### Step-by-step")
            for i, step in enumerate(summary.get("steps", [])):
                with st.expander(f"Step {i+1}: {step.get('step')}", expanded=(i < 3)):
                    st.write(f"**Time:** {step.get('timestamp')}")
                    st.json(step.get("details", {}))
        else:
            st.info("Run an analysis first to see the execution trace.")

    with tab3:
        st.subheader("📋 System Information")
        st.markdown("""
        **SatQuery AI MVP** – Built for ISRO Smart India Hackathon 2026 (PS-26167)

        ### Architecture Layers (matches system diagram)
        1. **Input Layer** – natural-language query + Single Image / Bi-temporal Pair / Optical+SAR Pair
        2. **Data Ingestion & Preprocessing** – file validation, metadata extraction, georeferencing (real or synthetic CRS/transform), tiling/normalization
        3. **AI Agent Orchestration Layer (The Brain)** – Query Understanding → Input Analyzer → Task Planner → Tool/Model Selector → Execution Manager (`src/agent/`)
        4. **Specialist Tools / Models Layer** – RS-VLM (Core), Change Detection Model, Optical-SAR Fusion Model, backed by a Model Registry (`src/models/`, `src/analysis/`)
        5. **Geospatial Processing Layer** – raster operations, change-area statistics (pixels → m² / ha / km²), coordinate transforms, map rendering (`src/utils/geotiff_utils.py`, `src/utils/visualization.py`)
        6. **Output Layer** – Natural-language answer, Visual evidence (bbox / change heatmap / overlay), Quantitative summary, Execution summary, and **Downloadables (GeoTIFF, PDF, JSON)**

        ### Current MVP Design Notes
        - RS-VLM (Core) defaults to a fast heuristic backend so the app runs with **zero downloads**; set `SATQUERY_USE_HF=1` in `.env` to route captioning/VQA through a real Hugging Face VLM (BLIP by default) with automatic fallback to heuristics on any failure.
        - Change detection uses grayscale absolute-difference + morphological cleanup — swappable for ChangeFormer/ChangeMamba behind the same `run_change_detection()` interface.
        - Georeferencing: real GeoTIFFs use their embedded CRS/transform; plain PNG/JPEG inputs get a synthetic 10 m/pixel transform so area statistics and GeoTIFF export still work end-to-end.
        - Grounding boxes are heuristic (query-keyword driven) — swappable for a real referring-expression model.

        ### Next Upgrades
        1. Fine-tune InternVL / Qwen2.5-VL on BigEarthNet (LoRA) as the RS-VLM backend
        2. Integrate a deep change-detection model (ChangeFormer/ChangeMamba)
        3. Real referring-expression grounding
        4. True raster co-registration for bi-temporal pairs
        5. Better confidence calibration
        """)

        st.success("Prototype is ready for demonstration and iterative improvement.")


if __name__ == "__main__":
    main()
