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

from src.agent.orchestrator import SatQueryOrchestrator
from src.utils.validation import get_input_summary

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

                    # Store in session
                    st.session_state["last_result"] = result
                    st.session_state["last_paths"] = paths

                # Display results
                if not result.get("success"):
                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                    if "validation" in result:
                        st.json(result["validation"])
                else:
                    st.markdown("### ✅ Analysis Complete")

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Primary Task", result.get("primary_task", "—").upper())
                    m2.metric("Input Type", result.get("input_type", "—"))
                    m3.metric("Confidence", f"{result.get('confidence', 0):.0%}")
                    m4.metric("Models Used", len(result.get("models_used", [])))

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
                        # Show original images
                        cols = st.columns(len(paths))
                        for i, p in enumerate(paths):
                            with cols[i]:
                                st.image(str(p), caption=f"Input {i+1}", use_container_width=True)

                    if result.get("bbox"):
                        st.info(f"Grounding BBox (demo): {result['bbox']}")

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

        ### Architecture Highlights
        - **Agentic Orchestrator**: Query understanding → Task classification → Model selection → Execution → Fusion
        - **Input Validation**: Format, count, modality, basic geospatial checks
        - **Specialist Modules**:
          - Single-image VQA / Caption / Grounding
          - Bi-temporal Change Detection
          - Optical + SAR Cross-modal Analysis
        - **Output**: Text answer + Visual evidence + Confidence + Full execution summary

        ### Current MVP Limitations (by design for speed)
        - Uses strong heuristics + lightweight methods (real BigEarthNet fine-tuned VLM can be plugged later)
        - Change detection is classical (SSIM/diff) – replaceable with ChangeFormer/ChangeMamba
        - Grounding is demonstrative (central region)
        - Full GeoTIFF CRS / co-registration checks are basic

        ### Next Upgrades
        1. Fine-tune InternVL / Qwen2.5-VL on BigEarthNet.txt (LoRA)
        2. Integrate deep change detection model
        3. Real referring-expression grounding
        4. Downloadable PDF + GeoJSON report
        5. Better confidence calibration
        """)

        st.success("Prototype is ready for demonstration and iterative improvement.")


if __name__ == "__main__":
    main()
