# SatQuery AI
### An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis

**Smart India Hackathon 2026 | Problem Statement ID: 26167**  
**Organization:** Indian Space Research Organisation (ISRO)

---

## Overview

SatQuery AI is an **agentic** vision-language system that answers natural-language queries about satellite imagery.  
It automatically:

1. Validates input images (GeoTIFF / Optical / SAR / Bi-temporal)
2. Understands the user query
3. Selects the right specialist models
4. Executes them
5. Fuses results
6. Returns **evidence-grounded** answers with confidence + execution summary

### Supported Capabilities (MVP)

- Single-image Visual Question Answering (VQA)
- Image Captioning / Scene Description
- Text-guided Region Grounding (basic)
- Bi-temporal Change Detection + Change Description
- Optical + SAR Cross-modal Analysis
- Agentic Orchestration + Auditable Execution Trace

---

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## MVP Design Decisions (1-2 Day Prototype)

- **Orchestrator**: Hybrid (rule-based + lightweight classification)
- **VLM**: Hugging Face ready models + strong heuristic fallback
- **Change Detection**: Structural similarity + absolute difference
- **Optical-SAR**: Side-by-side analysis + complementary extraction
- **Fine-tuning**: Placeholder ready for BigEarthNet adaptation
- **Output**: Text + visual evidence + confidence + execution summary

---

## Mandatory PS Coverage

| Requirement                        | Status in MVP |
|------------------------------------|---------------|
| Single-image VQA                   | Yes           |
| Captioning or Grounding            | Yes           |
| Bi-temporal Change Analysis        | Yes           |
| Optical + SAR Cross-modal          | Yes           |
| Agentic Model Selection            | Yes           |
| Input Validation                   | Yes           |
| Evidence + Confidence + Summary    | Yes           |
| Remote-sensing adaptation ready    | Yes (plug-in) |

---

**Built for SIH 2026 – Space Technology Theme**
