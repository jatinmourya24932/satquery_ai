# SatQuery AI
### An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis

**Smart India Hackathon 2026 | Problem Statement ID: 26167**  
**Organization:** Indian Space Research Organisation (ISRO)  
**Team:** Certified Bug Hunters  
**Theme:** Space Technology | Category: Software

---

## Overview

**SatQuery AI** is an agentic vision-language system that answers natural-language queries about satellite imagery.

It automatically:

1. Validates input images (GeoTIFF / Optical / SAR / Bi-temporal)
2. Understands the user query
3. Selects the right specialist pipeline
4. Executes the analysis
5. Fuses results
6. Returns **evidence-grounded** answers with visual highlights, confidence scores, and a full execution summary

### Supported Capabilities (MVP)

- Single-image Visual Question Answering (VQA)
- Image Captioning / Scene Description
- Text-guided Multi-region Grounding (Water / Built-up / Vegetation)
- Bi-temporal Change Detection + Change Description
- Optical + SAR Cross-modal Analysis
- Agentic Orchestration with Auditable Execution Trace
- Support for both normal images (PNG/JPG) and GeoTIFF

---

## Tech Stack

| Layer              | Technology                                      |
|--------------------|-------------------------------------------------|
| **Frontend**       | Next.js 13 + Tailwind CSS + shadcn/ui           |
| **Backend**        | FastAPI (Python)                                |
| **Core VLM**       | Qwen3-VL-4B-Instruct + BigEarthNet LoRA         |
| **Orchestration**  | Rule-based Task Classifier + Agentic Router     |
| **Geospatial**     | rasterio + GeoTIFF support                      |
| **Visualization**  | Multi-region Bounding Boxes + Evidence Panels   |

---

## Architecture Flow

```
User Query + Image(s)
        ↓
Input Validation (GeoTIFF / Optical / SAR / Bi-temporal)
        ↓
Task Classifier (VQA / Caption / Grounding / Change / Optical-SAR)
        ↓
Agentic Orchestrator → Selects specialist pipeline
        ↓
Model Execution (Qwen3-VL + dedicated modules)
        ↓
Result Fusion
        ↓
Evidence-Grounded Output
(Answer + Bounding Boxes + Confidence + Execution Summary)
```

---

## Quick Start

### 1. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
cd SatQuery_Frontend                     # or your Next.js project folder
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`  
Backend API runs at `http://localhost:8000`

---

## Environment Variables

Create a `.env` file in the backend root:

```env
SATQUERY_USE_HF=1
SATQUERY_HF_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
SATQUERY_LORA_ADAPTER=weights/bigearthnet_lora
SATQUERY_LOAD_4BIT=1
```

---

## Supported File Formats

| Format              | Supported | Notes                          |
|---------------------|-----------|--------------------------------|
| PNG / JPG / JPEG    | Yes       | Standard RGB images            |
| GeoTIFF (.tif/.tiff)| Yes       | Full geospatial metadata support |
| SAR images          | Yes       | Cross-modal analysis           |

---

## Example Queries

```
Detect water bodies
Identify built-up and water regions
Highlight vegetation
Where are the urban areas?
Describe this satellite image
Compare the two images and show changes
```

---

## Mandatory PS Coverage

| Requirement                         | Status in MVP |
|-------------------------------------|---------------|
| Single-image VQA                    | Yes           |
| Captioning / Grounding              | Yes           |
| Bi-temporal Change Analysis         | Yes           |
| Optical + SAR Cross-modal           | Yes           |
| Agentic Model / Pipeline Selection  | Yes           |
| Input Validation                    | Yes           |
| Evidence + Confidence + Summary     | Yes           |
| Remote-sensing adaptation           | Yes (Qwen3-VL + BigEarthNet LoRA) |

---

## Key Innovations

- **Agentic Orchestration** → Automatically routes query to the correct specialist pipeline
- **Multimodal Support** → Single image | Bi-temporal | Optical + SAR
- **Remote Sensing Adaptation** → Fine-tuned on BigEarthNet
- **Evidence-Grounded Output** → Answer + Highlighted Regions + Confidence
- **Fully Auditable** → Complete Execution Trace

---

## Project Structure (Backend)

```
satquery_ai/
├── api.py                          # FastAPI entry point
├── src/
│   ├── agent/
│   │   ├── orchestrator.py         # Main agentic brain
│   │   └── task_classifier.py      # Query intent classification
│   ├── analysis/
│   │   ├── single_image.py         # VQA / Caption / Grounding
│   │   ├── change_detection.py     # Bi-temporal analysis
│   │   ├── optical_sar.py          # Cross-modal analysis
│   │   └── fusion.py               # Result fusion
│   ├── models/
│   │   └── vlm_wrapper.py          # Qwen3-VL + LoRA wrapper
│   └── utils/
│       ├── validation.py
│       ├── visualization.py        # Multi-box drawing
│       └── geotiff_utils.py
└── weights/
    └── bigearthnet_lora/           # LoRA adapter weights
```

---

## Notes

- Grounding currently uses improved multi-region heuristic detection (color + contour based) because the base Qwen3-VL model does not natively output bounding boxes.
- The architecture is modular — additional specialist models can be plugged in easily.
- Designed as a 1–2 day MVP for Smart India Hackathon 2026.

---

**Built for SIH 2026 – Space Technology Theme**  
**Team: Certified Bug Hunters**
