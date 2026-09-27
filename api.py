"""
SatQuery AI - FastAPI backend
Exposes the existing SatQueryOrchestrator as REST API for the Next.js frontend.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from src.agent.orchestrator import SatQueryOrchestrator

app = FastAPI(
    title="SatQuery AI API",
    description="REST API for SatQuery AI agentic satellite image analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLES_DIR = ROOT / "data" / "samples"
SAMPLE_MAP = {
    "optical": SAMPLES_DIR / "sample_optical.png",
    "optical_t2": SAMPLES_DIR / "sample_optical_t2.png",
    "sar": SAMPLES_DIR / "sample_sar.png",
}


def _numpy_to_base64_png(arr: Any) -> Optional[str]:
    if arr is None:
        return None
    try:
        if isinstance(arr, np.ndarray):
            a = arr
            if a.dtype != np.uint8:
                if a.max() <= 1.0:
                    a = (a * 255).clip(0, 255).astype(np.uint8)
                else:
                    a = a.clip(0, 255).astype(np.uint8)
            if a.ndim == 2:
                img = Image.fromarray(a, mode="L")
            elif a.ndim == 3 and a.shape[2] in (3, 4):
                img = Image.fromarray(a)
            else:
                img = Image.fromarray(a.squeeze().astype(np.uint8))
        elif isinstance(arr, Image.Image):
            img = arr
        else:
            return None

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def _sanitize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "success": result.get("success", False),
        "query": result.get("query"),
        "input_type": result.get("input_type"),
        "modalities": result.get("modalities"),
        "primary_task": result.get("primary_task"),
        "plan": result.get("plan"),
        "answer": result.get("answer"),
        "confidence": result.get("confidence"),
        "bbox": result.get("bbox"),
        "models_used": result.get("models_used") or [],
        "validation_summary": result.get("validation_summary"),
        "quantitative_summary": result.get("quantitative_summary") or {},
        "geospatial_metadata": result.get("geospatial_metadata") or [],
        "execution_summary": result.get("execution_summary") or {},
        "error": result.get("error"),
    }
    out["visual_evidence_b64"] = _numpy_to_base64_png(result.get("visual_evidence"))
    out["change_mask_b64"] = _numpy_to_base64_png(result.get("change_mask"))
    return out


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "image.png").suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = upload.file.read()
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SatQuery AI API",
        "samples_available": [k for k, p in SAMPLE_MAP.items() if p.exists()],
    }


@app.get("/samples")
def list_samples():
    return {
        "samples": [
            {"id": k, "exists": p.exists(), "name": p.name}
            for k, p in SAMPLE_MAP.items()
        ]
    }


@app.post("/analyze")
async def analyze(
    query: str = Form(...),
    images: Optional[List[UploadFile]] = File(None),
    use_samples: Optional[str] = Form(None),
    modality_hints: Optional[str] = Form(None),
):
    query = (query or "").strip()
    if len(query) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    paths: List[Path] = []
    temp_paths: List[Path] = []

    try:
        if images:
            for f in images:
                if f and f.filename:
                    p = _save_upload(f)
                    paths.append(p)
                    temp_paths.append(p)

        if use_samples:
            for sid in [s.strip() for s in use_samples.split(",") if s.strip()]:
                sample_path = SAMPLE_MAP.get(sid)
                if not sample_path or not sample_path.exists():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown or missing sample id: {sid}",
                    )
                paths.append(sample_path)

        if not paths:
            default = SAMPLE_MAP["optical"]
            if not default.exists():
                raise HTTPException(
                    status_code=400,
                    detail="No images provided and no sample files found on server",
                )
            paths.append(default)

        hints: Optional[List[str]] = None
        if modality_hints:
            hints = [h.strip() for h in modality_hints.split(",") if h.strip()]
            if len(hints) != len(paths):
                hints = None

        orchestrator = SatQueryOrchestrator()
        result = orchestrator.run(paths, query, user_modality_hints=hints)
        payload = _sanitize_result(result)
        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "answer": None,
                "confidence": 0,
            },
        )
    finally:
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SATQUERY_API_PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)