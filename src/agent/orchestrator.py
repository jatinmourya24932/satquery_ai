"""
Agentic Orchestrator for SatQuery AI
Interprets query, validates input, selects models, executes, fuses results,
and packages the final answer + visual evidence + quantitative summary +
execution trace - i.e. it *is* the "AI Agent Orchestration Layer (The Brain)"
box in the architecture diagram (Query Understanding -> Input Analyzer ->
Task Planner -> Tool/Model Selector -> Execution Manager).
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import logging
from datetime import datetime

from src.utils.validation import validate_inputs, get_input_summary
from src.utils.geotiff_utils import read_geospatial_metadata, changed_area_stats
from src.utils.visualization import draw_bbox_on_image, side_by_side
from src.models.registry import ModelRegistry
from src.analysis.single_image import run_single_image_analysis, load_image
from src.analysis.change_detection import run_change_detection
from src.analysis.optical_sar import analyze_optical_sar
from src.analysis.fusion import fuse_results, build_quantitative_summary
from src.agent.task_classifier import classify_task

logger = logging.getLogger(__name__)


class ExecutionTrace:
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add(self, step: str, details: Dict[str, Any] = None):
        self.steps.append({
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        })

    def summary(self) -> Dict[str, Any]:
        return {
            "total_steps": len(self.steps),
            "duration_sec": round(time.time() - self.start_time, 2),
            "steps": self.steps,
        }


class SatQueryOrchestrator:
    def __init__(self):
        self.registry = ModelRegistry()
        self.trace = None

    def run(
        self,
        image_paths: List[Path],
        query: str,
        user_modality_hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.trace = ExecutionTrace()
        self.trace.add("start", {"query": query, "num_images": len(image_paths)})

        # --- 2. Data Ingestion & Preprocessing (validation + geo metadata) ---
        validation = validate_inputs(image_paths, query, user_modality_hints)
        self.trace.add("validation", validation.to_dict())

        if not validation.is_valid:
            return {
                "success": False,
                "error": "Input validation failed",
                "validation": validation.to_dict(),
                "execution_summary": self.trace.summary(),
            }

        geo_meta = [read_geospatial_metadata(p) for p in image_paths]
        self.trace.add("geospatial_metadata", {
            "sources": [m.get("georeferencing_source") for m in geo_meta],
            "crs": [m.get("crs") for m in geo_meta],
        })

        # --- 3. AI Agent Orchestration Layer: Query Understanding + Task Planner ---
        task_info = classify_task(query, validation.input_type, validation.modalities)
        self.trace.add("task_classification", task_info)

        primary = task_info["primary_task"]
        results: List[Dict[str, Any]] = []
        models_used: List[str] = []

        # --- 4. Specialist Tools / Models Layer: Tool/Model Selector + Execution Manager ---
        try:
            if primary == "change" and validation.image_count == 2:
                self.trace.add("select_model", {"model": "change_detector_v1"})
                res = run_change_detection(image_paths[0], image_paths[1], query)
                results.append(res)
                models_used.append(res.get("model_used", "change_detector_v1"))

            elif primary == "optical_sar" and validation.image_count == 2:
                self.trace.add("select_model", {"model": "optical_sar_analyzer"})
                mods = validation.modalities
                if mods[0] == "SAR" and mods[1] == "Optical":
                    opt_p, sar_p = image_paths[1], image_paths[0]
                else:
                    opt_p, sar_p = image_paths[0], image_paths[1]
                res = analyze_optical_sar(opt_p, sar_p, query)
                results.append(res)
                models_used.append(res.get("model_used", "optical_sar_analyzer"))

            elif primary == "grounding":
                self.trace.add("select_model", {"model": "rs_vlm (grounding)"})
                res = run_single_image_analysis(image_paths[0], query, task="grounding")
                results.append(res)
                models_used.append(res.get("model_used", "rs_vlm_heuristic"))

            elif primary == "caption":
                self.trace.add("select_model", {"model": "rs_vlm (caption)"})
                res = run_single_image_analysis(image_paths[0], query, task="caption")
                results.append(res)
                models_used.append(res.get("model_used", "rs_vlm_heuristic"))

            else:  # vqa default
                self.trace.add("select_model", {"model": "rs_vlm (vqa)"})
                res = run_single_image_analysis(image_paths[0], query, task="vqa")
                results.append(res)
                models_used.append(res.get("model_used", "rs_vlm_heuristic"))

                if validation.image_count == 2 and "change" not in query.lower():
                    self.trace.add("extra_analysis", {"note": "Also ran quick change check"})
                    extra = run_change_detection(image_paths[0], image_paths[1], query)
                    if extra.get("success"):
                        results.append(extra)
                        models_used.append(extra.get("model_used"))

        except Exception as e:
            logger.exception("Execution failed")
            self.trace.add("execution_error", {"error": str(e)})
            return {
                "success": False,
                "error": str(e),
                "execution_summary": self.trace.summary(),
            }

        # --- Fusion: Natural Language Answer ---
        fused = fuse_results(results, query, task_info)
        self.trace.add("fusion", {"num_results": len(results)})

        # --- 5. Geospatial Processing Layer: area stats + evidence panel rendering ---
        geo_area_stats = None
        if fused.get("change_mask") is not None:
            geo_area_stats = changed_area_stats(fused["change_mask"], geo_meta[0])
            self.trace.add("geospatial_processing", {"change_area_stats": geo_area_stats})

        visual_evidence = self._build_visual_evidence(fused, image_paths)

        # --- Quantitative Summary (Output Layer) ---
        quant_summary = build_quantitative_summary(results, geo_area_stats)
        self.trace.add("quantitative_summary", {"num_metrics": len(quant_summary.get("metrics", {}))})

        # --- 6. Output Layer: package final answer ---
        final = {
            "success": True,
            "query": query,
            "input_type": validation.input_type,
            "modalities": validation.modalities,
            "primary_task": primary,
            "plan": task_info.get("plan"),
            "answer": fused["answer"],
            "confidence": fused["confidence"],
            "visual_evidence": visual_evidence,
            "bbox": fused.get("bbox"),
            "change_mask": fused.get("change_mask"),
            "models_used": models_used,
            "validation_summary": get_input_summary(validation),
            "quantitative_summary": quant_summary,
            "geospatial_metadata": [
                {k: v for k, v in m.items() if k != "transform"} for m in geo_meta
            ],
            "raw_results": results,
        }

        self.trace.add("complete", {"confidence": fused["confidence"]})
        final["execution_summary"] = self.trace.summary()
        return final

    def _build_visual_evidence(self, fused: Dict[str, Any], image_paths: List[Path]):
        """Build the final annotated 'Visual Evidence' panel (Map Rendering)."""
        try:
            base = fused.get("visual_evidence")

            if base is not None:
                if fused.get("bbox") is not None:
                    try:
                        return draw_bbox_on_image(base, fused["bbox"], label="Region of interest")
                    except Exception:
                        return base
                return base

            if fused.get("bbox") is not None:
                _, arr = load_image(image_paths[0])
                return draw_bbox_on_image(arr, fused["bbox"], label="Region of interest")

            # Fallback: just show the input image(s) side-by-side
            arrs, labels = [], []
            for i, p in enumerate(image_paths):
                _, arr = load_image(p)
                arrs.append(arr)
                labels.append(f"Input {i+1}")
            if arrs:
                return side_by_side(*arrs, labels=labels)
            return None
        except Exception as e:
            logger.warning(f"Could not build visual evidence panel: {e}")
            return fused.get("visual_evidence")
