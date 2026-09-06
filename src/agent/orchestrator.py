"""
Agentic Orchestrator for SatQuery AI
Interprets query, validates input, selects models, executes, fuses results.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import logging
from datetime import datetime

from src.utils.validation import validate_inputs, get_input_summary, ValidationResult
from src.models.registry import ModelRegistry
from src.analysis.single_image import run_single_image_analysis
from src.analysis.change_detection import run_change_detection
from src.analysis.optical_sar import analyze_optical_sar

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


def classify_task(query: str, input_type: str) -> Dict[str, Any]:
    """
    Simple but effective task classification.
    Returns primary_task and secondary tasks.
    """
    q = query.lower().strip()

    # Priority rules
    if input_type == "bi_temporal" or any(k in q for k in ["change", "changed", "difference", "before", "after", "increased", "decreased"]):
        primary = "change"
    elif input_type == "optical_sar" or ("optical" in q and "sar" in q) or "cross-modal" in q or "both" in q:
        primary = "optical_sar"
    elif any(k in q for k in ["highlight", "locate", "where is", "show me the", "ground", "region"]):
        primary = "grounding"
    elif any(k in q for k in ["describe", "caption", "what does", "scene", "land cover", "land-cover"]):
        primary = "caption"
    else:
        primary = "vqa"

    return {
        "primary_task": primary,
        "input_type": input_type,
        "query": query,
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

        # 1. Validation
        validation = validate_inputs(image_paths, query, user_modality_hints)
        self.trace.add("validation", validation.to_dict())

        if not validation.is_valid:
            return {
                "success": False,
                "error": "Input validation failed",
                "validation": validation.to_dict(),
                "execution_summary": self.trace.summary(),
            }

        # 2. Task Classification
        task_info = classify_task(query, validation.input_type)
        self.trace.add("task_classification", task_info)

        primary = task_info["primary_task"]
        results = []
        models_used = []

        # 3. Model Selection & Execution
        try:
            if primary == "change" and validation.image_count == 2:
                self.trace.add("select_model", {"model": "change_detector_v1"})
                res = run_change_detection(image_paths[0], image_paths[1], query)
                results.append(res)
                models_used.append(res.get("model_used", "change_detector_v1"))

            elif primary == "optical_sar" and validation.image_count == 2:
                self.trace.add("select_model", {"model": "optical_sar_analyzer"})
                # Assume first is optical, second is SAR (or swap based on modality)
                mods = validation.modalities
                if mods[0] == "SAR" and mods[1] == "Optical":
                    opt_p, sar_p = image_paths[1], image_paths[0]
                else:
                    opt_p, sar_p = image_paths[0], image_paths[1]
                res = analyze_optical_sar(opt_p, sar_p, query)
                results.append(res)
                models_used.append(res.get("model_used", "optical_sar_analyzer"))

            elif primary == "grounding":
                self.trace.add("select_model", {"model": "rs_vlm_heuristic (grounding)"})
                res = run_single_image_analysis(image_paths[0], query, task="grounding")
                results.append(res)
                models_used.append(res.get("model_used", "rs_vlm_heuristic"))

            elif primary == "caption":
                self.trace.add("select_model", {"model": "rs_vlm_heuristic (caption)"})
                res = run_single_image_analysis(image_paths[0], query, task="caption")
                results.append(res)
                models_used.append(res.get("model_used", "rs_vlm_heuristic"))

            else:  # vqa default
                self.trace.add("select_model", {"model": "rs_vlm_heuristic (vqa)"})
                res = run_single_image_analysis(image_paths[0], query, task="vqa")
                results.append(res)
                models_used.append(res.get("model_used", "rs_vlm_heuristic"))

                # If two images and user asked something general, also run a light change check
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

        # 4. Fusion
        fused = self._fuse(results, query, task_info)
        self.trace.add("fusion", {"num_results": len(results)})

        # 5. Final package
        final = {
            "success": True,
            "query": query,
            "input_type": validation.input_type,
            "modalities": validation.modalities,
            "primary_task": primary,
            "answer": fused["answer"],
            "confidence": fused["confidence"],
            "visual_evidence": fused.get("visual_evidence"),
            "bbox": fused.get("bbox"),
            "change_mask": fused.get("change_mask"),
            "models_used": models_used,
            "validation_summary": get_input_summary(validation),
            "execution_summary": self.trace.summary(),
            "raw_results": results,
        }

        self.trace.add("complete", {"confidence": fused["confidence"]})
        return final

    def _fuse(self, results: List[Dict], query: str, task_info: Dict) -> Dict[str, Any]:
        if not results:
            return {"answer": "No results generated.", "confidence": 0.0}

        successful = [r for r in results if r.get("success")]
        if not successful:
            return {
                "answer": "Analysis failed: " + results[0].get("error", "Unknown error"),
                "confidence": 0.0,
            }

        # Simple fusion: take primary answer + average confidence
        primary = successful[0]
        answer_parts = []
        confidences = []

        for r in successful:
            if "answer" in r:
                answer_parts.append(r["answer"])
            elif "description" in r:
                answer_parts.append(r["description"])
            confidences.append(r.get("confidence", 0.5))

        answer = "\n\n".join(answer_parts)
        confidence = sum(confidences) / len(confidences) if confidences else 0.5

        fused = {
            "answer": answer,
            "confidence": round(confidence, 2),
        }

        # Carry visual evidence
        for r in successful:
            if "visualization" in r:
                fused["visual_evidence"] = r["visualization"]
            if "change_mask" in r:
                fused["change_mask"] = r["change_mask"]
            if "bbox" in r:
                fused["bbox"] = r["bbox"]

        return fused
