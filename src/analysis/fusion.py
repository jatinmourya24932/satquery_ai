"""
Fusion module - combines outputs from specialist tools into final answer.
"""

from typing import List, Dict, Any


def fuse_results(results: List[Dict[str, Any]], query: str, task_info: Dict[str, Any]) -> Dict[str, Any]:
    """Combine one or more specialist-tool outputs into a single answer."""
    if not results:
        return {"answer": "No results generated.", "confidence": 0.0}

    successful = [r for r in results if r.get("success")]
    if not successful:
        err = results[0].get("error", "Unknown error")
        return {"answer": f"Analysis failed: {err}", "confidence": 0.0}

    answer_parts, confidences = [], []
    for r in successful:
        if "answer" in r:
            answer_parts.append(r["answer"])
        elif "description" in r:
            answer_parts.append(r["description"])
        confidences.append(r.get("confidence", 0.5))

    fused = {
        "answer": "\n\n".join(answer_parts),
        "confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.5,
    }

    for r in successful:
        if "visualization" in r:
            fused["visual_evidence"] = r["visualization"]
        if "change_mask" in r:
            fused["change_mask"] = r["change_mask"]

        # Multiple boxes preferred
        if "bboxes" in r and r["bboxes"]:
            fused["bboxes"] = r["bboxes"]
            fused["bbox"] = r["bboxes"][0]["bbox"]
        elif "bbox" in r and r["bbox"] is not None:
            fused["bbox"] = r["bbox"]

    return fused


def build_quantitative_summary(
    results: List[Dict[str, Any]],
    geo_stats: Dict[str, Any] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"metrics": {}}

    for r in results:
        if not r.get("success"):
            continue

        if "change_ratio" in r:
            summary["metrics"]["change_ratio_pct"] = round(r["change_ratio"] * 100, 2)
        if "image_stats" in r:
            summary["metrics"].update({f"image_{k}": v for k, v in r["image_stats"].items()})
        if "stats" in r:
            for k, v in r["stats"].items():
                if isinstance(v, (int, float)):
                    summary["metrics"][k] = round(v, 2)
                elif isinstance(v, list):
                    summary["metrics"][k] = [
                        round(x, 2) if isinstance(x, (int, float)) else x for x in v
                    ]
        summary["metrics"].setdefault("confidence", r.get("confidence"))

        if "bboxes" in r and r["bboxes"]:
            summary["metrics"]["num_regions_detected"] = len(r["bboxes"])

    if geo_stats:
        summary["area_analysis"] = geo_stats

    summary["num_specialist_outputs"] = len([r for r in results if r.get("success")])
    return summary
