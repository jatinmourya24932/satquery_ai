"""
Task Classifier - decides primary task from query + input type.
"""

from typing import Dict, Any, List

CHANGE_KEYWORDS = [
    "change", "changed", "difference", "differ", "before", "after",
    "increased", "decreased", "increase", "decrease", "compare", "comparison",
    "over time", "time1", "time2", "t1", "t2",
]
OPTICAL_SAR_KEYWORDS = [
    "optical and sar", "sar and optical", "cross-modal", "cross modal",
    "fusion", "both images together",
]
GROUNDING_KEYWORDS = [
    "highlight", "locate", "where is", "where are", "show me the", "show the",
    "ground", "point out", "mark the", "detect", "find", "find all",
    "water bodies", "built-up", "vegetation", "draw box", "bounding box",
]
CAPTION_KEYWORDS = [
    "describe", "caption", "what does", "scene", "land cover", "land-cover", "summari",
]
QUANT_KEYWORDS = [
    "how much", "how many", "percentage", "percent", "area", "statistics",
    "stats", "quantify", "metrics", "number of",
]


def classify_task(query: str, input_type: str, modalities: List[str] = None) -> Dict[str, Any]:
    q = query.lower().strip()
    modalities = modalities or []

    both_modalities_present = "SAR" in modalities and "Optical" in modalities

    if (
        input_type == "optical_sar"
        or both_modalities_present
        or any(k in q for k in OPTICAL_SAR_KEYWORDS)
        or ("optical" in q and "sar" in q)
    ):
        primary = "optical_sar"
        plan = "Detected Optical + SAR → routing to Optical-SAR Fusion Model."
    elif input_type == "bi_temporal" or any(k in q for k in CHANGE_KEYWORDS):
        primary = "change"
        plan = "Detected temporal/comparison intent → routing to Change Detection Model."
    elif any(k in q for k in GROUNDING_KEYWORDS):
        primary = "grounding"
        plan = "Detected localization intent → routing to multi-region grounding."
    elif any(k in q for k in CAPTION_KEYWORDS):
        primary = "caption"
        plan = "Detected descriptive intent → routing to RS-VLM captioning."
    else:
        primary = "vqa"
        plan = "General question → routing to RS-VLM (Core) for VQA."

    needs_quant_summary = primary in ("change", "optical_sar") or any(k in q for k in QUANT_KEYWORDS)

    return {
        "primary_task": primary,
        "input_type": input_type,
        "query": query,
        "plan": plan,
        "needs_quant_summary": needs_quant_summary,
    }
