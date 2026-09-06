"""
Model Registry for SatQuery AI
Simple in-memory + JSON based registry of specialist models.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = {
    "rs_vlm_heuristic": {
        "name": "RS-VLM Heuristic + HF Fallback",
        "type": "vlm",
        "capabilities": ["vqa", "caption", "grounding", "optical", "sar", "single"],
        "description": "Lightweight VLM wrapper with strong remote-sensing heuristics",
        "status": "active",
        "priority": 1,
    },
    "change_detector_v1": {
        "name": "Structural Change Detector",
        "type": "change",
        "capabilities": ["change_mask", "bi_temporal", "change_description"],
        "description": "SSIM + absolute difference based change detection",
        "status": "active",
        "priority": 1,
    },
    "optical_sar_analyzer": {
        "name": "Optical-SAR Complementary Analyzer",
        "type": "fusion",
        "capabilities": ["optical_sar", "cross_modal"],
        "description": "Side-by-side + complementary information extraction",
        "status": "active",
        "priority": 1,
    },
    "captioner_v1": {
        "name": "Scene Captioner",
        "type": "caption",
        "capabilities": ["caption", "single"],
        "description": "Remote-sensing oriented scene description",
        "status": "active",
        "priority": 2,
    },
}


class ModelRegistry:
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path
        self.models: Dict[str, Dict[str, Any]] = DEFAULT_REGISTRY.copy()
        if registry_path and registry_path.exists():
            self.load(registry_path)

    def load(self, path: Path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.models.update(data)
            logger.info(f"Loaded registry from {path}")
        except Exception as e:
            logger.warning(f"Could not load registry: {e}")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.models, f, indent=2)

    def list_models(self, capability: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        for mid, info in self.models.items():
            if info.get("status") != "active":
                continue
            if capability and capability not in info.get("capabilities", []):
                continue
            entry = {"id": mid, **info}
            result.append(entry)
        result.sort(key=lambda x: x.get("priority", 99))
        return result

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.models.get(model_id)

    def select_best(self, required_capabilities: List[str]) -> Optional[str]:
        """Select best model that covers the required capabilities."""
        candidates = []
        for mid, info in self.models.items():
            if info.get("status") != "active":
                continue
            caps = set(info.get("capabilities", []))
            if all(c in caps for c in required_capabilities):
                candidates.append((info.get("priority", 99), mid))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]
