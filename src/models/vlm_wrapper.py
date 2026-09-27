"""
RS-VLM (Core) wrapper — fine-tuned on BigEarthNet.txt (Qwen3-VL + LoRA).

Backends (when SATQUERY_USE_HF=1):
  1. Qwen3-VL-4B-Instruct + optional BigEarthNet LoRA adapter
  2. Fallback: dependency-free heuristic analyzer

Env vars:
  SATQUERY_USE_HF=1|0
  SATQUERY_HF_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
  SATQUERY_LORA_ADAPTER=path/to/final_adapter
  SATQUERY_LOAD_4BIT=1|0
"""

from __future__ import annotations

from typing import Optional, Any
from pathlib import Path
import os
import logging
import gc

from PIL import Image

from src.analysis.single_image import heuristic_caption, heuristic_vqa

logger = logging.getLogger(__name__)

USE_HF = os.getenv("SATQUERY_USE_HF", "0") == "1"
HF_MODEL_ID = os.getenv("SATQUERY_HF_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
LORA_ADAPTER = os.getenv("SATQUERY_LORA_ADAPTER", "").strip()
LOAD_4BIT = os.getenv("SATQUERY_LOAD_4BIT", "1") == "1"


class RSVLMCore:
    """Singleton-style wrapper so the (optional) HF / LoRA model is loaded once."""

    _instance: Optional["RSVLMCore"] = None

    def __init__(self):
        self.backend = "heuristic"
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._model_id = HF_MODEL_ID
        self._adapter_path: Optional[str] = None

        if USE_HF:
            self._try_load_hf()

    @classmethod
    def get(cls) -> "RSVLMCore":
        if cls._instance is None:
            cls._instance = RSVLMCore()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            # free memory if possible
            try:
                del cls._instance._model
                del cls._instance._processor
            except Exception:
                pass
            cls._instance = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _try_load_hf(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig

            has_cuda = torch.cuda.is_available()
            self._device = "cuda" if has_cuda else "cpu"
            logger.info("Loading VLM backend: %s (device=%s)", HF_MODEL_ID, self._device)

            # Clear any previous memory
            gc.collect()
            if has_cuda:
                torch.cuda.empty_cache()

            self._processor = AutoProcessor.from_pretrained(
                HF_MODEL_ID, trust_remote_code=True
            )

            try:
                from transformers import Qwen3VLForConditionalGeneration
                model_cls = Qwen3VLForConditionalGeneration
            except Exception:
                from transformers import AutoModelForVision2Seq
                model_cls = AutoModelForVision2Seq

            # ---------- MEMORY SAFE LOADING (GTX 1650 4GB + 8GB RAM) ----------
            load_kwargs: dict = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,          # critical for 8GB RAM
            }

            if has_cuda and LOAD_4BIT:
                # 4-bit quantization — only realistic way on 4GB VRAM
                try:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,   # GTX 1650 supports FP16 well
                        bnb_4bit_use_double_quant=True,
                    )
                    load_kwargs["device_map"] = "auto"
                    # Hard limit so it doesn't try to use more than ~3.5GB on GPU
                    load_kwargs["max_memory"] = {0: "3500MB", "cpu": "4500MB"}
                    logger.info("Using 4-bit quantization + max_memory limit for 4GB card")
                except Exception as e:
                    logger.warning("4-bit config failed (%s) — falling back to FP16", e)
                    load_kwargs["torch_dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["max_memory"] = {0: "3500MB", "cpu": "4500MB"}
            elif has_cuda:
                # No 4-bit requested but CUDA available
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {0: "3500MB", "cpu": "4500MB"}
            else:
                # Pure CPU — still try float16 to save RAM
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "cpu"
                logger.info("CPU-only mode with float16")

            model = model_cls.from_pretrained(HF_MODEL_ID, **load_kwargs)

            # ---------- LoRA Adapter ----------
            adapter = LORA_ADAPTER
            if not adapter:
                candidates = [
                    Path("weights/bigearthnet_lora"),
                    Path("weights/final_adapter"),
                    Path(__file__).resolve().parents[2] / "weights" / "bigearthnet_lora",
                    Path(__file__).resolve().parents[2] / "weights" / "final_adapter",
                ]
                for c in candidates:
                    if (c / "adapter_config.json").exists() or (
                        c / "adapter_model.safetensors"
                    ).exists():
                        adapter = str(c)
                        break

            if adapter and Path(adapter).exists():
                try:
                    from peft import PeftModel
                    logger.info("Loading BigEarthNet LoRA adapter from %s", adapter)
                    model = PeftModel.from_pretrained(
                        model,
                        adapter,
                        # keep device_map so it doesn't reload everything
                    )
                    self._adapter_path = adapter
                    self.backend = "qwen3vl_lora"
                except Exception as e:
                    logger.warning("LoRA load failed (%s) — base Qwen3-VL only", e)
                    self.backend = "qwen3vl"
            else:
                self.backend = "qwen3vl"
                if adapter:
                    logger.warning("LoRA path set but not found: %s", adapter)

            model.eval()
            self._model = model
            logger.info("VLM ready — backend=%s | device=%s", self.backend, self._device)

            # Final cleanup
            gc.collect()
            if has_cuda:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning("Falling back to heuristic RS-VLM (HF load failed: %s)", e)
            self.backend = "heuristic"
            self._processor = None
            self._model = None
            # free whatever partial load happened
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 96) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move inputs to the same device as model
        try:
            device = next(self._model.parameters()).device
        except StopIteration:
            device = torch.device(self._device)

        inputs = {
            k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()
        }

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        text = self._processor.decode(out[0][prompt_len:], skip_special_tokens=True)
        return text.strip()

    def caption(self, image_path: Path, img_array=None, filename: str = "") -> str:
        if self.backend in ("qwen3vl", "qwen3vl_lora", "huggingface"):
            try:
                image = Image.open(image_path).convert("RGB")
                # Resize large images to save VRAM (important for 4GB)
                max_side = 1024
                if max(image.size) > max_side:
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

                prompt = (
                    "You are a remote-sensing analyst. Describe this satellite or "
                    "aerial image in detail: land cover, land use, notable structures, "
                    "water bodies, vegetation, and overall scene context."
                )
                text = self._generate(image, prompt, max_new_tokens=128)
                if text:
                    tag = (
                        "Qwen3-VL+BigEarthNet-LoRA"
                        if self.backend == "qwen3vl_lora"
                        else "Qwen3-VL"
                    )
                    return text
            except Exception as e:
                logger.warning("VLM caption failed, heuristic fallback: %s", e)

        if img_array is None:
            import numpy as np
            img_array = np.array(Image.open(image_path).convert("RGB"))
        return heuristic_caption(img_array, filename)

    def vqa(self, image_path: Path, query: str, img_array=None, filename: str = "") -> str:
        if self.backend in ("qwen3vl", "qwen3vl_lora", "huggingface"):
            try:
                image = Image.open(image_path).convert("RGB")
                # Resize large images to save VRAM
                max_side = 1024
                if max(image.size) > max_side:
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

                prompt = (
                    "You are a remote-sensing visual question answering system. "
                    f"Answer the following question about this satellite/aerial image.\n"
                    f"Question: {query}\n"
                    "Answer concisely and factually based only on what is visible."
                )
                text = self._generate(image, prompt, max_new_tokens=96)
                if text:
                    tag = (
                        "Qwen3-VL+BigEarthNet-LoRA"
                        if self.backend == "qwen3vl_lora"
                        else "Qwen3-VL"
                    )
                    return text
            except Exception as e:
                logger.warning("VLM VQA failed, heuristic fallback: %s", e)

        if img_array is None:
            import numpy as np
            img_array = np.array(Image.open(image_path).convert("RGB"))
        return heuristic_vqa(query, img_array, filename)

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "model_id": self._model_id if self.backend != "heuristic" else "rs_vlm_heuristic_v2",
            "lora_adapter": self._adapter_path,
            "device": self._device,
        }