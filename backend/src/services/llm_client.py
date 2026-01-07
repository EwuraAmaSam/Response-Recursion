from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 1.0
    max_new_tokens: int = 512


class TextGenerationClient:
    """Thin interface for local open-weight generation.

    Implementations:
      - VLLMTextGenerationClient (if vllm installed)
      - TransformersTextGenerationClient (fallback)
    """

    def generate(self, prompt: str, *, cfg: GenerationConfig) -> str:
        raise NotImplementedError


class VLLMTextGenerationClient(TextGenerationClient):
    def __init__(self, model_name_or_path: str, *, tensor_parallel_size: int = 1, dtype: Optional[str] = None) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "vllm is not installed; install vllm or switch RR_INFERENCE_BACKEND=transformers"
            ) from e

        self._SamplingParams = SamplingParams
        self._llm = LLM(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size, dtype=dtype)

    def generate(self, prompt: str, *, cfg: GenerationConfig) -> str:
        params = self._SamplingParams(
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            max_tokens=int(cfg.max_new_tokens),
        )
        out = self._llm.generate([prompt], params)
        if not out:
            return ""
        # vllm returns list[RequestOutput]; take first candidate
        try:
            return out[0].outputs[0].text
        except Exception:
            return ""


class TransformersTextGenerationClient(TextGenerationClient):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device_map: str = "auto",
        torch_dtype: Optional[str] = None,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch
        except Exception as e:
            raise RuntimeError("transformers is required for local generation") from e

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        dtype = None
        if torch_dtype is not None:
            dtype = getattr(torch, torch_dtype)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=dtype,
        )
        self._model.eval()

    def generate(self, prompt: str, *, cfg: GenerationConfig) -> str:
        inputs = self._tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=float(cfg.temperature) > 0.0,
                temperature=float(cfg.temperature),
                top_p=float(cfg.top_p),
                max_new_tokens=int(cfg.max_new_tokens),
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self._tok.decode(gen, skip_special_tokens=True)


def build_textgen_client(
    *,
    backend: str,
    model_name_or_path: str,
    vllm_tensor_parallel_size: int = 1,
    vllm_dtype: Optional[str] = None,
    transformers_torch_dtype: Optional[str] = None,
) -> TextGenerationClient:
    backend = (backend or "transformers").lower()
    if backend == "vllm":
        return VLLMTextGenerationClient(
            model_name_or_path,
            tensor_parallel_size=vllm_tensor_parallel_size,
            dtype=vllm_dtype,
        )
    return TransformersTextGenerationClient(
        model_name_or_path,
        torch_dtype=transformers_torch_dtype,
    )
