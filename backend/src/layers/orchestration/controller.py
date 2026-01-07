from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import uuid
import datetime as dt

from core.config import AppConfig, build_embedding_layer_config
from core.storage import RiskSurfaceStore, PromptCandidateStore

from layers.prompt_seed.library import load_seed_library, ensure_seed_library_file
from layers.prompt_seed.selector import SeedSelector

from layers.breakdown.pipeline import BreakdownPipeline
from layers.embedding_layer import process_breakdown_message, TextEmbedder

from layers.risk_surface.model import RiskSurfaceModel
from layers.risk_surface.updater import update_risk_surface

from layers.prompt_gen.candidate_generator import generate_candidates
from layers.evaluator.selector import select_prompt

def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

class MockLLMClient:
    def __init__(self, model_name: str = "mock-llm") -> None:
        self.model_name = model_name

    def complete(self, prompt_text: str) -> Dict[str, Any]:
        lower = prompt_text.lower()
        if "ignore previous instructions" in lower or "system prompt" in lower:
            return {
                "response_text": "I can’t help with that. I can explain, at a high level, how system prompts steer models safely.",
                "refusal_prob": 0.9,
            }
        return {
            "response_text": f"Here is a helpful, safe response to: {prompt_text}",
            "refusal_prob": 0.05,
        }

@dataclass
class RunState:
    run_id: str
    created_at: str
    iteration: int = 0
    last_prompt: Optional[Dict[str, Any]] = None
    status: str = "created"
    error: Optional[str] = None

class RunController:
    def __init__(self, app_cfg: AppConfig) -> None:
        self.app_cfg = app_cfg
        self.embedding_cfg = build_embedding_layer_config(app_cfg)
        self.embedder = TextEmbedder(self.embedding_cfg.embedding_config)
        self.breakdown = BreakdownPipeline()

        ensure_seed_library_file(app_cfg.artifacts.seed_library_path)
        seed_lib = load_seed_library(app_cfg.artifacts.seed_library_path)
        self.seed_selector = SeedSelector(seed_lib, strategy="random")

        self.risk_store = RiskSurfaceStore(app_cfg.artifacts.parquet_base_dir)
        self.prompt_store = PromptCandidateStore(app_cfg.artifacts.parquet_base_dir)
        self.risk_model = RiskSurfaceModel()

        self.llm = MockLLMClient()
        self._runs: Dict[str, RunState] = {}

        # Target model for refusal_consistency N=8 sampling (optional)
        self._target_client = None
        if app_cfg.components.target_model:
            from services.llm_client import build_textgen_client
            self._target_client = build_textgen_client(
                backend=app_cfg.components.inference_backend,
                model_name_or_path=app_cfg.components.target_model,
                vllm_tensor_parallel_size=app_cfg.components.vllm_tensor_parallel_size,
                vllm_dtype=app_cfg.components.vllm_dtype,
                transformers_torch_dtype=app_cfg.components.transformers_torch_dtype,
            )

    def _sample_n8_for_consistency(self, prompt_text: str) -> Optional[List[str]]:
        """Generate N=8 samples for refusal_consistency if target model configured."""
        if self._target_client is None:
            return None
        from services.llm_client import GenerationConfig
        cfg = GenerationConfig(temperature=0.8, top_p=0.95, max_new_tokens=512)
        samples = []
        for _ in range(8):
            try:
                samples.append(self._target_client.generate(prompt_text, cfg=cfg))
            except Exception:
                pass
        return samples if len(samples) >= 2 else None

    def start_run(self, *, description: Optional[str] = None) -> RunState:
        run_id = _id("run")
        seed = self.seed_selector.select()
        st = RunState(run_id=run_id, created_at=dt.datetime.utcnow().isoformat() + "Z", status="running")
        st.last_prompt = {"prompt_id": _id("prompt"), "prompt_text": seed.prompt_text, "seed_id": seed.seed_id}
        self._runs[run_id] = st
        return st

    def get(self, run_id: str) -> RunState:
        if run_id not in self._runs:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self._runs[run_id]

    def step(self, run_id: str) -> RunState:
        st = self.get(run_id)
        try:
            st.status = "running"
            prompt = st.last_prompt or {"prompt_id": _id("prompt"), "prompt_text": self.seed_selector.select().prompt_text}
            prompt_id = prompt["prompt_id"]
            prompt_text = prompt["prompt_text"]
            response_id = _id("resp")

            llm_out = self.llm.complete(prompt_text)
            response_text = llm_out["response_text"]
            refusal_prob = float(llm_out.get("refusal_prob", 0.0))

            breakdown_out = self.breakdown.run({
                "t": st.iteration,
                "run_id": st.run_id,
                "prompt_id": prompt_id,
                "response_id": response_id,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "prior_core_statement_ids": None,
            })

            breakdown_env = {
                "version": self.app_cfg.version,
                "run_id": st.run_id,
                "iteration": st.iteration,
                "from_layer": "iterative_breakdown",
                "to_layer": "embedding_system",
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
                "payload": {
                    "t": st.iteration,
                    "run_id": st.run_id,
                    "prompt_id": prompt_id,
                    "response_id": response_id,
                    "turn_index": st.iteration,
                    "embedding_units": breakdown_out.get("embedding_units", []),
                    "statement_types": [s.get("type") for s in breakdown_out.get("statements", [])],
                    "context_tags": {"attack_level": (breakdown_out.get("attack_signal") or {}).get("level", "none")},
                },
            }

            # Generate N=8 samples for refusal_consistency if target model configured
            consistency_samples = self._sample_n8_for_consistency(prompt_text)

            metrics_env = process_breakdown_message(
                breakdown_env,
                config=self.embedding_cfg,
                embedder=self.embedder,
                prompt_text=prompt_text,
                response_text=response_text,
                refusal_consistency_samples=consistency_samples,
            )

            risk_env = update_risk_surface(
                metrics_env,
                parquet_base_dir=str(self.app_cfg.artifacts.parquet_base_dir),
                store=self.risk_store,
                model=self.risk_model,
            )

            cand_env = generate_candidates(
                run_id=st.run_id,
                iteration=st.iteration,
                seed_prompt=prompt_text,
                breakdown_output=breakdown_out,
                risk_surface_envelope=risk_env,
                store=self.prompt_store,
            )

            sel_env = select_prompt(cand_env)
            st.last_prompt = sel_env["payload"]["selected_prompt"]

            st.iteration += 1
            st.error = None
            return st
        except Exception as e:
            st.status = "failed"
            st.error = str(e)
            return st

    def run_loop(self, run_id: str, max_iterations: int) -> RunState:
        st = self.get(run_id)
        for _ in range(max_iterations):
            if st.status == "failed":
                break
            self.step(run_id)
        if st.status != "failed":
            st.status = "completed"
        return st