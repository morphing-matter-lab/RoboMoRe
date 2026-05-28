import argparse
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[3]
ANT_DIR = REPO_ROOT / "robomore" / "ant"
RESULTS_DIR = ANT_DIR / "different_llm_experiment" / "results"

if str(ANT_DIR) not in sys.path:
    sys.path.insert(0, str(ANT_DIR))

from prompts import (  # noqa: E402
    morphology_div_prompts,
    morphology_format,
    morphology_prompts,
    rewardfunc_div_prompts,
    rewardfunc_prompts,
    zeroshot_rewardfunc_format,
)
from design import ant_design  # noqa: E402
from utils import Train, Eva, compute_ant_volume  # noqa: E402


DEFAULT_MODELS: Sequence[str] = (
    "gpt-4o-mini",
    "qwen3.7max",
    "gpt-5.5-medium",
    "deepseek-v4",
    "claude-opus-4.7",
)

# Prices are USD per 1M tokens. Fill in missing values for your provider account.
MODEL_PRICING_USD_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # 官方价格页没有 gpt-5.5 这个精确模型名；
    # 如果你这里的 medium 只是 reasoning effort / 内部别名，就按 gpt-5.5 标准价填
    "gpt-5.5-medium": {"input": 5.00, "output": 30.00},
    # deepseek-v4 官方分为 deepseek-v4-flash 和 deepseek-v4-pro
    # 如果你的 provider 的 deepseek-v4 指 V4-Pro，用下面这个：
    "deepseek-v4": {"input": 0.435, "output": 0.87},
    "claude-opus-4.7": {"input": 5.00, "output": 25.00},
    # Qwen3.7-Max (DashScope) pricing in USD per 1M tokens.
    "qwen3.7max": {"input": 2.50, "output": 7.50},
    "qwen3.7-max": {"input": 2.50, "output": 7.50},
}

MODEL_NAME_ALIASES: Dict[str, str] = {
    "deepseek-v4": "deepseek-v4-pro",
    # Some Anthropic-compatible providers expose this model id format.
    "claude-opus-4.7": "claude-opus-4-7",
    "qwen3.7max": "qwen3.7-max",
}


@dataclass
class LLMUsage:
    call_name: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    est_cost_usd: float
    llm_time_sec: float


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.openai_client: Optional[OpenAI] = None
        self.deepseek_client: Optional[OpenAI] = None
        self.dashscope_client: Optional[OpenAI] = None
        self.anthropic_client: Optional[Any] = None
        self.usage_rows: List[LLMUsage] = []
        self._active_run_dir: Optional[Path] = None
        self._active_model: Optional[str] = None
        self._active_model_tag: Optional[str] = None

    @staticmethod
    def _provider_for_model(model: str) -> str:
        m = model.lower()
        if m.startswith("deepseek"):
            return "deepseek"
        if m.startswith("qwen"):
            return "dashscope"
        if m.startswith("claude"):
            return "anthropic"
        return "openai"

    @staticmethod
    def _resolve_model_name(model: str) -> str:
        return MODEL_NAME_ALIASES.get(model, model)

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _safe_parameters(content: str) -> List[float]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\[(.*?)\]", content, re.DOTALL)
            if not match:
                raise
            values = [x.strip() for x in match.group(1).split(",")]
            params = [float(v) for v in values if v]
            if len(params) != 10:
                raise ValueError(f"Invalid morphology parameters: {params}")
            return params
        params = payload.get("parameters", [])
        if not isinstance(params, list) or len(params) != 10:
            raise ValueError(f"Invalid morphology parameters: {params}")
        return [float(x) for x in params]

    @staticmethod
    def _retry_delays(max_retries: int) -> Sequence[float]:
        delays = [1.0, 2.0, 4.0, 8.0]
        return delays[:max_retries]

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
        in_cost = (prompt_tokens / 1_000_000.0) * pricing["input"]
        out_cost = (completion_tokens / 1_000_000.0) * pricing["output"]
        return in_cost + out_cost

    def _record_usage(
        self,
        call_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        llm_time_sec: float,
    ) -> None:
        total_tokens = prompt_tokens + completion_tokens
        self.usage_rows.append(
            LLMUsage(
                call_name=call_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                est_cost_usd=self._estimate_cost(model, prompt_tokens, completion_tokens),
                llm_time_sec=llm_time_sec,
            )
        )
        self._snapshot_active_usage_if_possible(model)

    @staticmethod
    def _usage_field_names() -> List[str]:
        return [
            "call_name",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "est_cost_usd",
            "llm_time_sec",
        ]

    @staticmethod
    def _build_usage_summary_row(model: str, model_usages: List[LLMUsage]) -> Dict[str, Any]:
        total_calls = len(model_usages)
        total_prompt_tokens = int(sum(x.prompt_tokens for x in model_usages))
        total_completion_tokens = int(sum(x.completion_tokens for x in model_usages))
        total_tokens = int(sum(x.total_tokens for x in model_usages))
        total_cost = float(sum(x.est_cost_usd for x in model_usages))
        total_llm_time = float(sum(x.llm_time_sec for x in model_usages))
        denom = float(total_calls) if total_calls > 0 else 1.0
        return {
            "llm_model": model,
            "total_generation_calls": total_calls,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "total_llm_time_sec": total_llm_time,
            "avg_prompt_tokens_per_call": total_prompt_tokens / denom if total_calls > 0 else 0.0,
            "avg_completion_tokens_per_call": total_completion_tokens / denom if total_calls > 0 else 0.0,
            "avg_total_tokens_per_call": total_tokens / denom if total_calls > 0 else 0.0,
            "avg_cost_usd_per_call": total_cost / denom if total_calls > 0 else 0.0,
            "avg_llm_time_sec_per_call": total_llm_time / denom if total_calls > 0 else 0.0,
        }

    def _write_usage_outputs(self, model: str, run_dir: Path, model_tag: str) -> Tuple[Path, Path, Dict[str, Any]]:
        csv_dir = run_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        usage_csv = csv_dir / f"{model_tag}_usage.csv"
        summary_csv = csv_dir / f"{model_tag}_usage_summary.csv"

        model_usages = [x for x in self.usage_rows if x.model == model]
        usage_df = pd.DataFrame([x.__dict__ for x in model_usages], columns=self._usage_field_names())
        usage_df.to_csv(usage_csv, index=False)

        summary_row = self._build_usage_summary_row(model, model_usages)
        pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)
        return usage_csv, summary_csv, summary_row

    def _snapshot_active_usage_if_possible(self, model: str) -> None:
        if (
            self._active_run_dir is None
            or self._active_model_tag is None
            or self._active_model is None
            or model != self._active_model
        ):
            return
        try:
            self._write_usage_outputs(model, self._active_run_dir, self._active_model_tag)
        except Exception as exc:
            logging.warning("Failed to persist incremental usage snapshot: %s", exc)

    def _get_openai_client(self) -> OpenAI:
        if self.openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            if not api_key:
                raise ValueError("OpenAI models need OPENAI_API_KEY (or LLM_API_KEY).")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
        return self.openai_client

    def _get_deepseek_client(self) -> OpenAI:
        if self.deepseek_client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek models need DEEPSEEK_API_KEY.")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            self.deepseek_client = OpenAI(api_key=api_key, base_url=base_url)
        return self.deepseek_client

    def _get_dashscope_client(self) -> OpenAI:
        if self.dashscope_client is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("Qwen models need DASHSCOPE_API_KEY.")
            base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.dashscope_client = OpenAI(api_key=api_key, base_url=base_url)
        return self.dashscope_client

    def _get_anthropic_client(self):
        if self.anthropic_client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("Anthropic_KEY")
            if not api_key:
                raise ValueError("Claude models need ANTHROPIC_API_KEY (or Anthropic_KEY).")
            from anthropic import Anthropic

            base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
            self.anthropic_client = Anthropic(api_key=api_key, base_url=base_url)
        return self.anthropic_client

    def _chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        call_name: str,
        n: int = 1,
        response_format: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        provider = self._provider_for_model(model)
        request_model = self._resolve_model_name(model)
        max_retries = max(0, self.args.llm_retries)

        if provider in {"openai", "deepseek", "dashscope"}:
            if provider == "openai":
                client = self._get_openai_client()
            elif provider == "deepseek":
                client = self._get_deepseek_client()
            else:
                client = self._get_dashscope_client()
            kwargs: Dict[str, Any] = {"model": request_model, "messages": messages, "n": n}
            if response_format is not None:
                kwargs["response_format"] = response_format
            if provider == "dashscope":
                kwargs["extra_body"] = {"enable_thinking": True}

            last_error: Optional[Exception] = None
            for idx, delay in enumerate([0.0] + list(self._retry_delays(max_retries))):
                if delay > 0:
                    time.sleep(delay)
                try:
                    started = time.perf_counter()
                    response = client.chat.completions.create(**kwargs)
                    elapsed = time.perf_counter() - started
                    usage = response.usage
                    self._record_usage(
                        call_name=call_name,
                        model=model,
                        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                        llm_time_sec=float(elapsed),
                    )
                    return [choice.message.content or "" for choice in response.choices]
                except Exception as exc:
                    last_error = exc
                    if idx == 0 and "response_format" in kwargs:
                        kwargs.pop("response_format")
                        continue
            raise RuntimeError(f"LLM request failed for {model} ({provider}/{request_model}): {last_error}")

        client = self._get_anthropic_client()
        system_text = "\n\n".join([m["content"] for m in messages if m.get("role") == "system"])
        anth_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in {"user", "assistant"}]
        outputs: List[str] = []
        last_error: Optional[Exception] = None
        for i in range(n):
            call_name_i = f"{call_name}_{i}" if n > 1 else call_name
            for idx, delay in enumerate([0.0] + list(self._retry_delays(max_retries))):
                if delay > 0:
                    time.sleep(delay)
                try:
                    started = time.perf_counter()
                    response = client.messages.create(
                        model=request_model,
                        system=system_text if system_text else None,
                        messages=anth_messages,
                        max_tokens=self.args.max_tokens,
                    )
                    elapsed = time.perf_counter() - started
                    text_chunks = [blk.text for blk in response.content if getattr(blk, "type", "") == "text"]
                    outputs.append("\n".join(text_chunks).strip())
                    usage = response.usage
                    self._record_usage(
                        call_name=call_name_i,
                        model=model,
                        prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                        completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                        llm_time_sec=float(elapsed),
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    # Retry only for transient errors; auth/permission errors should fail fast.
                    msg = str(exc).lower()
                    if ("forbidden" in msg) or ("permission" in msg) or ("unauthorized" in msg):
                        anth_base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
                        raise RuntimeError(
                            f"Claude request forbidden for model={model} (resolved={request_model}) "
                            f"at {anth_base}. Verify key permissions, organization/region policy, "
                            f"or configure ANTHROPIC_BASE_URL for your provider relay."
                        ) from exc
                    if idx >= max_retries:
                        break
            else:
                pass
        if last_error and len(outputs) < n:
            anth_base = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
            raise RuntimeError(
                f"LLM request failed for {model} (anthropic/{request_model}) via {anth_base}: {last_error}"
            ) from last_error
        return outputs

    @staticmethod
    def _load_reward_function(rewardfunc_path: Path) -> None:
        shutil.copy(rewardfunc_path, ANT_DIR / "GPTrewardfunc.py")
        if "GPTrewardfunc" in sys.modules:
            module = importlib.reload(sys.modules["GPTrewardfunc"])
        else:
            module = importlib.import_module("GPTrewardfunc")
        import GPTAnt

        GPTAnt.GPTAntEnv._get_rew = module._get_rew

    @staticmethod
    def _validate_morphology_xml(xml_path: Path) -> Tuple[bool, str]:
        try:
            ET.parse(xml_path)
        except Exception as exc:
            return False, f"XML parse failed: {type(exc).__name__}: {exc}"

        try:
            from GPTAnt import GPTAntEnv
        except Exception as exc:
            return False, f"Failed to import GPTAntEnv: {type(exc).__name__}: {exc}"

        env = None
        try:
            env = GPTAntEnv(xml_file=str(xml_path), healthy_z_range=(0.1, 2.3))
            env.reset(seed=0)
            return True, ""
        except Exception as exc:
            return False, f"Environment reset failed: {type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    def _validate_reward_function(self, rewardfunc_path: Path) -> Tuple[bool, str]:
        try:
            code = rewardfunc_path.read_text()
            compile(code, str(rewardfunc_path), "exec")
        except Exception as exc:
            return False, f"Python compile failed: {type(exc).__name__}: {exc}"

        try:
            self._load_reward_function(rewardfunc_path)
        except Exception as exc:
            return False, f"Reward load failed: {type(exc).__name__}: {exc}"

        base_xml = ANT_DIR / "GPTAnt.xml"
        if not base_xml.exists():
            return True, "Skipped runtime check because GPTAnt.xml does not exist."

        try:
            from GPTAnt import GPTAntEnv
        except Exception as exc:
            return False, f"Failed to import GPTAntEnv: {type(exc).__name__}: {exc}"

        env = None
        try:
            env = GPTAntEnv(xml_file=str(base_xml), healthy_z_range=(0.1, 2.3))
            env.reset(seed=0)
            env.step(env.action_space.sample())
            return True, ""
        except Exception as exc:
            return False, f"Runtime step failed: {type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    @staticmethod
    def _extract_index_from_name(path: Path, prefix: str) -> int:
        match = re.search(rf"{re.escape(prefix)}_(\d+)\.", path.name)
        return int(match.group(1)) if match else -1

    @staticmethod
    def _params_from_ant_xml(xml_path: Path) -> List[float]:
        root = ET.parse(xml_path).getroot()
        torso = root.find(".//geom[@name='torso_geom']")
        aux = root.find(".//geom[@name='aux_1_geom']")
        leg = root.find(".//geom[@name='left_leg_geom']")
        ankle = root.find(".//geom[@name='left_ankle_geom']")
        if torso is None or aux is None or leg is None or ankle is None:
            raise ValueError(f"Cannot parse required geoms from xml: {xml_path}")

        def _float_attr(node: Any, key: str) -> float:
            val = node.get(key)
            if val is None:
                raise ValueError(f"Missing attribute {key} in {xml_path}")
            return float(val.strip().split()[0])

        def _fromto_xy(node: Any) -> Tuple[float, float]:
            val = node.get("fromto")
            if val is None:
                raise ValueError(f"Missing fromto in {xml_path}")
            nums = [float(x) for x in val.strip().split()]
            if len(nums) != 6:
                raise ValueError(f"Invalid fromto in {xml_path}: {val}")
            return nums[3], nums[4]

        p1 = _float_attr(torso, "size")
        p2, p3 = _fromto_xy(aux)
        p4, p5 = _fromto_xy(leg)
        p6, p7 = _fromto_xy(ankle)
        p8 = _float_attr(aux, "size")
        p9 = _float_attr(leg, "size")
        p10 = _float_attr(ankle, "size")
        return [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]

    def _load_existing_generation(
        self, source_run_dir: Path, rewardfunc_nums: int, morphology_nums: int
    ) -> Tuple[List[Path], List[Path], List[List[float]], List[float]]:
        env_dir = source_run_dir / "env"
        assets_dir = source_run_dir / "assets"
        if not env_dir.exists() or not assets_dir.exists():
            raise FileNotFoundError(
                f"Source run dir missing env/assets: {source_run_dir}. "
                f"Expected {env_dir} and {assets_dir}."
            )

        rewardfunc_paths = sorted(
            env_dir.glob("GPTrewardfunc_*.py"),
            key=lambda p: self._extract_index_from_name(p, "GPTrewardfunc"),
        )
        morphology_paths = sorted(
            assets_dir.glob("GPTAnt_*.xml"),
            key=lambda p: self._extract_index_from_name(p, "GPTAnt"),
        )
        if len(rewardfunc_paths) < rewardfunc_nums or len(morphology_paths) < morphology_nums:
            raise ValueError(
                f"Not enough generated files in {source_run_dir}: "
                f"rewardfunc={len(rewardfunc_paths)}/{rewardfunc_nums}, "
                f"morphology={len(morphology_paths)}/{morphology_nums}"
            )

        rewardfunc_paths = rewardfunc_paths[:rewardfunc_nums]
        morphology_paths = morphology_paths[:morphology_nums]
        parameter_list = [self._params_from_ant_xml(p) for p in morphology_paths]
        material_list = [compute_ant_volume(params) for params in parameter_list]
        return rewardfunc_paths, morphology_paths, parameter_list, material_list

    def _train(
        self,
        morphology_idx: int,
        rewardfunc_idx: int,
        run_dir: Path,
        *,
        stage: str,
        total_timesteps: float,
        fine_iter: Optional[int] = None,
    ) -> Path:
        stage_dir = run_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        return Path(
            Train(
                morphology_idx,
                rewardfunc_idx,
                str(run_dir),
                total_timesteps=float(total_timesteps),
                stage=stage,
                iter=int(fine_iter or 0),
            )
        )

    @staticmethod
    def _checkpoint_base_path(
        run_dir: Path,
        stage: str,
        morphology_idx: int,
        rewardfunc_idx: int,
        total_timesteps: float,
        fine_iter: Optional[int] = None,
    ) -> Path:
        if stage == "fine" and fine_iter:
            return run_dir / stage / f"SAC_iter{int(fine_iter)}_morphology{morphology_idx}_rewardfunc{rewardfunc_idx}_{float(total_timesteps)}steps"
        return run_dir / stage / f"SAC_morphology{morphology_idx}_rewardfunc{rewardfunc_idx}_{float(total_timesteps)}steps"

    @classmethod
    def _checkpoint_base_path_candidates(
        cls,
        *,
        run_dir: Path,
        stage: str,
        morphology_idx: int,
        rewardfunc_idx: int,
        total_timesteps: float,
        fine_iter: Optional[int] = None,
    ) -> List[Path]:
        primary = cls._checkpoint_base_path(
            run_dir=run_dir,
            stage=stage,
            morphology_idx=morphology_idx,
            rewardfunc_idx=rewardfunc_idx,
            total_timesteps=total_timesteps,
            fine_iter=fine_iter,
        )
        if stage == "fine":
            legacy = cls._checkpoint_base_path(
                run_dir=run_dir,
                stage=stage,
                morphology_idx=morphology_idx,
                rewardfunc_idx=rewardfunc_idx,
                total_timesteps=total_timesteps,
                fine_iter=None,
            )
            if legacy != primary:
                return [primary, legacy]
        return [primary]

    @staticmethod
    def _checkpoint_exists(checkpoint_base_path: Path) -> bool:
        return checkpoint_base_path.exists() or checkpoint_base_path.with_suffix(".zip").exists()

    @staticmethod
    def _copy_checkpoint_if_exists(source_base_path: Path, target_base_path: Path) -> bool:
        source_zip_path = source_base_path.with_suffix(".zip")
        target_zip_path = target_base_path.with_suffix(".zip")
        if source_zip_path.exists():
            target_zip_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_zip_path, target_zip_path)
            return True
        if source_base_path.exists():
            target_base_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_base_path, target_base_path)
            return True
        return False

    def _train_or_reuse_checkpoint(
        self,
        *,
        run_dir: Path,
        stage: str,
        morphology_idx: int,
        rewardfunc_idx: int,
        total_timesteps: float,
        reuse_training_from: Optional[Path],
        fine_iter: Optional[int] = None,
    ) -> Tuple[Path, bool]:
        target_base_path = self._checkpoint_base_path(
            run_dir=run_dir,
            stage=stage,
            morphology_idx=morphology_idx,
            rewardfunc_idx=rewardfunc_idx,
            total_timesteps=total_timesteps,
            fine_iter=fine_iter,
        )
        target_candidates = self._checkpoint_base_path_candidates(
            run_dir=run_dir,
            stage=stage,
            morphology_idx=morphology_idx,
            rewardfunc_idx=rewardfunc_idx,
            total_timesteps=total_timesteps,
            fine_iter=fine_iter,
        )
        if not self.args.force_retrain:
            for existing_target in target_candidates:
                if self._checkpoint_exists(existing_target):
                    logging.info(
                        "Skip training (checkpoint exists) stage=%s r=%s m=%s path=%s",
                        stage,
                        rewardfunc_idx,
                        morphology_idx,
                        existing_target,
                    )
                    return existing_target, True

        if not self.args.force_retrain and reuse_training_from is not None:
            source_candidates = self._checkpoint_base_path_candidates(
                run_dir=reuse_training_from,
                stage=stage,
                morphology_idx=morphology_idx,
                rewardfunc_idx=rewardfunc_idx,
                total_timesteps=total_timesteps,
                fine_iter=fine_iter,
            )
            for source_base_path in source_candidates:
                copied = self._copy_checkpoint_if_exists(source_base_path, target_base_path)
                if copied:
                    logging.info(
                        "Skip training (reuse checkpoint) stage=%s r=%s m=%s source=%s target=%s",
                        stage,
                        rewardfunc_idx,
                        morphology_idx,
                        source_base_path,
                        target_base_path,
                    )
                    return target_base_path, True

        model_path = self._train(
            morphology_idx,
            rewardfunc_idx,
            run_dir,
            stage=stage,
            total_timesteps=total_timesteps,
            fine_iter=fine_iter,
        )
        return model_path, False

    @staticmethod
    def _write_metric_matrix(rows: List[Dict[str, Any]], value_key: str, output_csv: Path) -> None:
        if not rows:
            pd.DataFrame(columns=["rewardfunc_idx"]).to_csv(output_csv, index=False)
            return
        df = pd.DataFrame(rows)
        matrix_df = (
            df.pivot_table(
                index="rewardfunc_idx",
                columns="morphology_idx",
                values=value_key,
                aggfunc="first",
            )
            .sort_index()
            .sort_index(axis=1)
        )
        matrix_df.index.name = "rewardfunc_idx"
        matrix_df.columns.name = "morphology_idx"
        matrix_df.to_csv(output_csv)

    def _write_stage_matrices(
        self,
        *,
        model_tag: str,
        run_dir: Path,
        coarse_rows: List[Dict[str, Any]],
        fine_rows: List[Dict[str, Any]],
    ) -> None:
        csv_dir = run_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        self._write_metric_matrix(
            coarse_rows,
            "efficiency",
            csv_dir / f"{model_tag}_coarse_efficiency_matrix.csv",
        )
        self._write_metric_matrix(
            coarse_rows,
            "fitness",
            csv_dir / f"{model_tag}_coarse_fitness_matrix.csv",
        )
        self._write_metric_matrix(
            fine_rows,
            "efficiency",
            csv_dir / f"{model_tag}_fine_efficiency_matrix.csv",
        )
        self._write_metric_matrix(
            fine_rows,
            "fitness",
            csv_dir / f"{model_tag}_fine_fitness_matrix.csv",
        )

    def _evaluate(self, model_path: Path) -> Tuple[float, float]:
        return Eva(str(model_path), run_steps=self.args.eval_runs)

    @staticmethod
    def _best_row_by_efficiency(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not rows:
            return None
        return max(rows, key=lambda x: float(x.get("efficiency", float("-inf"))))

    def _export_best_sample(
        self,
        *,
        stage_name: str,
        model_tag: str,
        run_dir: Path,
        row: Dict[str, Any],
        export_root: Path,
    ) -> None:
        stage_dir = export_root / f"{model_tag}_{stage_name}_best"
        stage_dir.mkdir(parents=True, exist_ok=True)

        reward_src = Path(str(row["rewardfunc_path"]))
        morph_src = Path(str(row["morphology_path"]))
        reward_dst = stage_dir / reward_src.name
        morph_dst = stage_dir / morph_src.name
        shutil.copy2(reward_src, reward_dst)
        shutil.copy2(morph_src, morph_dst)

        model_key = "coarse_model_path" if stage_name == "coarse" else "fine_model_path"
        model_base = Path(str(row[model_key]))
        model_dst_base = stage_dir / model_base.name
        self._copy_checkpoint_if_exists(model_base, model_dst_base)

        summary = {
            "llm_model": row.get("llm_model"),
            "stage": stage_name,
            "rewardfunc_idx": int(row["rewardfunc_idx"]),
            "morphology_idx": int(row["morphology_idx"]),
            "fitness": float(row["fitness"]),
            "efficiency": float(row["efficiency"]),
            "material_cost": float(row["material_cost"]),
            "rewardfunc_path": str(reward_dst),
            "morphology_path": str(morph_dst),
            "model_path": str(model_dst_base),
        }
        if "coarse_efficiency" in row:
            summary["coarse_efficiency"] = float(row["coarse_efficiency"])

        (stage_dir / "best_summary.json").write_text(json.dumps(summary, indent=2))

        if self.args.export_best_video:
            videos_dir = stage_dir / f"videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib, shutil, sys\n"
                        f"sys.path.insert(0, {str(ANT_DIR)!r})\n"
                        f"ant_dir = {str(ANT_DIR)!r}\n"
                        f"morph = {str(morph_src)!r}\n"
                        f"reward = {str(reward_src)!r}\n"
                        f"model = {str(model_base)!r}\n"
                        f"video_dir = {str(videos_dir)!r}\n"
                        f"eval_runs = {int(self.args.eval_runs)}\n"
                        f"reward_idx = {int(row['rewardfunc_idx'])}\n"
                        f"morph_idx = {int(row['morphology_idx'])}\n"
                        "from pathlib import Path\n"
                        "from utils import Eva\n"
                        "ant_path = Path(ant_dir)\n"
                        "shutil.copy2(morph, ant_path / 'GPTAnt.xml')\n"
                        "shutil.copy2(reward, ant_path / 'GPTrewardfunc.py')\n"
                        "import GPTrewardfunc\n"
                        "importlib.reload(GPTrewardfunc)\n"
                        "Eva(model, run_steps=eval_runs, folder_name=video_dir, video=True, rewardfunc_index=reward_idx, morphology_index=morph_idx)\n"
                    ),
                ],
                cwd=str(ANT_DIR),
                text=True,
                capture_output=True,
            )
            if video_proc.returncode == 0:
                summary["video_dir"] = str(videos_dir)
                summary["video_export_status"] = "ok"
            else:
                summary["video_export_status"] = "failed"
                summary["video_export_returncode"] = int(video_proc.returncode)
                summary["video_export_stderr_tail"] = "\n".join((video_proc.stderr or "").splitlines()[-12:])
                logging.warning(
                    "Video export failed for stage=%s model=%s r=%s m=%s returncode=%s",
                    stage_name,
                    model_tag,
                    int(row["rewardfunc_idx"]),
                    int(row["morphology_idx"]),
                    video_proc.returncode,
                )
            (stage_dir / "best_summary.json").write_text(json.dumps(summary, indent=2))

    def _export_best_samples(
        self,
        *,
        model_tag: str,
        run_dir: Path,
        coarse_rows: List[Dict[str, Any]],
        fine_rows: List[Dict[str, Any]],
    ) -> None:
        export_dir_raw = (self.args.best_output_dir or "").strip()
        if not export_dir_raw:
            return
        export_root = Path(export_dir_raw).expanduser().resolve()
        export_root.mkdir(parents=True, exist_ok=True)

        best_coarse = self._best_row_by_efficiency(coarse_rows)
        if best_coarse is not None:
            self._export_best_sample(
                stage_name="coarse",
                model_tag=model_tag,
                run_dir=run_dir,
                row=best_coarse,
                export_root=export_root,
            )
            logging.info(
                "Exported best coarse sample model=%s r=%s m=%s efficiency=%.5f",
                model_tag,
                int(best_coarse["rewardfunc_idx"]),
                int(best_coarse["morphology_idx"]),
                float(best_coarse["efficiency"]),
            )

        best_fine = self._best_row_by_efficiency(fine_rows)
        if best_fine is not None:
            self._export_best_sample(
                stage_name="fine",
                model_tag=model_tag,
                run_dir=run_dir,
                row=best_fine,
                export_root=export_root,
            )
            logging.info(
                "Exported best fine sample model=%s r=%s m=%s efficiency=%.5f",
                model_tag,
                int(best_fine["rewardfunc_idx"]),
                int(best_fine["morphology_idx"]),
                float(best_fine["efficiency"]),
            )
        else:
            logging.warning("No accepted fine rows to export for model=%s", model_tag)

    def _load_existing_coarse_rows(
        self,
        *,
        model: str,
        model_tag: str,
        run_dir: Path,
        rewardfunc_paths: List[Path],
        morphology_paths: List[Path],
        parameter_list: List[List[float]],
        material_list: List[float],
    ) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        expected = len(rewardfunc_paths) * len(morphology_paths)
        if expected <= 0:
            return None

        csv_dir = run_dir / "csv"
        result_csv = csv_dir / f"{model_tag}_results.csv"

        # First choice: recover full coarse rows from existing results.csv.
        if result_csv.exists():
            try:
                df = pd.read_csv(result_csv)
                if {"stage", "rewardfunc_idx", "morphology_idx"}.issubset(set(df.columns)):
                    df = df[df["stage"] == "coarse"].copy()
                    if "llm_model" in df.columns and not df.empty:
                        df_model = df[df["llm_model"] == model]
                        if not df_model.empty:
                            df = df_model
                    if not df.empty:
                        df["rewardfunc_idx"] = df["rewardfunc_idx"].astype(int)
                        df["morphology_idx"] = df["morphology_idx"].astype(int)
                        df = df.sort_values(["rewardfunc_idx", "morphology_idx"]).drop_duplicates(
                            subset=["rewardfunc_idx", "morphology_idx"], keep="last"
                        )
                        if len(df) >= expected:
                            rows: List[Dict[str, Any]] = []
                            candidates: List[Dict[str, Any]] = []
                            for r_idx, reward_path in enumerate(rewardfunc_paths):
                                for m_idx, morph_path in enumerate(morphology_paths):
                                    hit = df[(df["rewardfunc_idx"] == r_idx) & (df["morphology_idx"] == m_idx)]
                                    if hit.empty:
                                        raise ValueError(f"Missing coarse row r={r_idx}, m={m_idx} in {result_csv}")
                                    rec = hit.iloc[0].to_dict()
                                    fitness = float(rec.get("fitness", 0.0))
                                    avg_reward = float(rec.get("avg_reward", 0.0))
                                    efficiency = float(rec.get("efficiency", 0.0))
                                    material = float(material_list[m_idx])
                                    coarse_model_path = str(
                                        rec.get(
                                            "coarse_model_path",
                                            self._checkpoint_base_path(
                                                run_dir=run_dir,
                                                stage="coarse",
                                                morphology_idx=m_idx,
                                                rewardfunc_idx=r_idx,
                                                total_timesteps=float(self.args.total_timesteps),
                                            ),
                                        )
                                    )
                                    row = {
                                        "llm_model": model,
                                        "stage": "coarse",
                                        "rewardfunc_idx": r_idx,
                                        "morphology_idx": m_idx,
                                        "rewardfunc_path": str(reward_path),
                                        "morphology_path": str(morph_path),
                                        "parameters": json.dumps(parameter_list[m_idx]),
                                        "material_cost": material,
                                        "fitness": fitness,
                                        "avg_reward": avg_reward,
                                        "efficiency": efficiency,
                                        "llm_total_tokens_so_far": int(rec.get("llm_total_tokens_so_far", 0) or 0),
                                        "llm_total_cost_usd_so_far": float(rec.get("llm_total_cost_usd_so_far", 0.0) or 0.0),
                                        "coarse_model_path": coarse_model_path,
                                        "fine_model_path": "",
                                        "coarse_fitness": fitness,
                                        "coarse_avg_reward": avg_reward,
                                        "coarse_efficiency": efficiency,
                                        "selected_from_coarse_rank": "",
                                    }
                                    rows.append(row)
                                    candidates.append(
                                        {
                                            "rewardfunc_idx": r_idx,
                                            "morphology_idx": m_idx,
                                            "rewardfunc_path": str(reward_path),
                                            "morphology_path": str(morph_path),
                                            "material_cost": material,
                                            "fitness": fitness,
                                            "avg_reward": avg_reward,
                                            "efficiency": efficiency,
                                            "coarse_model_path": coarse_model_path,
                                        }
                                    )
                            logging.info("Loaded coarse rows from existing results csv: %s", result_csv)
                            return rows, candidates
            except Exception as exc:
                logging.warning("Failed to load coarse rows from %s: %s", result_csv, exc)

        # Fallback: reconstruct coarse rows from coarse matrices only.
        coarse_eff_csv = csv_dir / f"{model_tag}_coarse_efficiency_matrix.csv"
        coarse_fit_csv = csv_dir / f"{model_tag}_coarse_fitness_matrix.csv"
        if not coarse_eff_csv.exists() or not coarse_fit_csv.exists():
            return None
        try:
            eff_df = pd.read_csv(coarse_eff_csv)
            fit_df = pd.read_csv(coarse_fit_csv)
            if "rewardfunc_idx" not in eff_df.columns or "rewardfunc_idx" not in fit_df.columns:
                return None
            eff_df = eff_df.set_index("rewardfunc_idx")
            fit_df = fit_df.set_index("rewardfunc_idx")
            rows: List[Dict[str, Any]] = []
            candidates: List[Dict[str, Any]] = []
            token_sum = int(sum(x.total_tokens for x in self.usage_rows if x.model == model))
            cost_sum = float(sum(x.est_cost_usd for x in self.usage_rows if x.model == model))
            for r_idx, reward_path in enumerate(rewardfunc_paths):
                if r_idx not in eff_df.index or r_idx not in fit_df.index:
                    raise ValueError(f"Missing rewardfunc row {r_idx} in coarse matrices")
                for m_idx, morph_path in enumerate(morphology_paths):
                    col = str(m_idx)
                    if col not in eff_df.columns or col not in fit_df.columns:
                        raise ValueError(f"Missing morphology column {m_idx} in coarse matrices")
                    eff_val = eff_df.loc[r_idx, col]
                    fit_val = fit_df.loc[r_idx, col]
                    if pd.isna(eff_val) or pd.isna(fit_val):
                        raise ValueError(f"NaN in coarse matrices at r={r_idx}, m={m_idx}")
                    efficiency = float(eff_val)
                    fitness = float(fit_val)
                    material = float(material_list[m_idx])
                    avg_reward = 0.0
                    coarse_model_path = str(
                        self._checkpoint_base_path(
                            run_dir=run_dir,
                            stage="coarse",
                            morphology_idx=m_idx,
                            rewardfunc_idx=r_idx,
                            total_timesteps=float(self.args.total_timesteps),
                        )
                    )
                    row = {
                        "llm_model": model,
                        "stage": "coarse",
                        "rewardfunc_idx": r_idx,
                        "morphology_idx": m_idx,
                        "rewardfunc_path": str(reward_path),
                        "morphology_path": str(morph_path),
                        "parameters": json.dumps(parameter_list[m_idx]),
                        "material_cost": material,
                        "fitness": fitness,
                        "avg_reward": avg_reward,
                        "efficiency": efficiency,
                        "llm_total_tokens_so_far": token_sum,
                        "llm_total_cost_usd_so_far": cost_sum,
                        "coarse_model_path": coarse_model_path,
                        "fine_model_path": "",
                        "coarse_fitness": fitness,
                        "coarse_avg_reward": avg_reward,
                        "coarse_efficiency": efficiency,
                        "selected_from_coarse_rank": "",
                    }
                    rows.append(row)
                    candidates.append(
                        {
                            "rewardfunc_idx": r_idx,
                            "morphology_idx": m_idx,
                            "rewardfunc_path": str(reward_path),
                            "morphology_path": str(morph_path),
                            "material_cost": material,
                            "fitness": fitness,
                            "avg_reward": avg_reward,
                            "efficiency": efficiency,
                            "coarse_model_path": coarse_model_path,
                        }
                    )
            logging.info(
                "Loaded coarse rows from existing coarse matrices: %s and %s",
                coarse_eff_csv,
                coarse_fit_csv,
            )
            return rows, candidates
        except Exception as exc:
            logging.warning("Failed to reconstruct coarse rows from matrices: %s", exc)
            return None

    def generate_rewardfunc_div(self, model: str, rewardfunc_nums: int, run_dir: Path) -> List[Path]:
        messages = [
            {"role": "system", "content": "You are a reinforcement learning reward function designer"},
            {"role": "user", "content": rewardfunc_prompts + zeroshot_rewardfunc_format},
        ]
        rewardfunc_paths: List[Path] = []
        env_dir = run_dir / "env"
        env_dir.mkdir(parents=True, exist_ok=True)
        existing_rewardfunc_paths = sorted(
            env_dir.glob("GPTrewardfunc_*.py"),
            key=lambda p: self._extract_index_from_name(p, "GPTrewardfunc"),
        )
        for path in existing_rewardfunc_paths[:rewardfunc_nums]:
            rewardfunc_paths.append(path)
            try:
                code_text = path.read_text()
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Accepted reward function #{len(rewardfunc_paths) - 1}:\n```python\n{code_text}\n```",
                    }
                )
            except Exception as exc:
                logging.warning("Failed to read existing reward function %s: %s", path, exc)
        if rewardfunc_paths:
            logging.info(
                "Loaded existing reward functions in %s: %s/%s",
                run_dir,
                len(rewardfunc_paths),
                rewardfunc_nums,
            )
        while len(rewardfunc_paths) < rewardfunc_nums:
            i = len(rewardfunc_paths)
            call_name = "reward_seed" if i == 0 else f"reward_div_{i}"
            p = env_dir / f"GPTrewardfunc_{i}.py"
            attempt = 1
            last_error: Optional[str] = None
            while True:
                request_messages: List[Dict[str, str]] = list(messages)
                if i > 0:
                    request_messages.append({"role": "user", "content": rewardfunc_div_prompts + zeroshot_rewardfunc_format})
                    request_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Reflection requirement: compare against all previously accepted reward functions "
                                "in this conversation, avoid repeating their design patterns, and generate a more novel design."
                            ),
                        }
                    )
                if last_error:
                    request_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Previous attempt failed executable validation with error:\n{last_error}\n"
                                "Reflect on the failure, fix it, and provide a corrected but still novel reward function."
                            ),
                        }
                    )
                content = self._chat(model=model, messages=request_messages, call_name=f"{call_name}_try_{attempt}")
                code = self._extract_code(content[0] if content else "")
                if not code:
                    last_error = "No Python code block found in LLM response."
                    logging.warning(
                        "Reward generation invalid for model=%s idx=%s attempt=%s: %s",
                        model,
                        i,
                        attempt,
                        "No Python code block found in LLM response.",
                    )
                    attempt += 1
                    continue

                p.write_text("import numpy as np\n" + code + "\n")
                ok, reason = self._validate_reward_function(p)
                if ok:
                    rewardfunc_paths.append(p)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Accepted reward function #{i}:\n```python\n{code}\n```",
                        }
                    )
                    logging.info("Reward validated for model=%s idx=%s attempt=%s", model, i, attempt)
                    break

                logging.warning(
                    "Reward validation failed for model=%s idx=%s attempt=%s: %s",
                    model,
                    i,
                    attempt,
                    reason,
                )
                last_error = reason
                attempt += 1
        return rewardfunc_paths

    def generate_morphology_div(
        self, model: str, morphology_nums: int, run_dir: Path
    ) -> Tuple[List[Path], List[List[float]], List[float]]:
        messages = [
            {"role": "system", "content": "You are a helpful mujoco robot designer"},
            {"role": "user", "content": morphology_prompts + morphology_format},
        ]
        morphology_paths: List[Path] = []
        parameter_list: List[List[float]] = []
        material_list: List[float] = []
        assets_dir = run_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        existing_morphology_paths = sorted(
            assets_dir.glob("GPTAnt_*.xml"),
            key=lambda p: self._extract_index_from_name(p, "GPTAnt"),
        )
        for path in existing_morphology_paths[:morphology_nums]:
            try:
                params = self._params_from_ant_xml(path)
                material = compute_ant_volume(params)
                morphology_paths.append(path)
                parameter_list.append(params)
                material_list.append(material)
                messages.append({"role": "assistant", "content": json.dumps({"parameters": params})})
            except Exception as exc:
                logging.warning("Skip invalid existing morphology %s: %s", path, exc)
        if morphology_paths:
            logging.info(
                "Loaded existing morphologies in %s: %s/%s",
                run_dir,
                len(morphology_paths),
                morphology_nums,
            )
        while len(morphology_paths) < morphology_nums:
            i = len(morphology_paths)
            call_name = "morph_seed" if i == 0 else f"morph_div_{i}"
            p = assets_dir / f"GPTAnt_{i}.xml"
            attempt = 1
            last_error: Optional[str] = None
            while True:
                request_messages: List[Dict[str, str]] = list(messages)
                if i > 0:
                    request_messages.append({"role": "user", "content": morphology_div_prompts + morphology_format})
                    request_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Reflection requirement: review all previously accepted morphologies in this conversation, "
                                "avoid repeating similar parameter patterns, and produce a more novel yet valid design."
                            ),
                        }
                    )
                if last_error:
                    request_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Previous attempt failed executable validation with error:\n{last_error}\n"
                                "Reflect on the failure, fix it, and return corrected morphology parameters."
                            ),
                        }
                    )
                content = self._chat(
                    model=model,
                    messages=request_messages,
                    call_name=f"{call_name}_try_{attempt}",
                    response_format={"type": "json_object"},
                )
                try:
                    params = self._safe_parameters(content[0] if content else "{}")
                    p.write_text(ant_design(params))
                except Exception as exc:
                    last_error = f"Parameter/XML generation failed: {type(exc).__name__}: {exc}"
                    logging.warning(
                        "Morphology generation invalid for model=%s idx=%s attempt=%s: %s",
                        model,
                        i,
                        attempt,
                        f"Parameter/XML generation failed: {type(exc).__name__}: {exc}",
                    )
                    attempt += 1
                    continue

                ok, reason = self._validate_morphology_xml(p)
                if ok:
                    parameter_list.append(params)
                    material_list.append(compute_ant_volume(params))
                    messages.append({"role": "assistant", "content": json.dumps({"parameters": params})})
                    morphology_paths.append(p)
                    logging.info("Morphology validated for model=%s idx=%s attempt=%s", model, i, attempt)
                    break

                logging.warning(
                    "Morphology validation failed for model=%s idx=%s attempt=%s: %s",
                    model,
                    i,
                    attempt,
                    reason,
                )
                last_error = reason
                attempt += 1

        return morphology_paths, parameter_list, material_list

    def run_one_model(self, model: str) -> Tuple[Path, Path, Path]:
        model_tag = model.replace("/", "_").replace(".", "_").replace("-", "_")
        run_dir_raw = (self.args.run_dir or "").strip()
        if run_dir_raw:
            run_dir = Path(run_dir_raw).expanduser().resolve()
        else:
            run_dir = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Running model: %s in %s", model, run_dir)
        self._active_run_dir = run_dir
        self._active_model = model
        self._active_model_tag = model_tag

        try:
            reuse_dir_raw = (self.args.reuse_generation_from or "").strip()
            reuse_training_raw = (self.args.reuse_training_from or "").strip()
            reuse_training_dir: Optional[Path] = Path(reuse_training_raw).expanduser().resolve() if reuse_training_raw else None

            source_run_dir: Optional[Path] = None
            if reuse_dir_raw:
                source_run_dir = Path(reuse_dir_raw).expanduser().resolve()
            elif run_dir_raw:
                # Resume-friendly default: if --run-dir points to an existing generated run,
                # reuse generated artifacts instead of asking LLM to regenerate.
                maybe_source_run_dir = run_dir
                env_dir = maybe_source_run_dir / "env"
                assets_dir = maybe_source_run_dir / "assets"
                if env_dir.exists() or assets_dir.exists():
                    source_run_dir = maybe_source_run_dir
                    logging.info(
                        "Auto-reusing generated artifacts from --run-dir %s",
                        source_run_dir,
                    )
                else:
                    logging.info(
                        "No env/assets found in --run-dir %s, will generate new artifacts.",
                        maybe_source_run_dir,
                    )

            if source_run_dir is not None and source_run_dir != run_dir:
                logging.info("Reusing generated artifacts from %s", source_run_dir)
                rewardfunc_paths, morphology_paths, parameter_list, material_list = self._load_existing_generation(
                    source_run_dir, self.args.rewardfunc_nums, self.args.morphology_nums
                )
                if reuse_training_dir is None:
                    reuse_training_dir = source_run_dir
            else:
                if source_run_dir is not None:
                    logging.info(
                        "Resuming generation in-place from %s (will keep existing artifacts and only fill missing ones)",
                        run_dir,
                    )
                rewardfunc_paths = self.generate_rewardfunc_div(model, self.args.rewardfunc_nums, run_dir)
                morphology_paths, parameter_list, material_list = self.generate_morphology_div(
                    model, self.args.morphology_nums, run_dir
                )
                if reuse_training_dir is None and source_run_dir is not None:
                    reuse_training_dir = source_run_dir

            usage_csv, summary_csv, summary_row = self._write_usage_outputs(model, run_dir, model_tag)
            logging.info(
                "Saved generation usage snapshot model=%s calls=%s tokens=%s cost=%.6f",
                model,
                summary_row["total_generation_calls"],
                summary_row["total_tokens"],
                summary_row["total_cost_usd"],
            )

            coarse_rows: List[Dict[str, Any]] = []
            coarse_candidates: List[Dict[str, Any]] = []

            logging.info("================================ enter coarse stage ================================")
            cached_coarse = self._load_existing_coarse_rows(
                model=model,
                model_tag=model_tag,
                run_dir=run_dir,
                rewardfunc_paths=rewardfunc_paths,
                morphology_paths=morphology_paths,
                parameter_list=parameter_list,
                material_list=material_list,
            )
            if cached_coarse is not None and not self.args.force_retrain:
                coarse_rows, coarse_candidates = cached_coarse
                logging.info("Skip coarse evaluation and reuse cached coarse rows (count=%s)", len(coarse_rows))
            else:
                for r_idx, reward_path in enumerate(rewardfunc_paths):
                    for m_idx, morph_path in enumerate(morphology_paths):
                        shutil.copy(morph_path, ANT_DIR / "GPTAnt.xml")
                        self._load_reward_function(reward_path)
                        model_path, reused_coarse_ckpt = self._train_or_reuse_checkpoint(
                            run_dir=run_dir,
                            stage="coarse",
                            morphology_idx=m_idx,
                            rewardfunc_idx=r_idx,
                            total_timesteps=float(self.args.total_timesteps),
                            reuse_training_from=reuse_training_dir,
                        )
                        fitness, avg_reward = self._evaluate(model_path)
                        material = material_list[m_idx]
                        efficiency = fitness / material if material > 0 else 0.0
                        token_sum = int(sum(x.total_tokens for x in self.usage_rows if x.model == model))
                        cost_sum = float(sum(x.est_cost_usd for x in self.usage_rows if x.model == model))
                        row = {
                            "llm_model": model,
                            "stage": "coarse",
                            "rewardfunc_idx": r_idx,
                            "morphology_idx": m_idx,
                            "rewardfunc_path": str(reward_path),
                            "morphology_path": str(morph_path),
                            "parameters": json.dumps(parameter_list[m_idx]),
                            "material_cost": material,
                            "fitness": fitness,
                            "avg_reward": avg_reward,
                            "efficiency": efficiency,
                            "llm_total_tokens_so_far": token_sum,
                            "llm_total_cost_usd_so_far": cost_sum,
                            "coarse_model_path": str(model_path),
                            "fine_model_path": "",
                            "coarse_fitness": fitness,
                            "coarse_avg_reward": avg_reward,
                            "coarse_efficiency": efficiency,
                            "selected_from_coarse_rank": "",
                        }
                        coarse_rows.append(row)
                        coarse_candidates.append(
                            {
                                "rewardfunc_idx": r_idx,
                                "morphology_idx": m_idx,
                                "rewardfunc_path": str(reward_path),
                                "morphology_path": str(morph_path),
                                "material_cost": material,
                                "fitness": fitness,
                                "avg_reward": avg_reward,
                                "efficiency": efficiency,
                                "coarse_model_path": str(model_path),
                            }
                        )
                        logging.info(
                            "[coarse] model=%s r=%s m=%s fitness=%.5f efficiency=%.5f tokens=%s cost=%.6f reused_ckpt=%s",
                            model,
                            r_idx,
                            m_idx,
                            fitness,
                            efficiency,
                            token_sum,
                            cost_sum,
                            reused_coarse_ckpt,
                        )
            logging.info("================================ end coarse stage ================================")

            sorted_candidates = sorted(coarse_candidates, key=lambda x: x["efficiency"], reverse=True)
            if self.args.fine_top_k > 0:
                top_k = min(self.args.fine_top_k, len(sorted_candidates))
            else:
                ratio = max(0.0, float(self.args.fine_top_ratio))
                top_k = max(1, int(len(sorted_candidates) * ratio)) if sorted_candidates else 0
            selected_candidates = sorted_candidates[:top_k]
            logging.info(
                "Fine stage selection: total=%s selected=%s top_ratio=%.4f top_k_override=%s",
                len(sorted_candidates),
                len(selected_candidates),
                float(self.args.fine_top_ratio),
                self.args.fine_top_k,
            )

            fine_rows: List[Dict[str, Any]] = []
            rejected_fine_rows = 0
            logging.info("================================ enter fine stage ================================")
            for coarse_rank, candidate in enumerate(selected_candidates, start=1):
                r_idx = int(candidate["rewardfunc_idx"])
                m_idx = int(candidate["morphology_idx"])
                reward_path = Path(str(candidate["rewardfunc_path"]))
                morph_path = Path(str(candidate["morphology_path"]))

                shutil.copy(morph_path, ANT_DIR / "GPTAnt.xml")
                self._load_reward_function(reward_path)
                fine_model_path, reused_fine_ckpt = self._train_or_reuse_checkpoint(
                    run_dir=run_dir,
                    stage="fine",
                    morphology_idx=m_idx,
                    rewardfunc_idx=r_idx,
                    total_timesteps=float(self.args.fine_timesteps),
                    reuse_training_from=reuse_training_dir,
                    fine_iter=coarse_rank,
                )
                fitness, avg_reward = self._evaluate(fine_model_path)
                material = material_list[m_idx]
                efficiency = fitness / material if material > 0 else 0.0
                coarse_efficiency = float(candidate["efficiency"])
                token_sum = int(sum(x.total_tokens for x in self.usage_rows if x.model == model))
                cost_sum = float(sum(x.est_cost_usd for x in self.usage_rows if x.model == model))
                if efficiency <= coarse_efficiency:
                    rejected_fine_rows += 1
                    logging.warning(
                        "[fine] reject model=%s rank=%s r=%s m=%s coarse_eff=%.5f fine_eff=%.5f (must improve)",
                        model,
                        coarse_rank,
                        r_idx,
                        m_idx,
                        coarse_efficiency,
                        efficiency,
                    )
                    continue

                fine_rows.append(
                    {
                        "llm_model": model,
                        "stage": "fine",
                        "rewardfunc_idx": r_idx,
                        "morphology_idx": m_idx,
                        "rewardfunc_path": str(reward_path),
                        "morphology_path": str(morph_path),
                        "parameters": json.dumps(parameter_list[m_idx]),
                        "material_cost": material,
                        "fitness": fitness,
                        "avg_reward": avg_reward,
                        "efficiency": efficiency,
                        "llm_total_tokens_so_far": token_sum,
                        "llm_total_cost_usd_so_far": cost_sum,
                        "coarse_model_path": str(candidate["coarse_model_path"]),
                        "fine_model_path": str(fine_model_path),
                        "coarse_fitness": float(candidate["fitness"]),
                        "coarse_avg_reward": float(candidate["avg_reward"]),
                        "coarse_efficiency": coarse_efficiency,
                        "selected_from_coarse_rank": coarse_rank,
                    }
                )
                logging.info(
                    "[fine] model=%s rank=%s r=%s m=%s coarse_eff=%.5f fine_eff=%.5f reused_ckpt=%s",
                    model,
                    coarse_rank,
                    r_idx,
                    m_idx,
                    coarse_efficiency,
                    efficiency,
                    reused_fine_ckpt,
                )
            logging.info(
                "Fine stage accepted=%s rejected=%s (strictly require fine_eff > coarse_eff)",
                len(fine_rows),
                rejected_fine_rows,
            )
            logging.info("================================ end fine stage ================================")

            rows: List[Dict[str, Any]] = coarse_rows + fine_rows

            csv_dir = run_dir / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            result_csv = csv_dir / f"{model_tag}_results.csv"
            pd.DataFrame(rows).to_csv(result_csv, index=False)
            self._write_stage_matrices(model_tag=model_tag, run_dir=run_dir, coarse_rows=coarse_rows, fine_rows=fine_rows)
            self._export_best_samples(model_tag=model_tag, run_dir=run_dir, coarse_rows=coarse_rows, fine_rows=fine_rows)
            usage_csv, summary_csv, summary_row = self._write_usage_outputs(model, run_dir, model_tag)
            logging.info(
                "LLM usage summary model=%s calls=%s avg_tokens=%.2f avg_cost=%.6f avg_time=%.3fs",
                model,
                summary_row["total_generation_calls"],
                summary_row["avg_total_tokens_per_call"],
                summary_row["avg_cost_usd_per_call"],
                summary_row["avg_llm_time_sec_per_call"],
            )
            return result_csv, usage_csv, summary_csv
        finally:
            self._active_run_dir = None
            self._active_model = None
            self._active_model_tag = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ant experiment across multiple LLM models.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--morphology-nums", type=int, default=5)
    parser.add_argument("--rewardfunc-nums", type=int, default=3)
    parser.add_argument("--total-timesteps", type=float, default=5e5, help="Coarse stage timesteps.")
    parser.add_argument("--fine-timesteps", type=float, default=5e5, help="Fine stage timesteps.")
    parser.add_argument(
        "--fine-top-ratio",
        type=float,
        default=0.05,
        help="Select top ratio from coarse stage by efficiency when --fine-top-k <= 0.",
    )
    parser.add_argument(
        "--fine-top-k",
        type=int,
        default=-1,
        help="If >0, override --fine-top-ratio and select exactly top-k coarse candidates.",
    )
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument(
        "--reuse-generation-from",
        type=str,
        default="",
        help=(
            "Path to an existing run dir containing env/GPTrewardfunc_*.py and assets/GPTAnt_*.xml. "
            "If provided, skip LLM generation and reuse these artifacts."
        ),
    )
    parser.add_argument(
        "--reuse-training-from",
        type=str,
        default="",
        help=(
            "Path to an existing run dir containing coarse/fine SAC checkpoints. "
            "If checkpoint for a pair exists, skip training and run evaluation directly."
        ),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing checkpoints and always retrain.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help=(
            "Output run directory. If set, write outputs/checkpoints into this directory "
            "instead of creating a new timestamped folder."
        ),
    )
    parser.add_argument(
        "--best-output-dir",
        type=str,
        default="",
        help=(
            "If set, export best coarse/fine samples into this directory: "
            "copy morphology/reward/model and write a summary json."
        ),
    )
    parser.add_argument(
        "--export-best-video",
        action="store_true",
        help="When used with --best-output-dir, export MuJoCo videos for best coarse/fine samples.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.run_dir or "").strip() and len(args.models) != 1:
        raise ValueError("--run-dir currently supports exactly one model per run to avoid output collisions.")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(args)
    for model in args.models:
        result_csv, usage_csv, summary_csv = runner.run_one_model(model)
        print(f"[DONE] {model} -> result: {result_csv}")
        print(f"[DONE] {model} -> usage:  {usage_csv}")
        print(f"[DONE] {model} -> summary:{summary_csv}")


if __name__ == "__main__":
    main()
