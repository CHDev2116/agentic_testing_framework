import base64
import json
import logging
import os
from importlib import import_module
from typing import Any, Dict, List

from models.contract_repair import ContractRepairSettings, run_contract_inference_loop
from models.contracts import InferenceOutput
from models.llama_quantizer import LlamaQuantizer

logger = logging.getLogger(__name__)


def _normalize_result(result: Any, default_msg: str) -> Dict[str, Any]:
    return InferenceOutput.from_payload(result, default_msg=default_msg).to_dict()


_REQUESTS_MODULE = None


def _get_requests():
    global _REQUESTS_MODULE
    if _REQUESTS_MODULE is None:
        _REQUESTS_MODULE = import_module("requests")
    return _REQUESTS_MODULE


def apply_post_repair_fallback_policy(
    *,
    result: Dict[str, Any],
    photo_path: str,
    metrics: Dict[str, Any],
    label: str,
    backend_name: str,
    contract: ContractRepairSettings,
    simulated_fallback: Any,
    fallback_to_simulated: bool,
) -> Dict[str, Any]:
    repair_failed = (
        result.get("code") == "ERR_MODEL_RESPONSE_422"
        and "repair_exhausted" in str(result.get("msg", ""))
    )
    if repair_failed and fallback_to_simulated and not contract.strict_contract:
        fallback = simulated_fallback.predict_quality(photo_path, metrics)
        fallback["msg"] = f"{label} fallback to simulated inference after contract repair failure"
        fallback["backend"] = f"{backend_name}->simulated"
        return fallback
    if repair_failed and fallback_to_simulated and contract.strict_contract:
        meta = dict(result.get("contract_meta") or {})
        meta["strict_fallback_blocked"] = True
        tagged = dict(result)
        tagged["contract_meta"] = meta
        return tagged
    return result


class SimulatedInferenceEngine:
    backend_name = "simulated"

    def __init__(self, thresholds: Dict[str, Any]):
        self.quantizer = LlamaQuantizer(thresholds=thresholds)

    def predict_quality(self, photo_path: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        result = self.quantizer.predict_quality(metrics)
        return InferenceOutput.from_payload(
            result,
            default_msg="Simulated inference returned invalid response.",
            backend=self.backend_name,
        ).to_dict()


class OllamaVisionInferenceEngine:
    backend_name = "ollama_vision"

    def __init__(
        self,
        thresholds: Dict[str, Any],
        inference_cfg: Dict[str, Any],
        *,
        replay_mode: str = "off",
    ):
        self.thresholds = thresholds
        self.simulated_fallback = SimulatedInferenceEngine(thresholds)
        self.fallback_to_simulated = bool(inference_cfg.get("fallback_to_simulated", True))
        self._contract = ContractRepairSettings.from_inference_cfg(
            inference_cfg, replay_mode=replay_mode
        )
        ollama_cfg = inference_cfg.get("ollama", {})
        self.host = str(ollama_cfg.get("host", "http://localhost:11434")).rstrip("/")
        self.model = str(ollama_cfg.get("model", "llava:7b"))
        self.timeout_s = float(ollama_cfg.get("timeout_s", 45))
        self.prompt_template = str(
            ollama_cfg.get(
                "prompt_template",
                (
                    "You are a strict image QA assistant. "
                    "Given image quality metrics and the image itself, return only JSON with keys "
                    "decision, code, msg. Valid decisions: Optimal, Blurry, Under-exposed, Over-exposed, Error."
                ),
            )
        )

    def _encode_image(self, photo_path: str) -> str:
        with open(photo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_base_prompt(self, metrics: Dict[str, Any]) -> str:
        return (
            f"{self.prompt_template}\n"
            f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
            f"Thresholds: {json.dumps(self.thresholds, ensure_ascii=False)}"
        )

    def _fetch_ollama_text(self, photo_path: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "images": [self._encode_image(photo_path)],
            "format": "json",
        }
        response = _get_requests().post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout_s
        )
        response.raise_for_status()
        body = response.json()
        return str(body.get("response", ""))

    def _apply_post_repair_fallback_policy(
        self,
        result: Dict[str, Any],
        photo_path: str,
        metrics: Dict[str, Any],
        *,
        label: str,
    ) -> Dict[str, Any]:
        return apply_post_repair_fallback_policy(
            result=result,
            photo_path=photo_path,
            metrics=metrics,
            label=label,
            backend_name=self.backend_name,
            contract=self._contract,
            simulated_fallback=self.simulated_fallback,
            fallback_to_simulated=self.fallback_to_simulated,
        )

    def predict_quality(self, photo_path: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        try:
            result = run_contract_inference_loop(
                self._contract,
                backend=self.backend_name,
                default_msg="Ollama returned unparsable response.",
                build_initial_prompt=lambda: self._build_base_prompt(metrics),
                fetch_model_text=lambda prompt: self._fetch_ollama_text(photo_path, prompt),
            )
            return self._apply_post_repair_fallback_policy(
                result, photo_path, metrics, label="Ollama"
            )
        except Exception as exc:
            logger.warning(
                "Ollama predict_quality failed: backend=%s error=%s",
                self.backend_name,
                exc,
            )
            if self.fallback_to_simulated and not self._contract.strict_contract:
                fallback = self.simulated_fallback.predict_quality(photo_path, metrics)
                fallback["msg"] = f"Ollama fallback to simulated inference: {exc}"
                fallback["backend"] = f"{self.backend_name}->simulated"
                return fallback
            return InferenceOutput(
                decision="Error",
                code="ERR_MODEL_BACKEND_503",
                msg=f"Ollama inference failed: {exc}",
                backend=self.backend_name,
            ).to_dict()


class MockAPIInferenceEngine:
    backend_name = "mock_api"

    def __init__(self, thresholds: Dict[str, Any], inference_cfg: Dict[str, Any]):
        self.thresholds = thresholds
        self.simulated_fallback = SimulatedInferenceEngine(thresholds)
        self.fallback_to_simulated = bool(inference_cfg.get("fallback_to_simulated", True))
        mock_cfg = inference_cfg.get("mock_api", {})
        self.url = str(mock_cfg.get("url", "http://localhost:8080/infer"))
        self.timeout_s = float(mock_cfg.get("timeout_s", 10))
        self.api_key_env = str(mock_cfg.get("api_key_env", "MOCK_INFER_API_KEY"))

    def predict_quality(self, photo_path: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "photo_path": photo_path,
            "metrics": metrics,
            "thresholds": self.thresholds,
        }

        try:
            response = _get_requests().post(
                self.url, json=payload, headers=headers, timeout=self.timeout_s
            )
            response.raise_for_status()
            body = response.json()
            result = body.get("result", body)
            return InferenceOutput.from_payload(
                result,
                default_msg="Mock API returned invalid response.",
                backend=self.backend_name,
            ).to_dict()
        except Exception as exc:
            if self.fallback_to_simulated:
                fallback = self.simulated_fallback.predict_quality(photo_path, metrics)
                fallback["msg"] = f"Mock API fallback to simulated inference: {exc}"
                fallback["backend"] = f"{self.backend_name}->simulated"
                return fallback
            return InferenceOutput(
                decision="Error",
                code="ERR_MODEL_BACKEND_503",
                msg=f"Mock API inference failed: {exc}",
                backend=self.backend_name,
            ).to_dict()


class LlamaCppInferenceEngine:
    backend_name = "llama_cpp"

    def __init__(
        self,
        thresholds: Dict[str, Any],
        inference_cfg: Dict[str, Any],
        *,
        replay_mode: str = "off",
    ):
        self.thresholds = thresholds
        self.simulated_fallback = SimulatedInferenceEngine(thresholds)
        self.fallback_to_simulated = bool(inference_cfg.get("fallback_to_simulated", True))
        self._contract = ContractRepairSettings.from_inference_cfg(
            inference_cfg, replay_mode=replay_mode
        )
        llama_cpp_cfg = inference_cfg.get("llama_cpp", {})
        self.host = str(llama_cpp_cfg.get("host", "http://127.0.0.1:8080")).rstrip("/")
        self.endpoint = str(llama_cpp_cfg.get("endpoint", "/v1/chat/completions"))
        self.model = str(llama_cpp_cfg.get("model", "local-model"))
        self.timeout_s = float(llama_cpp_cfg.get("timeout_s", 45))
        self.temperature = float(llama_cpp_cfg.get("temperature", 0.0))
        self.max_tokens = int(llama_cpp_cfg.get("max_tokens", 256))
        self.use_response_format = bool(llama_cpp_cfg.get("use_response_format", True))
        self.prompt_template = str(
            llama_cpp_cfg.get(
                "prompt_template",
                (
                    "You are a strict image QA assistant. "
                    "Given image quality metrics and thresholds, return only JSON with keys "
                    "decision, code, msg, confidence. Valid decisions: Optimal, Blurry, "
                    "Under-exposed, Over-exposed, Error."
                ),
            )
        )

    def _encode_image_data_uri(self, photo_path: str) -> str:
        image_base64 = ""
        with open(photo_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(photo_path)[1].lower()
        mime_type = "image/jpeg"
        if ext == ".png":
            mime_type = "image/png"
        elif ext == ".webp":
            mime_type = "image/webp"
        return f"data:{mime_type};base64,{image_base64}"

    def _build_messages(self, photo_path: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        text_prompt = (
            f"{self.prompt_template}\n"
            f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
            f"Thresholds: {json.dumps(self.thresholds, ensure_ascii=False)}"
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image_data_uri(photo_path)},
                    },
                ],
            }
        ]

    def _fetch_llama_text(self, photo_path: str, metrics: Dict[str, Any], prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image_data_uri(photo_path)},
                    },
                ],
            }
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}

        response = _get_requests().post(
            f"{self.host}{self.endpoint}", json=payload, timeout=self.timeout_s
        )
        if response.status_code >= 400 and "response_format" in payload:
            payload_without_format = dict(payload)
            payload_without_format.pop("response_format", None)
            response = _get_requests().post(
                f"{self.host}{self.endpoint}",
                json=payload_without_format,
                timeout=self.timeout_s,
            )
        response.raise_for_status()
        body = response.json()
        return str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))

    def _build_base_prompt(self, metrics: Dict[str, Any]) -> str:
        return (
            f"{self.prompt_template}\n"
            f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
            f"Thresholds: {json.dumps(self.thresholds, ensure_ascii=False)}"
        )

    def _apply_post_repair_fallback_policy(
        self,
        result: Dict[str, Any],
        photo_path: str,
        metrics: Dict[str, Any],
        *,
        label: str,
    ) -> Dict[str, Any]:
        return apply_post_repair_fallback_policy(
            result=result,
            photo_path=photo_path,
            metrics=metrics,
            label=label,
            backend_name=self.backend_name,
            contract=self._contract,
            simulated_fallback=self.simulated_fallback,
            fallback_to_simulated=self.fallback_to_simulated,
        )

    def predict_quality(self, photo_path: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        try:
            result = run_contract_inference_loop(
                self._contract,
                backend=self.backend_name,
                default_msg="llama.cpp returned unparsable response.",
                build_initial_prompt=lambda: self._build_base_prompt(metrics),
                fetch_model_text=lambda prompt: self._fetch_llama_text(
                    photo_path, metrics, prompt
                ),
            )
            return self._apply_post_repair_fallback_policy(
                result, photo_path, metrics, label="llama.cpp"
            )
        except Exception as exc:
            logger.warning(
                "llama.cpp predict_quality failed: backend=%s error=%s",
                self.backend_name,
                exc,
            )
            if self.fallback_to_simulated and not self._contract.strict_contract:
                fallback = self.simulated_fallback.predict_quality(photo_path, metrics)
                fallback["msg"] = f"llama.cpp fallback to simulated inference: {exc}"
                fallback["backend"] = f"{self.backend_name}->simulated"
                return fallback
            return InferenceOutput(
                decision="Error",
                code="ERR_MODEL_BACKEND_503",
                msg=f"llama.cpp inference failed: {exc}",
                backend=self.backend_name,
            ).to_dict()


def build_inference_engine(config: Dict[str, Any]):
    thresholds = config.get("thresholds", {})
    inference_cfg = config.get("model_settings", {}).get("inference", {})
    runtime_cfg = config.get("runtime", {})
    replay_mode = str(runtime_cfg.get("replay_mode", "off")).lower()
    backend = str(inference_cfg.get("backend", "simulated")).lower()

    if backend == "llama_cpp":
        return LlamaCppInferenceEngine(
            thresholds=thresholds,
            inference_cfg=inference_cfg,
            replay_mode=replay_mode,
        )
    if backend == "ollama_vision":
        return OllamaVisionInferenceEngine(
            thresholds=thresholds,
            inference_cfg=inference_cfg,
            replay_mode=replay_mode,
        )
    if backend == "mock_api":
        return MockAPIInferenceEngine(thresholds=thresholds, inference_cfg=inference_cfg)
    return SimulatedInferenceEngine(thresholds=thresholds)
