import base64
import json
import os
from importlib import import_module
from typing import Any, Dict, List

from models.contracts import InferenceOutput
from models.llama_quantizer import LlamaQuantizer


def _normalize_result(result: Any, default_msg: str) -> Dict[str, Any]:
    return InferenceOutput.from_payload(result, default_msg=default_msg).to_dict()


_REQUESTS_MODULE = None


def _get_requests():
    global _REQUESTS_MODULE
    if _REQUESTS_MODULE is None:
        _REQUESTS_MODULE = import_module("requests")
    return _REQUESTS_MODULE


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

    def __init__(self, thresholds: Dict[str, Any], inference_cfg: Dict[str, Any]):
        self.thresholds = thresholds
        self.simulated_fallback = SimulatedInferenceEngine(thresholds)
        self.fallback_to_simulated = bool(inference_cfg.get("fallback_to_simulated", True))
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

    def _extract_json_object(self, raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw_text[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}

    def predict_quality(self, photo_path: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        prompt = (
            f"{self.prompt_template}\n"
            f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n"
            f"Thresholds: {json.dumps(self.thresholds, ensure_ascii=False)}"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "images": [self._encode_image(photo_path)],
            "format": "json",
        }

        try:
            response = _get_requests().post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout_s
            )
            response.raise_for_status()
            body = response.json()
            model_text = str(body.get("response", ""))
            parsed = self._extract_json_object(model_text)
            return InferenceOutput.from_payload(
                parsed,
                default_msg="Ollama returned unparsable response.",
                backend=self.backend_name,
            ).to_dict()
        except Exception as exc:
            if self.fallback_to_simulated:
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

    def __init__(self, thresholds: Dict[str, Any], inference_cfg: Dict[str, Any]):
        self.thresholds = thresholds
        self.simulated_fallback = SimulatedInferenceEngine(thresholds)
        self.fallback_to_simulated = bool(inference_cfg.get("fallback_to_simulated", True))
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

    def _extract_json_object(self, raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw_text[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}

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

    def predict_quality(self, photo_path: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        payload = {
            "model": self.model,
            "messages": self._build_messages(photo_path, metrics),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}

        try:
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
            model_text = str(
                body.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            parsed = self._extract_json_object(model_text)
            return InferenceOutput.from_payload(
                parsed,
                default_msg="llama.cpp returned unparsable response.",
                backend=self.backend_name,
            ).to_dict()
        except Exception as exc:
            if self.fallback_to_simulated:
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
    backend = str(inference_cfg.get("backend", "simulated")).lower()

    if backend == "llama_cpp":
        return LlamaCppInferenceEngine(thresholds=thresholds, inference_cfg=inference_cfg)
    if backend == "ollama_vision":
        return OllamaVisionInferenceEngine(thresholds=thresholds, inference_cfg=inference_cfg)
    if backend == "mock_api":
        return MockAPIInferenceEngine(thresholds=thresholds, inference_cfg=inference_cfg)
    return SimulatedInferenceEngine(thresholds=thresholds)
