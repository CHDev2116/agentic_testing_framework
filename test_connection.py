import logging
import time

import requests

logger = logging.getLogger(__name__)


def test_llama_health_check(url="http://localhost:8080/v1", model="llama-3.1-8b"):
    endpoint = f"{url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }

    try:
        start = time.perf_counter()
        res = requests.post(endpoint, json=payload, timeout=30)
        res.raise_for_status()

        latency = time.perf_counter() - start
        res.json()

        logger.info("[%s] Connected.", model)
        logger.info("TTFT (Approx): %.4fs", latency)

    except requests.exceptions.RequestException as e:
        logger.error("Connection Failed: %s", e)
    except KeyError:
        logger.error("Malformed Response: %s", res.text)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _src = Path(__file__).resolve().parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from util.cli_logging import configure_cli_logging  # noqa: E402

    configure_cli_logging()
    test_llama_health_check()
