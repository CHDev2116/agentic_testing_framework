# Adaptive Backoff (async inference)

Implemented in `src/util/adaptive_backoff.py` and wired into `src/models/async_inference.py`.

## Purpose

When live inference backends return **429/5xx**, async batch runs can retry with exponential backoff + jitter instead of immediately falling back to simulated results.

## Config (`runtime.adaptive_backoff`)

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | Master switch |
| `max_retries` | `2` | Extra attempts after pressure response |
| `base_delay_s` | `0.25` | Initial backoff |
| `max_delay_s` | `8.0` | Cap per sleep |
| `jitter_ratio` | `0.2` | Randomized delay fraction |
| `concurrency_floor` | `1` | Future hook for lowering async concurrency under pressure |

## Enable (example)

```json
"runtime": {
  "adaptive_backoff": {
    "enabled": true,
    "max_retries": 3,
    "base_delay_s": 0.5,
    "max_delay_s": 10.0
  }
}
```

## Notes

- Retries are **HTTP-layer only**; they do not re-run vision metrics.
- Simulated backend is unaffected (no HTTP).
- Pair with `--async-concurrency` tuning on live profiles after measuring 429 rates.
