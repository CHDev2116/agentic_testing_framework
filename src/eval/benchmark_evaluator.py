def calculate_quality_score(metrics, ai_result, thresholds):
    if not isinstance(metrics, dict):
        return 0.0

    sharpness = float(metrics.get("sharpness", 0.0))
    brightness = float(metrics.get("avg_brightness", 0.0))
    min_sharpness = float(thresholds.get("min_sharpness", 20.0))
    min_brightness = float(thresholds.get("min_brightness", 40.0))
    max_brightness = float(thresholds.get("max_brightness", 220.0))

    sharpness_score = min(1.0, sharpness / max(min_sharpness, 1e-6)) * 60
    center_brightness = (min_brightness + max_brightness) / 2
    half_range = max((max_brightness - min_brightness) / 2, 1e-6)
    exposure_score = max(0.0, 1 - abs(brightness - center_brightness) / half_range) * 40
    total_score = sharpness_score + exposure_score

    if ai_result.get("code") != "SUCCESS_200":
        total_score *= 0.4

    return round(min(100.0, max(0.0, total_score)), 2)


def build_rankings(result_rows, thresholds):
    ranked_rows = []
    for row in result_rows:
        decision = row.get("decision", {})
        ranked_rows.append({
            "file": row.get("file"),
            "score": calculate_quality_score(row.get("metrics"), decision, thresholds),
            "decision": decision.get("decision", "Unknown"),
            "code": decision.get("code", "UNKNOWN"),
            "latency_ms": row.get("latency_ms", 0),
            "status": row.get("status", "UNKNOWN")
        })

    ranked_rows.sort(key=lambda item: (-item["score"], item["latency_ms"], item["file"]))
    for index, item in enumerate(ranked_rows, start=1):
        item["rank"] = index
    return ranked_rows


def get_release_decision(pass_rate, avg_latency, config):
    target_pass_rate = float(config.get("quality_gate", {}).get("target_pass_rate", 90.0))
    timeout_ms = float(config.get("thresholds", {}).get("timeout_ms", 5000))
    latency_limit = timeout_ms * 0.5

    if pass_rate >= target_pass_rate and avg_latency <= latency_limit:
        return "GO", f"Pass rate >= {target_pass_rate}% and latency <= {latency_limit:.0f}ms."
    if pass_rate >= max(target_pass_rate - 15, 70):
        return "REVIEW", "Close to target but requires manual review."
    return "NO_GO", "Pass rate/latency did not satisfy release gate."


def generate_benchmark_insights(profile_outputs, ordered):
    if not ordered:
        return []

    fastest = min(profile_outputs, key=lambda item: item["summary"]["avg_latency_ms"])
    strictest = max(profile_outputs, key=lambda item: item["summary"].get("target_pass_rate", 0))
    best_ranked = ordered[0]

    insights = [
        {
            "trade_off": "Stricter quality gates can reduce release readiness.",
            "observation": (
                f"{strictest['profile']} uses the strictest target pass rate "
                f"({strictest['summary'].get('target_pass_rate', 0)}%)."
            ),
            "decision_implication": (
                "Use strict profiles for certification/benchmark scenarios, "
                "not as default release gates for fast iteration."
            )
        },
        {
            "trade_off": "Lower latency does not guarantee a GO decision.",
            "observation": (
                f"{fastest['profile']} is fastest at "
                f"{fastest['summary']['avg_latency_ms']}ms avg latency."
            ),
            "decision_implication": (
                "Evaluate speed and quality gates together when selecting a deployment profile."
            )
        },
        {
            "trade_off": "A top-ranked profile may still need manual review.",
            "observation": (
                f"Best overall profile is {best_ranked['profile']} "
                f"with decision {best_ranked['summary']['release_decision']}."
            ),
            "decision_implication": (
                "Treat ranking as prioritization input and keep final release decision as policy-based."
            )
        }
    ]
    return insights
