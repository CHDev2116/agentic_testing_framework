“The prompt didn’t change.
The model didn’t change.
But suddenly, the pipeline broke.”

That was the moment I realized traditional testing assumptions don’t work well with probabilistic AI systems.

Recently, I’ve been experimenting with a self-healing AI vision testing framework for Visual QA workflows.

Traditional approaches usually fall into two extremes:

• Manual inspection → too slow and expensive
• Pixel-level comparison → too brittle for real-world variability

A slight lighting change can trigger a completely false failure.

But with Generative AI systems, what we actually care about is semantic quality:

• Does the image look natural?
• Is the primary subject recognizable?
• Is the output usable from a human perspective?

This is where Vision Language Models (VLMs) become interesting — and also where a new challenge appears:

Inference instability.

AI outputs are probabilistic, not deterministic.
Static assertions alone are no longer enough.

So instead of treating evaluation as simple Pass/Fail logic, the framework introduces a bounded self-healing loop.

When the system detects a NO_GO decision, it can:

Diagnose probable causes
under-exposure
over-exposure
blur / sharpness degradation
Apply targeted remediation
brightness adjustment
dimming
sharpening
Re-run inference through:

Engine → Model → Eval

The important part is that recovery is guardrail-bounded:

• retry limits
• gain thresholds
• oscillation checks
• bounded remediation policies

This prevents the system from turning into “retry until green.”

In practice, the workflow behaves less like a static test script and more like iterative QA.

For example, when an image is flagged as too dark, the framework can automatically brighten the image, re-run inference, and evaluate whether the result improves — all within a constrained retry budget.

Another challenge quickly appeared during development:

LLM outputs are not guaranteed to be valid JSON.

Even unchanged prompts can suddenly produce:

• malformed JSON
• markdown wrappers
• unexpected prose
• partially invalid structured output

To improve resilience, I added a lightweight recovery layer that performs:

• best-effort JSON extraction
• schema normalization
• graceful fallback handling

If parsing still fails, remote inference can fall back to a deterministic simulated engine so the batch pipeline continues running.

The goal is not “perfect AI behavior.”

The goal is operational resilience.

On the infrastructure side, the project also experiments with local inference using:

• GGUF / Q4-style quantized models
• Ollama
• llama.cpp
• deterministic CI-style simulation backends

This significantly reduces latency and removes most marginal inference cost for large batch runs while keeping sensitive datasets local.

Architecturally, the system is intentionally separated into:

• Engine → deterministic image metrics
• Model → backend abstraction layer
• Evaluation → GO / REVIEW / NO_GO arbitration

That separation makes backend swapping a configuration problem instead of a rewrite problem.

One thing became very clear while building this:

Traditional QA frameworks were designed around deterministic assumptions.

AI systems break those assumptions.

We are gradually moving from:

Boolean Testing

toward:

Probabilistic Evaluation.

This project is my experiment in building resilient AI testing systems — systems capable not only of detecting failures, but also of diagnosing instability and attempting bounded recovery.

We are no longer just validating correctness.

We are engineering reliability for probabilistic software systems.

#AI #LLM #GenAI #Testing #QA #MachineLearning #Ollama #LlamaCpp #AIEngineering #SoftwareEngineering