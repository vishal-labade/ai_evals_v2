# CHANGE LOG --- AI Evals

This changelog documents the evolution from V1 (single-turn factual
evaluation) to V2 (multi-turn behavioral reliability + context-boundary
memory evaluation).

This version aligns with: - README.md - RUNBOOK_v2.md -
DATA_CONTRACT_v2.md - SCORING_v2.md - REPRODUCIBILITY_v2.md

------------------------------------------------------------------------

## v1_final

Scope: Single-turn factual accuracy and hallucination detection.

Highlights: - Frozen V1 prompt suite. - Deterministic scoring
pipeline. - Suite expansion (paraphrase + numeric perturbations). -
Canonical numeric scoring for exact tasks (NOT_IN_CONTEXT strict). - NIC
metrics (global + conditional + explicit k/n). - Wilson confidence
interval leaderboard output. - Stable leaderboard artifact.

Outcome: V1 established reproducible offline evaluation for factual
reliability.

------------------------------------------------------------------------

## v2 --- Behavioral Reliability (Multi-Turn)

Motivation: V1 measured factual correctness. It did not evaluate
multi-turn stability, persona adherence, instruction persistence, or
conversational consistency.

Key Additions:

### Scenario Architecture

-   Introduced structured multi-turn scenarios.
-   Added deterministic check system:
    -   persona
    -   memory
    -   format
    -   recovery
    -   calibration
    -   consistency

### Suite Expansion

-   Base scenarios defined in `v2_scenarios.jsonl`.
-   Expanded via generator to `v2_scenarios_expanded.jsonl`.
-   Frozen benchmark created: `v2_suite_50.jsonl`.

### Behavioral Metrics

-   Persona Stability Score (PSS)
-   Behavioral memory score (short-horizon recall)
-   Format score
-   Recovery score
-   Consistency / contradiction rate
-   Calibration error
-   Behavioral Reliability Index (BRI)

Observed Baseline (frozen suite): - PSS ≈ 0.33--0.43 - Memory score ≈
1.0 (short-horizon recall) - Consistency ≈ 0.75--0.91 - BRI ≈ 0.55--0.61

Outcome: V2 introduced structured, deterministic multi-turn behavioral
evaluation.

------------------------------------------------------------------------

## v2.1 --- Experiment Families

Motivation: Separate model scale, context length, and tradeoff effects
using controlled experiments.

Added: - `experiments.yaml` - Experiment family abstraction

Experiment Families:

1.  Behavioral baseline
    -   Fixed operating point (e.g., num_ctx=2048)
    -   Cross-model comparison
2.  Model size (fixed context)
    -   Compare 3B vs 14B vs 20B class models
    -   Isolate parameter scale effects
3.  Tradeoff (size vs context)
    -   Medium model + long context vs larger model + short context
4.  MoE vs dense comparison
    -   Compare mixture-of-experts vs dense models at fixed context

Outcome: Enabled causal comparison across model classes and operating
regimes.

------------------------------------------------------------------------

## v2.2 --- Context Cliff Track (MCS)

Motivation: Behavioral memory ≠ context-window memory mechanics.

Short-horizon recall was near-perfect across models. To isolate context
window effects, a dedicated stress test was introduced.

Added: - `v2_ctx_cliff_suite_20.jsonl` - Memory Compliance Score (MCS) -
drop_turn metric - MCS AUC metric

Context Cliff Observations (example): - num_ctx=1024 → mcs ≈ 0.33 -
num_ctx=2048 → mcs ≈ 0.33 - num_ctx≥4096 → mcs = 1.0

Conclusion: Memory degradation is strongly driven by context window
size, not parameter scale.

Outcome: Clear empirical boundary between reasoning quality and memory
horizon.

------------------------------------------------------------------------

## v2.3 --- Enrichment + Plotting Layer

Motivation: Raw metrics are insufficient without experiment context.

Added: - Metadata enrichment layer - Structured grouping by: - family -
profile - model - num_ctx - temperature - Plot generation: - MCS vs
num_ctx - drop_turn vs num_ctx - MCS AUC vs num_ctx - BRI comparison
across models

Outcome: Produced publication-ready, reproducible artifacts.

------------------------------------------------------------------------

## v2_final (Current State)

Architecture:

Behavioral Track: - Frozen 50-scenario benchmark - Deterministic
scoring - BRI aggregation - Cross-model reliability comparison

Context Track: - Frozen memory stress suite - MCS, drop_turn, AUC -
Context-boundary identification

Reproducibility: - Frozen benchmarks - Deterministic scoring - Explicit
configuration logging - Separation of experimental variables

V2 establishes a structured, academically defensible framework for: -
Multi-turn behavioral reliability - Context window memory mechanics -
Controlled model comparisons

------------------------------------------------------------------------

End of change log.