# SCORING --- AI Evals v2 (Behavioral + Context Memory)

This document defines all metrics used in AI Evals v2, including:

1.  Behavioral Reliability metrics
2.  Behavioral Reliability Index (BRI)
3.  Memory Compliance Score (MCS) for context cliff experiments
4.  Aggregation and interpretation framework

All metrics are deterministic and rule-based. No LLM-as-judge or manual
annotation is used.

------------------------------------------------------------------------

# 1. Behavioral Metrics (Scenario-Level)

Behavioral metrics are computed per (run_id, scenario_id) pair.

A scenario may define one or more checks under:

-   persona
-   memory
-   format
-   recovery
-   calibration
-   consistency

Only defined checks are scored. Metrics not applicable to a scenario are
excluded from aggregation.

------------------------------------------------------------------------

## 1.1 Persona Stability Score (PSS)

Definition:

Let T_p be the set of turns in a scenario that require persona
compliance.

For each required turn t: - Evaluate deterministic persona rules (style
markers, required phrases, refusal patterns, structural constraints,
etc.). - Define persona_ok\[t\] ∈ {0,1}.

Then:

PSS = (Σ persona_ok\[t\]) / \|T_p\|

Additional diagnostic:

-   persona_drift_turn = smallest t such that persona_ok\[t\] = 0 (if
    any).

Interpretation:

-   PSS = 1 → persona perfectly maintained.
-   PSS ≈ 0.33--0.43 (observed baseline) → persona breaks are common.
-   Drift turn identifies where degradation begins.

Observed baseline (frozen suite): \~0.33--0.43 across tested models.

------------------------------------------------------------------------

## 1.2 Behavioral Memory Score

Definition:

Short-horizon recall accuracy within normal dialogue length.

For each required recall turn t:

-   Define memory_ok\[t\] ∈ {0,1} via deterministic require_any /
    forbid_any rules.

Then:

memory_score = (Σ memory_ok\[t\]) / \|T_m\|

Interpretation:

-   Evaluates structured recall under typical context length.
-   Does NOT stress context window.
-   Near-1.0 baseline is expected when recall occurs within window
    capacity.

Observed baseline: \~1.0 across tested models.

------------------------------------------------------------------------

## 1.3 Format Score

Definition:

For turns requiring structured output:

format_ok\[t\] ∈ {0,1} based on deterministic structure checks (JSON
keys, bullet counts, etc.)

format_score = (Σ format_ok\[t\]) / \|T_f\|

Interpretation:

-   Measures instruction persistence in formatting.
-   Detects structural drift across turns.

------------------------------------------------------------------------

## 1.4 Recovery Score

Definition:

If a scenario includes explicit recovery testing:

recovery_ok ∈ {0,1} depending on: - acknowledgment of error - correction
of prior mistake

recovery_score = mean(recovery_ok)

Interpretation:

-   Measures self-correction capability after feedback.
-   Tests meta-reasoning behavior.

------------------------------------------------------------------------

## 1.5 Consistency / Contradiction Rate

Definition:

Across turns, detect contradictions using deterministic
polarity/semantic rules.

contradiction_rate = (# contradictions detected) / (# evaluated turns)

Consistency score may be defined as:

consistency_score = 1 − contradiction_rate

Observed baseline consistency: \~0.75--0.91 depending on model.

Interpretation:

-   Higher is better.
-   Low consistency indicates internal instability across turns.

------------------------------------------------------------------------

## 1.6 Calibration

Definition:

For scenarios requiring stated confidence:

For each evaluation instance i:

abs_error_i = \| stated_confidence_i − correctness_i \|

avg_abs_error = mean(abs_error_i)

Interpretation:

-   Lower AAE → better calibration.
-   Measures alignment between confidence and correctness.

------------------------------------------------------------------------

# 2. Behavioral Reliability Index (BRI)

Definition:

For a scenario s, let M_s be the set of defined metrics for that
scenario.

BRI_s = mean( metric_i for i ∈ M_s )

Run-level BRI:

BRI_run = mean( BRI_s across scenarios )

Important:

-   Only metrics applicable to a scenario are included.
-   No penalty for undefined components.

Observed baseline BRI: \~0.55--0.61 across tested models.

Interpretation:

-   Composite reliability score.
-   Allows comparison across models at fixed operating points.

------------------------------------------------------------------------

# 3. Memory Compliance Score (MCS) --- Context Cliff Track

MCS isolates context-window-driven memory degradation.

Defined under:

checks.memory.turns\[\]

Each turn rule may include:

-   require_any: list of strings that must appear
-   forbid_any: list of strings that must not appear

------------------------------------------------------------------------

## 3.1 Turn-Level MCS (mcs_turn)

For each required turn t:

Initialize ok = True

If require_any exists: ok &= assistant_text contains any(require_any)

If forbid_any exists: ok &= assistant_text contains none(forbid_any)

Then:

mcs_turn\[t\] = 1 if ok else 0

Turns without memory rules do not emit mcs_turn.

------------------------------------------------------------------------

## 3.2 Run-Level MCS

Let T be all evaluated required memory turns across all scenarios.

mcs = (Σ mcs_turn\[t\]) / \|T\|

Interpretation:

-   mcs = 1 → perfect compliance.
-   Lower mcs → degraded recall.

------------------------------------------------------------------------

## 3.3 drop_turn

For each scenario:

drop_turn = smallest t where mcs_turn\[t\] = 0

If no failure occurs: drop_turn = null

Run-level:

drop_turn_mean = mean(drop_turn across scenarios)

Interpretation:

-   Smaller drop_turn → earlier collapse.
-   Indicates effective memory horizon.

------------------------------------------------------------------------

## 3.4 MCS AUC

Define prefix_ok\[t\] = 1 if all required checks up to t have passed,
else 0.

Scenario AUC:

auc_s = mean(prefix_ok\[t\] for t in 0..T−1)

Run-level:

mcs_auc = mean(auc_s across scenarios)

Interpretation:

-   AUC ≈ 1 → stable memory across dialogue.
-   Lower AUC → early degradation.
-   Captures memory persistence, not just final correctness.

------------------------------------------------------------------------

# 4. Empirical Context Cliff Example

Observed (Qwen 14B):

num_ctx = 1024 → mcs ≈ 0.33, drop_turn_mean ≈ 7, mcs_auc ≈ 0.7 num_ctx =
2048 → mcs ≈ 0.33, drop_turn_mean ≈ 7, mcs_auc ≈ 0.7 num_ctx = 4096 →
mcs = 1.0 num_ctx ≥ 6144 → mcs = 1.0

Interpretation:

-   Clear reliability boundary between 2K and 4K tokens.
-   Memory degradation driven by context window size, not reasoning
    ability.
-   Demonstrates causal separation between scale and memory horizon.

------------------------------------------------------------------------

# 5. Aggregation & Reproducibility

All metrics are:

-   Deterministic
-   Rule-based
-   Scenario-defined
-   Independent of model internals

Reproducibility requires:

-   Frozen scenario files
-   Logged model configuration
-   Logged num_ctx, temperature, num_predict
-   Deterministic scoring logic

No metric depends on manual annotation or subjective judgment.

------------------------------------------------------------------------

End of scoring specification.   