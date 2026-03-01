# Methodology (V2 --- Behavioral Reliability + Context Memory)

This document explains how AI Evals v2 is constructed, how experiments
are designed, and how metrics are computed and interpreted.

V2 extends V1 from single-turn factual evaluation into structured
multi-turn behavioral reliability and context-boundary memory analysis.

------------------------------------------------------------------------

# 1. Evaluation Philosophy

V2 is designed around three principles:

1.  **Separation of concerns**
    -   Behavioral reliability (persona, instruction persistence,
        consistency)
    -   Context-window memory mechanics (eviction and collapse)
2.  **Deterministic scoring**
    -   All metrics are rule-based
    -   No manual labeling
    -   No LLM-as-judge scoring
3.  **Controlled experiments**
    -   Same frozen scenario suite
    -   Only one variable changes per experiment family

------------------------------------------------------------------------

# 2. Scenario Construction

## 2.1 Base scenarios

Base multi-turn templates are defined in:

-   `data/v2_scenarios.jsonl`

Each scenario includes: - `scenario_id` - `tags` - `checks` - multi-turn
dialogue structure

Checks define what to evaluate: - persona - memory - format - recovery -
calibration - consistency

## 2.2 Expansion

To increase coverage, base scenarios are expanded into variants:

-   `data/v2_scenarios_expanded.jsonl`

Expansion may vary: - persona style - instruction framing - adversarial
pressure - structured values

## 2.3 Frozen benchmark

The expanded set was frozen into:

-   `data/v2_suite_50.jsonl`

This frozen suite is used for all behavioral experiments to ensure
comparability across runs.

A separate frozen suite exists for memory stress testing:

-   `data/v2_ctx_cliff_suite_20.jsonl`

------------------------------------------------------------------------

# 3. Experimental Design

Experiments are defined in:

-   `configs/experiments.yaml`
-   `configs/experiments_ctx_cliff.yaml`

Each experiment family reruns the same frozen suite while varying a
single dimension.

## 3.1 Behavioral baseline

Purpose: - Establish comparable reliability across models at a standard
operating point (e.g., `num_ctx=2048`)

## 3.2 Model size (fixed context)

Purpose: - Hold `num_ctx` constant - Compare 3B vs 14B vs 20B class
models - Isolate parameter scale effects

## 3.3 Tradeoff (size vs context)

Purpose: - Compare medium model with longer context vs larger model with
shorter context - Quantify real-world deployment tradeoffs

## 3.4 Context cliff sweep

Purpose: - Fix model - Sweep `num_ctx` (1K → 8K) - Identify boundary
where memory collapses

This is intentionally separated from behavioral reliability to isolate
context-window mechanics.

------------------------------------------------------------------------

# 4. Behavioral Metrics

Behavioral scoring is produced by:

-   `scripts/v2/scorecard.py`

Metrics are computed per scenario per run, then aggregated.

## 4.1 Persona Stability Score (PSS)

Definition: Fraction of persona-required turns that satisfy persona
constraints.

Observed baseline: \~0.33--0.43 across evaluated models.

Interpretation: Persona maintenance is one of the hardest behavioral
components.

------------------------------------------------------------------------

## 4.2 Memory Score (Behavioral)

Definition: Structured short-horizon recall accuracy within normal
dialogue length.

Observed baseline: \~1.0 across evaluated models.

Interpretation: Behavioral suite memory checks are lightweight and
within context limits.\
Stress testing of memory is handled separately in the context cliff
track.

------------------------------------------------------------------------

## 4.3 Format Score

Definition: Fraction of formatting constraints satisfied (e.g., JSON
keys, required structure).

------------------------------------------------------------------------

## 4.4 Recovery Score

Definition: Whether the model corrects itself after an explicit error
signal.

------------------------------------------------------------------------

## 4.5 Consistency / Contradiction Rate

Definition: Rate of detected contradictions across turns.

Observed baseline consistency: \~0.75--0.91 depending on model.

------------------------------------------------------------------------

## 4.6 Calibration

Definition: Average absolute difference between stated confidence and
correctness (when required).

------------------------------------------------------------------------

## 4.7 Behavioral Reliability Index (BRI)

Definition: Mean of all applicable behavioral components for a scenario.

Only metrics present for a scenario are included in the mean.

Observed baseline BRI: \~0.55--0.61 across tested models.

------------------------------------------------------------------------

# 5. Context Cliff Metrics

Computed via:

-   `memory_compliance.py`

## 5.1 Memory Compliance Score (MCS)

Definition: Fraction of required memory checkpoints satisfied across
turns.

------------------------------------------------------------------------

## 5.2 drop_turn_mean

Definition: Average turn index where memory first fails.

------------------------------------------------------------------------

## 5.3 MCS AUC

Definition: Area under MCS-by-turn curve, capturing prefix stability.

------------------------------------------------------------------------

## 5.4 Observed Example (Qwen 14B)

  num_ctx   MCS
  --------- ------
  1024      0.33
  2048      0.33
  4096      1.00
  6144      1.00
  8192      1.00

Interpretation: Clear reliability boundary between 2K and 4K context
length.

Conclusion: Multi-turn reliability is strongly influenced by context
window size, not just model scale.

------------------------------------------------------------------------

# 6. Enrichment Layer

Enrichment joins run metadata with metric outputs.

It appends: - profile - model - cell_id - family - temperature -
num_ctx - host

Purpose: - Enable grouping by experiment family - Plot MCS vs context -
Compare cells cleanly

Enrichment never modifies metric values.

------------------------------------------------------------------------

# 7. Interpretation Framework

V2 enables separate conclusions:

Behavioral track answers: - Does the model maintain persona? - Does it
remain consistent? - Does it follow instructions over multiple turns?

Context track answers: - At what context size does memory collapse? -
How far into the dialogue does reliability persist? - Is context window
the bottleneck?

This separation produces defensible, causal insights.

------------------------------------------------------------------------

End of methodology.