# AI Evals v2 --- Extended Academic Empirical Analysis

## 1. Overview

This document provides a formal, effect-size--oriented interpretation of
AI Evals v2 results across:

-   Multi-turn Behavioral Reliability
-   Controlled Model-Scale Experiments
-   Context-Window Memory Compliance (Context Cliff)

All results derive from frozen benchmark suites and deterministic
scoring.

------------------------------------------------------------------------

# 2. Behavioral Reliability --- Quantitative Analysis

## 2.1 Aggregate Model Results

  --------------------------------------------------------------------------------
  Model                             BRI   Persona_PSS   Memory_Score   Consistency
  ---------------------------- -------- ------------- -------------- -------------
  qwen2.5:3b                     0.5545        0.3333              1        0.7647

  qwen2.5:14b                    0.6042        0.4298              1          0.75

  gpt-oss-safeguard:20b          0.5962        0.3553              1        0.8382

  gemma2:9b-instruct-q4_K\_M     0.6098        0.3596              1        0.8603

  phi3:mini                       0.609        0.3903              1        0.8971

  mixtral:8x7b                   0.6042        0.3377              1        0.9118
  --------------------------------------------------------------------------------

------------------------------------------------------------------------

## 2.2 Effect Size --- Behavioral Reliability Index (BRI)

Minimum BRI: 0.5545\
Maximum BRI: 0.6098\
Absolute ΔBRI: 0.0553\
Relative improvement: 9.97%

Interpretation:

-   The total spread across models is 0.0553, representing only a 9.97%
    improvement from lowest to highest.
-   This indicates **incremental scale benefit**, not structural
    reliability transformation.
-   Multi-turn conversational robustness remains bounded near \~0.61
    even for larger models.

------------------------------------------------------------------------

## 2.3 Effect Size --- Persona Stability (PSS)

Minimum PSS: 0.3333\
Maximum PSS: 0.4298\
Absolute ΔPSS: 0.0965

Interpretation:

-   Persona adherence exhibits moderate variance but remains \<0.43 even
    at best.
-   This suggests persona persistence is an unsolved stability problem
    independent of memory mechanics.

------------------------------------------------------------------------

## 2.4 Effect Size --- Consistency

Minimum Consistency: 0.75\
Maximum Consistency: 0.9118\
Absolute ΔConsistency: 0.1618

Interpretation:

-   Coherence improvements across architectures are meaningful but
    insufficient to eliminate drift.
-   Mixture-of-Experts and certain mid-sized models demonstrate improved
    internal stability.

------------------------------------------------------------------------

# 3. Context Cliff --- Formal Interpretation

Observed MCS values for Qwen 14B:

  num_ctx   MCS
  --------- --------
  1024      0.3333
  2048      0.3333
  4096      1.0000
  6144      1.0000
  8192      1.0000

Effect Observation:

ΔMCS (2048 → 4096) = 0.6667 (absolute jump)

Interpretation:

-   The reliability gain from doubling context window is discontinuous
    and non-linear.
-   Memory failure behaves like a threshold phenomenon rather than
    gradual degradation.
-   Context allocation has a larger reliability effect size than
    moderate model scaling.

------------------------------------------------------------------------

# 4. Comparative Insight --- Scale vs Context

Scale Effect (BRI spread): 0.0553\
Context Effect (MCS jump): 0.6667

Conclusion:

The magnitude of reliability improvement from context expansion is an
order of magnitude larger than improvement from parameter scaling.

Implication:

Deployment optimization should prioritize context window allocation
before moderate increases in parameter count.

------------------------------------------------------------------------

# 5. Formal Conclusions

1.  Multi-turn behavioral reliability remains structurally constrained
    (\~0.60 ceiling).
2.  Persona persistence is a primary instability driver.
3.  Short-horizon memory recall is solved under normal context.
4.  Long-horizon memory collapse is context-bound, not reasoning-bound.
5.  Context window size produces discontinuous reliability improvements.
6.  Parameter scale yields incremental gains but not structural shifts.

------------------------------------------------------------------------

End of extended empirical analysis.