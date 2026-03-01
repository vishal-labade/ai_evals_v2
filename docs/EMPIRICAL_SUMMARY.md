# AI Evals v2 --- Empirical Findings and System-Level Insights

## 1. Executive Summary

AI Evals v2 extends single-turn factual evaluation (V1) into structured
multi-turn behavioral reliability and context-window memory analysis.

This merged document integrates: - Original qualitative analysis -
Complete quantitative behavioral tables - Controlled experiment
comparisons - Context-cliff sweep results

------------------------------------------------------------------------

# 2. Behavioral Baseline --- Multi-Turn Reliability

Behavioral evaluation was conducted on a frozen 50-scenario multi-turn
suite.

## 2.1 Aggregate Model Metrics

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

### Observations

-   BRI ranges from **0.5545 to 0.6098**
-   Persona stability (PSS) ranges from **0.3333 to 0.4298**
-   Memory score is uniformly **1.0**
-   Consistency ranges from **0.75 to 0.9118**

### Interpretation

-   Multi-turn conversational reliability remains moderate (\~0.60
    ceiling).
-   Persona adherence is the dominant failure mode.
-   Short-horizon recall is effectively solved.
-   Consistency varies meaningfully across architectures.

------------------------------------------------------------------------

# 3. Experiment Families --- Controlled Comparisons

## Model Size (Fixed Context @ 2048)

  del BR                       I Pe     rsona_PSS Co   nsistency
  ---------------------------- -------- -------------- -----------
  qwen2.5:3b                   0.5545   0.3333         0.7647
  qwen2.5:14b                  0.6042   0.4298         0.7500
  gpt-oss-safeguard:20b        0.5962   0.3553         0.8382
  gemma2:9b-instruct-q4_K\_M   0.6098   0.3596         0.8603
  phi3:mini                    0.6090   0.3903         0.8971
  mixtral:8x7b                 0.6042   0.3377         0.9118

### Observations

-   Scale improves BRI modestly (3B → \~0.55 vs larger models \~0.60).
-   Consistency improves for some architectures (MoE strongest).
-   Gains are incremental, not step-function.

------------------------------------------------------------------------

# 4. Context Cliff --- Memory Compliance Sweep

## Qwen 14B Context Sweep Results

  m_ctx MC   S dr     op_turn_mean MC   S_AUC
  ---------- -------- ----------------- -------
  1024       0.3333   7.0               0.7
  2048       0.3333   7.0               0.7
  4096       1.0000   ---               1.0
  6144       1.0000   ---               1.0
  8192       1.0000   ---               1.0

### Observations

-   Clear discontinuity between 2048 and 4096 tokens.
-   Memory collapse is abrupt rather than gradual.
-   Increasing context eliminates failures without changing weights.

### Interpretation

-   Memory degradation is driven by context eviction.
-   Context window size is a primary deployment lever.
-   Parameter scale alone does not resolve long-horizon memory.

------------------------------------------------------------------------

# 5. Cross-Track Synthesis

  mension Be               havioral Track Co          ntext Track
  ------------------------ -------------------------- ----------------------
  Primary Failure          Persona drift              Context eviction
  Memory (short horizon)   Solved                     ---
  Memory (long horizon)    ---                        Context-bound
  Scale Impact             Incremental                Secondary to context
  Deployment Lever         Prompting / architecture   Context allocation

------------------------------------------------------------------------

# 5. System-Level Conclusions

1.  Multi-turn reliability remains unsolved (BRI \~0.60 ceiling).
2.  Persona persistence is harder than factual recall.
3.  Context window size has discontinuous reliability effects.
4.  Deployment optimization should prioritize context allocation before
    moderate model scaling.
5.  Evaluation frameworks must separate behavioral reliability from
    memory mechanics.

------------------------------------------------------------------------

------------------------------------------------------------------------

# 6. Strategic Conclusions

1.  Multi-turn reliability remains unsolved (BRI \~0.60 ceiling).
2.  Persona persistence is harder than factual recall.
3.  Context window size has discontinuous reliability effects.
4.  Deployment optimization should prioritize context allocation before
    moderate model scaling.
5.  Evaluation frameworks must separate behavioral reliability from
    memory mechanics.

------------------------------------------------------------------------

End of merged empirical summary.