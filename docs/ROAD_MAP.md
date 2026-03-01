# Roadmap --- AI Evals

This roadmap aligns with the finalized V2 architecture described in:

-   README.md
-   RUNBOOK_v2.md
-   SCORING_v2.md
-   REPRODUCIBILITY_v2.md

It documents completed milestones and future directions.

------------------------------------------------------------------------

# V1 (v1_final) --- Completed

Focus: Single-turn factual reliability.

Delivered:

-   Stable prompt suite with deterministic expansion
-   Canonical numeric scoring for exact tasks
-   NIC metrics (global, conditional, explicit k/n)
-   Wilson confidence intervals
-   Stable leaderboard artifact
-   Fully reproducible offline scoring pipeline

Outcome:

Established a rigorous factual evaluation baseline for open-weight
models.

------------------------------------------------------------------------

# V2 --- Behavioral Reliability + Context Memory (Completed)

V2 expanded scope from factual correctness to structured multi-turn
reliability.

## Phase 1: Multi-Turn Behavioral Suite

Delivered:

-   Structured multi-turn scenario architecture
-   Deterministic check system:
    -   persona
    -   memory (short-horizon)
    -   format
    -   recovery
    -   consistency
    -   calibration
-   Suite expansion pipeline
-   Frozen benchmark (`v2_suite_50.jsonl`)
-   Behavioral Reliability Index (BRI)

Outcome:

Enabled controlled cross-model comparison for conversational
reliability.

------------------------------------------------------------------------

## Phase 2: Experiment Families

Delivered:

-   Controlled experiment abstraction (`experiments.yaml`)
-   Behavioral baseline (fixed operating point)
-   Model-size comparison (fixed context)
-   Tradeoff experiments (size vs context)
-   MoE vs dense comparison

Outcome:

Enabled causal interpretation of parameter scale vs context effects.

------------------------------------------------------------------------

## Phase 3: Context Cliff (MCS Track)

Delivered:

-   Dedicated context stress suite (`v2_ctx_cliff_suite_20.jsonl`)
-   Memory Compliance Score (MCS)
-   drop_turn metric
-   MCS AUC metric
-   Context-boundary identification plots

Outcome:

Demonstrated clear empirical separation between reasoning quality and
memory horizon.

------------------------------------------------------------------------

# Current State (v2_final)

The system now supports:

Behavioral Track: - Persona stability analysis - Instruction persistence
measurement - Multi-turn consistency scoring - Structured behavioral
aggregation (BRI)

Context Track: - Context-window failure boundary detection - Memory
degradation curve analysis - Controlled num_ctx sweeps

All metrics are deterministic and reproducible.

------------------------------------------------------------------------

# V3 --- Future Directions

V3 will extend from model-level evaluation toward consumer AI system
evaluation.

## 1. Memory Strategy Experiments

Compare:

-   Raw rolling window
-   Structured pinned memory
-   Summarization compression
-   Retrieval-based memory (RAG of past turns)

Measure impact on: - MCS - contradiction_rate - persona stability -
drop_turn distribution

------------------------------------------------------------------------

## 2. RAG vs No-RAG Tradeoff

Controlled experiment:

-   Same model
-   Same num_ctx
-   With and without retrieval augmentation

Hypothesis:

-   Factual accuracy ↑
-   Hallucination ↓
-   Multi-turn memory compliance may degrade due to context competition

------------------------------------------------------------------------

## 3. Long-Horizon Behavioral Stability

Extend: - 20-turn → 50+ turn scenarios - Drift accumulation tracking -
Stability half-life estimation

------------------------------------------------------------------------

## 4. System-Level Evaluation

Move from pure LLM evaluation to application-level reliability:

-   Tool use evaluation
-   Agent memory policy evaluation
-   Multi-session state persistence
-   Cost--reliability frontier modeling

------------------------------------------------------------------------

## 5. Hardware & Deployment Analysis

Integrate:

-   VRAM utilization logging
-   Context length vs memory failure vs cost curves
-   Multi-GPU KV sharding experiments
-   Deployment tradeoff analysis

------------------------------------------------------------------------

# Strategic Direction

V1 → factual correctness\
V2 → behavioral reliability + context memory\
V3 → system-level consumer AI evaluation

The roadmap transitions from model benchmarking toward production-grade
AI reliability evaluation.

------------------------------------------------------------------------

End of roadmap.