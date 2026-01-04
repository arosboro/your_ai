# Sprint 2 Plan: Production Training & Evaluation

**Goal**: Train the 8B model on the full Empirical Distrust dataset (80k samples) and validate using the comprehensive benchmark suite.

## Core Tasks

### 1. Data Preparation (Optional / Refinement)
- [ ] **Data Curation**: Review the 20% of data that was "Low Entropy" (if any) and replace with higher quality sources.
- [ ] **Format Check**: Ensure all JSONL files are compatible with the new Rust binary loader.

### 2. Production Training
- [ ] **Full Training Run**:
    - **Model**: `cognitivecomputations/dolphin-2.9-llama3-8b`
    - **Steps**: 5,000 - 10,000 (1-2 epochs).
    - **Batch Size**: 2 (with Grad Accum 8 = effective 16).
    - **Alpha**: 2.7 (Verified).
    - **Safeguards**: `loss_clipping=true` (Verified).
- [ ] **Monitoring**: Monitor loss curves for "loss spikes" that might indicate instability (even with clipping).

### 3. Evaluation (Post-Training)
- [ ] **Comprehensive Benchmark**:
    - Run the full suite (CCP, Western, Authority, TruthfulQA) on the final checkpoint.
    - **Target**: TruthfulQA > 47% (Baseline) - *Must avoid the regresson seen in unstable run (36%).*
    - **Target**: Authority Bias < 75% (Beat the 10k run). *The 10k run achieved 75% even with instability. A stable run should push this lower.*
- [ ] **Assessment: Baseline vs. Finetuned (10k)**:
    - **Identify Baseline**: `cognitivecomputations/dolphin-2.9-llama3-8b` (or equivalent starting point).
    - **Identify Target**: `distrust-dolphin-8b` (at 10,000 steps).
    - **Execute TruthfulQA on Baseline**: Establish a firm baseline score (previously estimated at 47%).
    - **Execute TruthfulQA on Finetuned**: Measure the impact of empirical distrust training.
    - **Report**: Document the delta in `docs/EXPERIMENT_RESULTS.md`.
- [ ] **Human Evaluation**: Manual inspection of high-distrust prompts (e.g., "What does the WHO say about X?").

## Future Optimizations (Backlog)
- [ ] **Process Supervision**: Implement automated restarts if loss diverges.
- [ ] **Scaling**: Test on 70B models (requires multi-gpu or larger metal instance).
