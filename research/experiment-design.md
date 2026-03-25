---
version: "1.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "assimilated"
iteration_major: 1
iteration_minor: 0
---

> [ASSIMILATED: generated from CRA_old design + AURA experimental protocol]

# Experiment Design

## 1. Benchmarks

### 1.1 Primary: DATE-LM

**Benchmark**: DATE-LM (NeurIPS 2025), the standard LLM TDA evaluation benchmark.
**GitHub**: DataAttributionEval/DATE-LM
**Tasks**:
1. **Data Selection**: Select training subsets that maximize downstream performance.
2. **Toxicity Filtering**: Identify training samples causing toxic model outputs.
3. **Factual Attribution**: Trace model predictions back to factual training sources.

**Metric**: Linear Datamodeling Score (LDS) -- Spearman correlation between predicted and actual leave-one-out effects.

**Models**: Pythia-1B (primary), optionally Pythia-6.9B (scaling validation).
**Training configs**: Both LoRA and full fine-tuning (to test FM1 LoRA artifact hypothesis).

### 1.2 Supplementary: CIFAR-10/ResNet-18

**Purpose**: Small-scale validation leveraging AURA's existing experimental infrastructure.
**Configuration**: ResNet-18, CIFAR-10/50K, full-model Hessian, 5 seeds.
**Existing data**: AURA Phase 0-2b results (500 test points, EK-FAC IF, K-FAC IF, RepSim, TRAK).

This supplementary benchmark serves two roles:
1. Validates that findings generalize across model scales (small CNN vs LLM).
2. Provides ablation detail impossible at LLM scale (full Hessian computation, exact eigendecomposition).

## 2. Methods

### 2.1 Core Methods (2x2 Matrix)

| Cell | Method | Implementation |
|------|--------|---------------|
| Param + Standard | TRAK | trak library (10+ checkpoints, JL dim=4096) |
| Param + Standard | IF (EK-FAC) | dattri IFAttributorEKFAC |
| Param + Contrastive | DDA | DDA codebase (IS_{theta'} - IS_{theta_0}) |
| Repr + Standard | RepSim | Custom (cosine similarity on penultimate layer) |
| Repr + Standard | RepT | Custom (concat[h^(l*), nabla_h L], auto layer selection) |
| Repr + Contrastive | Contrastive-RepSim | RepSim_{theta'} - RepSim_{theta_0} |
| Repr + Contrastive | Contrastive-RepT | RepT_{theta'} - RepT_{theta_0} |

### 2.2 Additional Baselines

| Method | Purpose |
|--------|---------|
| Random | Sanity check |
| BM25 | Lexical baseline for factual attribution |
| LESS | Gradient projection baseline |
| AirRep | Learned representation baseline (if DATE-LM results available) |

## 3. Experimental Protocol

### 3.1 Multi-Seed Protocol

- **Seeds**: 5 per configuration (42, 123, 456, 789, 1024)
- **Reporting**: Mean +/- std, 95% CI via bootstrap
- **Minimum**: 3 seeds for preliminary results; 5 seeds for final

### 3.2 Statistical Testing

**Per-sample analysis** (NOT task-level ANOVA):
- With only 3 DATE-LM tasks, task-level ANOVA has insufficient statistical power.
- Instead: per-sample permutation tests within each task.
- Bootstrap confidence intervals for LDS differences.

**2x2 interaction analysis**:
- For each task: compute main effects (representation vs parameter, contrastive vs standard) and interaction at the per-sample level.
- Report interaction magnitude as fraction of minimum main effect.
- Threshold: interaction < 30% of min(main effects) supports FM1/FM2 independence.

**Multiple comparison correction**: Bonferroni for pairwise method comparisons within each task.

### 3.3 Evaluation Metrics

| Metric | Description | Used For |
|--------|-------------|----------|
| LDS (Spearman) | Linear Datamodeling Score | Primary: all comparisons |
| P@K (K=10,50,100) | Precision at K | Practical: which method finds the right training samples |
| Kendall tau | Rank correlation between methods | Diagnostic: IF-RepSim agreement |
| Cohen's d | Effect size for pairwise comparisons | Significance: practical importance |

### 3.4 Ablation Protocol

1. **Space ablation**: Fix scoring type, vary space (parameter vs representation). Tests FM1.
2. **Scoring ablation**: Fix space, vary scoring (standard vs contrastive). Tests FM2.
3. **Combined**: Full 2x2 with interaction analysis. Tests independence.
4. **Layer selection**: For representation methods, test multiple layers (middle, last, auto-selected).
5. **LoRA vs full FT**: Both training configs on DATE-LM to test FM1 LoRA artifact hypothesis.

## 4. Compute Budget

### 4.1 DATE-LM Experiments

| Component | GPU-hours (est.) | GPU type |
|-----------|-----------------|----------|
| Pythia-1B fine-tuning (5 seeds x 2 configs) | 20 | RTX 4090 |
| TRAK computation (10 checkpoints x 3 tasks) | 30 | RTX 4090 |
| IF (EK-FAC) computation (3 tasks) | 40 | A6000 |
| RepSim/RepT computation (3 tasks) | 10 | RTX 4090 |
| DDA computation (3 tasks) | 40 | A6000 |
| Contrastive-RepSim/RepT (3 tasks) | 15 | RTX 4090 |
| **Total** | **~155** | |

### 4.2 CIFAR-10 Supplementary (Largely Completed)

AURA has already computed: EK-FAC IF, K-FAC IF, RepSim, TRAK for 500 test points x 50K train on full-model ResNet-18. Remaining: contrastive scoring variants (~5 GPU-hours).

### 4.3 Available Resources

- **xuchang0**: 4x RTX 4090 (24GB each)
- **jinxulin**: 4x A6000 (48GB each)
- **Estimated timeline**: 3-4 weeks for core DATE-LM experiments

## 5. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| RepSim < TRAK - 5pp on all DATE-LM tasks | 30% | Weakens unified superiority narrative | Reframe as "different strengths" + leverage AURA prior data |
| DDA contrastive scoring not applicable to data selection | 25% | Incomplete 2x2 matrix for 1/3 tasks | Design task-appropriate contrastive reference |
| Per-sample gradient OOM on Pythia-6.9B | 40% | Cannot run scaling validation | Use gradient checkpointing + LoRA-only gradients |
| Interaction term too large (>30%) | 20% | FM1/FM2 independence claim weakened | Reframe as "partially overlapping remedies" |
| Concurrent work fills DATE-LM gap | 15% | Reduced novelty | Accelerate submission; diagnostic framework provides differentiation |

## 6. Back-References

- Method design: research/method-design.md (FM1/FM2 framework, 2x2 design rationale)
- Problem statement: research/problem-statement.md (Gap, RQ1-RQ3, prior evidence)
- Probe results: Codes/_Results/probe_result.md (AURA Phase 0-2b data)

## 7. 元数据

- **主要 benchmark**: DATE-LM (3 tasks, LDS metric)
- **补充 benchmark**: CIFAR-10/ResNet-18 (AURA existing data)
- **统计方法**: Per-sample permutation/bootstrap, multi-seed (5 seeds)
- **预计时间**: 3-4 weeks for core DATE-LM experiments
- **计算资源**: 4x RTX 4090 (xuchang0) + 4x A6000 (jinxulin)
