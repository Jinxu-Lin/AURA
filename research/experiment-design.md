---
version: "1.1"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "dr_revise"
iteration_major: 1
iteration_minor: 1
---

> **v1.1 dr_revise**: Revision of FM1/FM2 experiment design addressing design_review round-1 practical concerns. Changes: compute budget increased to ~210 GPU-hours; DDA downgraded to optional with Contrastive-TRAK as primary; gradient-norm baseline added; Hessian quality ablation added; interaction thresholds refined; timeline adjusted to 4-5 weeks.

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

**Models**: Pythia-1B (primary). Pythia-6.9B is an explicit stretch goal (NOT in core budget).
**Training configs**: Both LoRA and full fine-tuning (to test FM1 LoRA artifact hypothesis).

### 1.2 Supplementary: CIFAR-10/ResNet-18

**Purpose**: Small-scale validation leveraging AURA existing infrastructure.
**Configuration**: ResNet-18, CIFAR-10/50K, full-model Hessian, 5 seeds.
**Existing data**: AURA Phase 0-2b results (500 test points, EK-FAC IF, K-FAC IF, RepSim, TRAK).

## 2. Methods

### 2.1 Core Methods (2x2 Matrix)

| Cell | Method | Implementation | Notes |
|------|--------|---------------|-------|
| Param + Standard | TRAK | trak library (10+ checkpoints, JL dim=4096) | Random projection baseline |
| Param + Standard | IF (EK-FAC) | dattri IFAttributorEKFAC | Full parameter space |
| Param + Contrastive | Contrastive-TRAK | TRAK_{theta'} - TRAK_{theta_0} | **Primary** contrastive param method |
| Param + Contrastive | DDA | DDA codebase | **Optional** -- 40% failure risk on data selection |
| Repr + Standard | RepSim | Cosine similarity on penultimate layer | Task-structured feature space |
| Repr + Standard | RepT | concat[h^(l*), nabla_h L], auto layer selection | Enriched representation |
| Repr + Contrastive | Contrastive-RepSim | RepSim_{theta'} - RepSim_{theta_0} | Both FM1 + FM2 remedies |
| Repr + Contrastive | Contrastive-RepT | RepT_{theta'} - RepT_{theta_0} | Full remedy cell |

### 2.2 Additional Baselines

| Method | Purpose |
|--------|---------|
| Random | Sanity check |
| BM25 | Lexical baseline for factual attribution |
| LESS | Gradient projection baseline |
| Gradient-norm | Zero-cost sanity check: per-sample gradient norm as attribution proxy |
| AirRep | Learned representation baseline (if DATE-LM results available) |

## 3. Experimental Protocol

### 3.1 Multi-Seed Protocol

- **Seeds**: 5 per configuration (42, 123, 456, 789, 1024)
- **Reporting**: Mean +/- std, 95% CI via bootstrap (10,000 resamples)
- **Minimum**: 3 seeds for preliminary results; 5 seeds for final

### 3.2 Statistical Testing

**Per-sample analysis** (NOT task-level ANOVA):
- With only 3 DATE-LM tasks, task-level ANOVA has insufficient statistical power.
- Per-sample permutation tests within each task.
- Bootstrap confidence intervals for LDS differences.

**2x2 interaction analysis** (PRIMARY test of FM1/FM2 remedy additivity):
- For each task: compute main effects (representation vs parameter, contrastive vs standard) and interaction at per-sample level.
- Report interaction magnitude as fraction of minimum main effect.
- Interaction interpretation:
  - < 10% of min(main effects) AND Cohen's d < 0.2: strong approximate additivity
  - 10-30%: approximate additivity with noted interaction
  - > 30%: interacting remedies requiring joint treatment
- Permutation test: permute space and scoring labels within each sample, 10,000 iterations.

**Multiple comparison correction**: Bonferroni for pairwise method comparisons within each task.

### 3.3 Evaluation Metrics

| Metric | Description | Used For |
|--------|-------------|----------|
| LDS (Spearman) | Linear Datamodeling Score | Primary: all comparisons |
| P@K (K=10,50,100) | Precision at K | Practical: finding right training samples |
| Kendall tau | Rank correlation between methods | Diagnostic: IF-RepSim agreement |
| Cohen's d | Effect size for pairwise comparisons | Significance |

### 3.4 Ablation Protocol

1. **Space ablation**: Fix scoring type, vary space. Tests FM1.
2. **Scoring ablation**: Fix space, vary scoring. Tests FM2.
3. **Combined**: Full 2x2 with interaction analysis. Tests complementarity.
4. **Layer selection**: For representation methods, test middle/last/auto-selected layers.
5. **LoRA vs full FT**: Both training configs on DATE-LM. If FM1 is LoRA artifact, full FT shows LARGER repr-space advantage.
6. **Hessian quality**: EK-FAC vs K-FAC for IF on Pythia-1B. Tests whether Hessian approximation quality is independent factor.
7. **TRAK projection dimension**: dim=2048/4096/8192. Tests random projection ceiling.

## 4. Compute Budget

### 4.1 DATE-LM Core Experiments

| Component | GPU-hours (est.) | GPU type |
|-----------|-----------------|----------|
| Pythia-1B fine-tuning (5 seeds x 2 configs) | 25 | RTX 4090 |
| TRAK computation (10 ckpts x 3 tasks + dim ablation) | 35 | RTX 4090 |
| IF (EK-FAC) computation (3 tasks) | 45 | A6000 |
| IF (K-FAC) computation (3 tasks) | 15 | A6000 |
| RepSim/RepT computation (3 tasks) | 10 | RTX 4090 |
| Contrastive-TRAK (3 tasks) | 20 | RTX 4090 |
| Contrastive-RepSim/RepT (3 tasks) | 10 | RTX 4090 |
| Gradient-norm baseline | <1 | RTX 4090 |
| **Core subtotal** | **~161** | |

### 4.2 Debugging and Setup Buffer

| Component | GPU-hours (est.) |
|-----------|-----------------|
| DATE-LM environment setup + validation | 10 |
| EK-FAC numerical stability debugging | 10 |
| Re-runs for failed/corrupted experiments | 15 |
| **Buffer subtotal** | **~35** |

### 4.3 CIFAR-10 Supplementary (Largely Completed)

Remaining: contrastive scoring variants (~5 GPU-hours).

### 4.4 Total Budget

| Category | GPU-hours |
|----------|-----------|
| DATE-LM core | ~161 |
| Debug/setup buffer | ~35 |
| CIFAR-10 remaining | ~5 |
| **Total** | **~201** |
| **With 10% contingency** | **~220** |

### 4.5 Available Resources

- **xuchang0**: 4x RTX 4090 (24GB each)
- **jinxulin**: 4x A6000 (48GB each)
- **Estimated timeline**: 4-5 weeks for core DATE-LM experiments

**Pythia-6.9B**: NOT in core budget. Stretch goal only.

## 5. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| RepSim < TRAK - 5pp on all tasks | 30% | Weakens FM1 narrative | Reframe as "different strengths"; RepT may recover |
| DDA not applicable to data selection | 40% | Incomplete 2x2 for 1 task | Contrastive-TRAK is primary; DDA optional |
| EK-FAC OOM on Pythia-1B | 25% | Cannot compute IF | K-FAC fallback; LoRA-only gradients |
| Interaction > 30% | 20% | Complementarity requires reinterpretation | Design handles both outcomes |
| DATE-LM setup fails | 15% | Week 1 blocker | Full Week 1 for setup; manual pipeline fallback |
| Concurrent work | 15% | Reduced novelty | Framework + TRAK paradox analysis differentiate |

## 6. Back-References

- Method design: research/method-design.md (FM1/FM2 framework, TRAK paradox)
- Problem statement: research/problem-statement.md (Gap, RQs, prior evidence)
- Probe results: Codes/_Results/probe_result.md (AURA Phase 0-2b)
- Design review: Reviews/research-design/round-1/synthesis.md (M1-M4, S1-S5)

## 7. Metadata

- **Primary benchmark**: DATE-LM (3 tasks, LDS, Pythia-1B)
- **Supplementary**: CIFAR-10/ResNet-18 (AURA existing data)
- **Statistical methods**: Per-sample permutation/bootstrap, 5 seeds, 2x2 interaction, Bonferroni
- **Compute**: ~210 GPU-hours (core + buffer), ~220 with contingency
- **Timeline**: 4-5 weeks
- **Resources**: 4x RTX 4090 + 4x A6000
- **Design review addressal**: S3 (budget revised); DDA optional; gradient-norm baseline; Hessian ablation; Pythia-6.9B stretch goal; interaction thresholds refined
