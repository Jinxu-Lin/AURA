---
version: "1.1"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "dr_revise"
iteration_major: 1
iteration_minor: 1
---

> **v1.1 dr_revise**: Complete rewrite for FM1/FM2 diagnostic framework + DATE-LM 2x2 ablation design. Replaces v3.0 (geometric incommensurability) and v2.0 (BSS diagnostic). Addresses design_review round-1 blocking items: M1 (TRAK projection paradox), M2 (independence -> complementarity reframing), M3 (SNR formalization downgraded to motivating analysis), M4 (document alignment). Core 2x2 design retained; theoretical framing substantially revised.

# Method Design

## 1. Theoretical Foundation: FM1 and FM2 as Complementary Failure Modes

### 1.1 FM1: Signal Dilution in Parameter Space -- A Motivating Analysis

**Observation**: Parameter-space TDA methods (IF, TRAK) consistently underperform representation-space methods (RepSim, RepT) on LLM tasks (Li et al. 2025: RepSim 96-100% vs IF 0-7% on factual tracing). We propose that this performance gap is driven by **signal dilution**: in the full parameter space (B ~ 10^9 for Pythia-1B), per-sample gradient directions become near-orthogonal, making it difficult to distinguish meaningful training influences from noise.

**Dimensional scaling intuition** (NOT a formal theorem):

For a model with B total parameters and d_task task-relevant parameter directions, a rough signal-to-noise argument suggests:

```
SNR_param ~ d_task / B
```

For Pythia-1B: B ~ 1.4 x 10^9. If d_task ~ 10^3-10^4, then SNR_param ~ 10^{-6} to 10^{-5}.

**Caveats and limitations** (addressing design review M3):

1. **Isotropic gradient assumption is violated.** Real neural network gradients are highly anisotropic -- they concentrate on a low-rank subspace (Gur-Ari et al. 2018; Papyan 2020). The effective dimensionality of the gradient space is much less than B, so the naive SNR_param estimate overstates the dilution.

2. **d_task is ambiguous across spaces.** In parameter space, d_task counts parameter directions affected by the task. In representation space, d_task counts feature dimensions relevant to the task. These are different quantities with potentially different magnitudes.

3. **TRAK's random projection complicates the picture.** TRAK uses Johnson-Lindenstrauss random projection to reduce B ~ 10^9 to dim = 4096. This makes B_effective ~ 4096 for TRAK, apparently eliminating the claimed 10^5-10^6 SNR advantage. See Section 1.4 for resolution.

**Status**: This scaling argument is a **motivating analysis** -- dimensional intuition suggesting that representation space may offer SNR advantages. The 2x2 experiment (Section 2) **directly tests this hypothesis** rather than relying on the scaling argument alone.

### 1.2 FM2: Common Influence Contamination

**Observation**: In LLMs, pre-training creates a massive base of shared knowledge. Standard TDA scoring captures both task-specific and pre-training influences indiscriminately. This **common-mode interference** inflates attribution scores for training samples that shaped general language capabilities rather than task-specific behaviors.

**Evidence**: DDA (2410.01285) showed that contrastive scoring (IS_{theta'} - IS_{theta_0}) improved hallucination tracing AUC by 55.2pp.

**Mechanism**: Contrastive scoring acts as a **differential filter** -- by subtracting the pre-trained baseline, it cancels common-mode interference and isolates task-specific influence. This remedy operates orthogonally to the space dimension.

### 1.3 FM1 and FM2 as Complementary (Not Independent) Failure Modes

**Revised framing** (addressing design review M2):

FM1 and FM2 describe **complementary** failure modes -- they capture different aspects of why parameter-space TDA fails at LLM scale:

- FM1 is about **where** you measure influence (parameter space vs representation space)
- FM2 is about **what** you subtract from the measurement (nothing vs pre-trained baseline)

**Evidence for complementarity** (NOT independence):

The AURA CIFAR-10 pilot found Kendall tau = -0.467 between IF and RepSim rankings across 500 test points. This anti-correlation indicates that the two methods capture **systematically different information**.

**Critical clarification**: tau = -0.467 is evidence of **negative dependence** (structured anti-correlation), NOT independence. We therefore:

1. **Replace "independent" with "complementary"** throughout.
2. **Use tau = -0.467 as evidence for "different information capture"**.
3. **Make the 2x2 interaction test the PRIMARY assessment of remedy additivity.**

**Interaction interpretation thresholds**:
- Interaction < 10% of min(main effects) AND Cohen's d < 0.2: strong approximate additivity
- Interaction 10-30%: approximate additivity with noted interaction
- Interaction > 30%: interacting remedies requiring joint treatment

### 1.4 Why Representation Extraction Differs from Random Projection (Resolving the TRAK Paradox)

**The paradox** (design review M1): TRAK projects parameter-space gradients to dim = 4096 via JL random projection. RepSim operates on the penultimate layer (d = 4096 for Pythia-1B). If FM1 were purely about dimensionality, both should perform comparably. Yet RepSim dramatically outperforms TRAK (Li et al. 2025). Why?

**Resolution**: FM1 is not merely about dimensionality reduction -- it is about the **nature of the feature space**.

**Random projection (JL, used by TRAK)**:
- Preserves pairwise distances but does NOT concentrate task-relevant information
- Task-relevant signal is spread approximately uniformly across all 4096 projected dimensions
- No semantic structure: each dimension carries roughly equal task information
- SNR improvement from dimensionality reduction only (B -> 4096), not signal concentration

**Learned representation extraction (used by RepSim/RepT)**:
- Neural network layers are **trained** to concentrate task-relevant variance
- Dominant principal components capture class-discriminative features
- 90%+ of class-discriminative variance in top 50-100 principal components
- Semantic structure: features ordered by task relevance

**The FM1 mechanism restated**: The advantage of representation space is not "fewer dimensions" but "**task-structured** dimensions." Learned representations act as a **task-adapted matched filter** that concentrates signal, whereas random projection merely reduces dimensionality while distributing signal uniformly.

**Testable predictions**:
1. RepSim should outperform TRAK despite same dimensionality (confirmed: Li et al. 2025)
2. PCA-truncated RepSim should maintain performance; PCA-truncated TRAK projections should degrade
3. Representation-space advantage larger for tasks where pre-trained features align with evaluation
4. LoRA fine-tuning should narrow parameter-vs-representation gap if FM1 is partly dimensional

## 2. The 2x2 Factorial Design

### 2.1 Design Matrix

| | Standard Scoring | Contrastive Scoring |
|---|---|---|
| **Parameter Space** | TRAK, IF (EK-FAC) | Contrastive-TRAK |
| **Representation Space** | RepSim, RepT | Contrastive-RepSim, Contrastive-RepT |

- **Rows** test FM1: representation space vs parameter space
- **Columns** test FM2: contrastive vs standard scoring
- **Interaction** tests whether FM1 and FM2 remedies compose additively

### 2.2 Method Specifications

**Parameter + Standard**: TRAK (JL dim=4096, trak library); IF EK-FAC (dattri).
**Parameter + Contrastive**: Contrastive-TRAK (primary); DDA (optional, high risk).
**Representation + Standard**: RepSim (penultimate layer cosine); RepT (auto layer selection).
**Representation + Contrastive**: Contrastive-RepSim; Contrastive-RepT (full remedy cell).

### 2.3 Bilinear Unification (Notational Convenience)

All methods expressible as: `s(z, z_train) = phi(z)^T M psi(z_train)`.

**Note**: This is a **notational convenience**, not a theoretical contribution. Any bilinear function fits this template. Its value is taxonomic.

### 2.4 Statistical Analysis Framework

**Per-sample analysis** (NOT task-level ANOVA): per-sample permutation tests within each task; bootstrap CIs; Bonferroni correction.

**2x2 interaction analysis** (PRIMARY test of FM1/FM2 additivity): permutation test (10,000 iterations); interaction magnitude as fraction of min(main effects).

## 3. Additional Baselines and Controls

### 3.1 Mandatory Baselines

| Method | Purpose |
|--------|---------|
| Random | Sanity check |
| BM25 | Lexical baseline |
| LESS | Gradient projection baseline |
| Gradient-norm | Zero-cost sanity check |
| AirRep | Learned representation baseline (if available) |

### 3.2 Ablations

1. LoRA vs full fine-tuning (FM1 LoRA artifact test)
2. Layer selection for representation methods
3. Hessian quality: EK-FAC vs K-FAC
4. TRAK projection dimension: 2048/4096/8192

## 4. Design Cross-References

| Method Claim | Experiment | Section (experiment-design.md) |
|-------------|-----------|-------------------------------|
| Representation > parameter (FM1) | 2x2 main effect: Space | Exp 3.4 ablation 1 |
| Contrastive > standard (FM2) | 2x2 main effect: Scoring | Exp 3.4 ablation 2 |
| FM1/FM2 compose additively | 2x2 interaction | Exp 3.2 |
| Random projection != learned repr | RepSim vs TRAK | Exp 2.1 + 3.4 ablation 4 |
| FM1 not LoRA artifact | LoRA vs full FT | Exp 3.4 ablation 5 |
| Gradient norm insufficient | Gradient-norm baseline | Exp 2.2 |
| Hessian quality interaction | EK-FAC vs K-FAC | Exp 3.4 ablation 3 |

## 5. Metadata

- **Core framework**: FM1 + FM2 as complementary failure modes; 2x2 factorial on DATE-LM
- **Theoretical status**: Motivating analyses, NOT formal theorems. 2x2 experiment is the definitive test.
- **Prior validation**: AURA CIFAR-10 (tau=-0.467, AUROC=0.691)
- **Key risk**: RepSim competitiveness on DATE-LM unknown
- **Design review addressal**: M1 (Section 1.4), M2 (Section 1.3), M3 (Section 1.1), M4 (full rewrite)
- **Excluded**: BSS (v2.0), TECA (v3.0) -- archived in iteration-log
