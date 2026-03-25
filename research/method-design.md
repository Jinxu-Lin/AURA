---
version: "1.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "assimilated"
iteration_major: 1
iteration_minor: 0
---

> [ASSIMILATED: generated from CRA_old + AURA methodology]

# Method Design

## 1. FM1/FM2 Diagnostic Framework

### 1.1 Signal-Processing Formalization

We frame TDA failure through two independent signal-processing defects:

**FM1 (Signal Dilution)**: In parameter space R^B (B ~ 10^9), per-sample gradients are near-orthogonal (Johnson-Lindenstrauss). The task-relevant signal subspace has dimension d_task << B, so the Signal-to-Noise Ratio of gradient-based attribution collapses:

```
SNR_param ~ d_task / B -> 0   as B -> infinity
```

Representation-space operation acts as **matched filtering**: by projecting to layer activations h^(l) in R^d (d ~ 4096), we concentrate signal in the task-relevant subspace:

```
SNR_repr ~ d_task / d >> SNR_param
```

The SNR gain from matched filtering is approximately B/d ~ 10^5-10^6.

**FM2 (Common Influence Contamination)**: Standard IF scoring I(z_test, z_train) = nabla_theta L(z_test)^T H^{-1} nabla_theta L(z_train) measures **total** influence, dominated by shared pre-training knowledge. The attribution decomposes as:

```
I(z_test, z_train) = I_task(z_test, z_train) + I_common(z_test, z_train)
```

where I_common >> I_task due to the dominance of pre-training information in parameter space. DDA's evidence: removing debias (which subtracts I_common) drops AUC by 55.2pp, confirming I_common is the dominant term.

Contrastive scoring acts as **differential detection**: by subtracting a reference signal (base-model attribution), common-mode interference is canceled:

```
I_contrastive(z_test, z_train) = I_{theta'}(z_test, z_train) - I_{theta_0}(z_test, z_train)
                                ~ I_task(z_test, z_train)
```

**Independence claim**: FM1 and FM2 are distinct failure modes with independent remedies. In signal processing, matched filtering (dimension reduction) and differential detection (common-mode rejection) are classically orthogonal operations with 70+ years of theoretical foundation. Their composition should yield approximately additive gains.

### 1.2 Empirical Support from AURA

AURA's CIFAR-10/ResNet-18 experiments (500 test points, full-model) provide direct evidence:

- **IF-RepSim anti-correlation** (tau = -0.467): The two spaces capture systematically different information, consistent with FM1/FM2 operating on different signal components.
- **Variance decomposition**: Jaccard@10 between EK-FAC and K-FAC has 77.5% residual variance after class/gradient-norm control, confirming Hessian sensitivity is per-sample.
- **Disagreement predictability** (AUROC = 0.691): The IF-RepSim disagreement is structured and predictable, not random noise.

### 1.3 Relationship to Hessian Approximation

FM1, FM2, and Hessian approximation error form **three complementary bottlenecks**:

| Bottleneck | Nature | Remedy | Evidence |
|------------|--------|--------|----------|
| Hessian approximation | Computational | Better Hessian (H > GGN >> EK-FAC >> K-FAC) | Better Hessians Matter (2509.23437) |
| FM1 (Signal Dilution) | Structural (dimensionality) | Representation-space operation | Li et al. (2409.19998), RepT (2510.02334) |
| FM2 (Common Contamination) | Structural (knowledge coupling) | Contrastive scoring | DDA (2410.01285), In-the-Wild |

At small scale (MLP, <1M params), FM1 is mild and FM2 is absent -> Hessian improvement dominates.
At LLM scale (>1B params, pre-trained), FM1 and FM2 become dominant -> Hessian improvement has diminishing marginal returns.

## 2. Unified Representation-Space TDA Family

### 2.1 Bilinear Framework

All five representation-space methods share the bilinear attribution form:

```
s(z_test, z_train) = phi(z_test)^T M psi(z_train)
```

where phi, psi are feature extractors and M is a (possibly identity) metric matrix.

| Method | phi | psi | M | Implicit Contrastive? |
|--------|-----|-----|---|----------------------|
| RepSim | h^(l) | h^(l) | I | No |
| RepT | [h^(l*); nabla_h L] | [h^(l*); nabla_h L] | I | Yes (nabla_h L) |
| In-the-Wild | h(x_chosen) - h(x_rejected) | h(x_data) - h(x_ref) | I | Yes (explicit diff) |
| Concept IF | J_l^T v | nabla_theta f | H^{-1} | No |
| AirRep | Enc(z) | Agg(Enc(z_i)) | learned | No |

### 2.2 Contrastive Scoring as Orthogonal Enhancement

Contrastive scoring can be applied to any base method:

```
s_contrastive(z_test, z_train) = s_{theta'}(z_test, z_train) - s_{theta_0}(z_test, z_train)
```

For parameter-space methods: DDA already implements this.
For representation-space methods: subtract base-model representation similarity from fine-tuned similarity.

This creates the 2x2 factor structure:

|  | Standard scoring | Contrastive scoring |
|--|-----------------|-------------------|
| **Parameter-space** | IF / TRAK (baseline) | DDA / Contrastive-TRAK |
| **Representation-space** | RepSim / RepT | Contrastive-RepSim / Contrastive-RepT |

## 3. 2x2 Factor Experiment Design

### 3.1 Design Rationale

The 2x2 design {parameter-space, representation-space} x {standard, contrastive} directly tests FM1/FM2 independence:

- **Main effect of representation-space** (row effect): Measures FM1 remediation gain.
- **Main effect of contrastive scoring** (column effect): Measures FM2 remediation gain.
- **Interaction term**: If small (<30% of min main effect), supports FM1/FM2 independence.

### 3.2 Methods in Each Cell

| Cell | Primary | Secondary |
|------|---------|-----------|
| Param + Standard | TRAK, IF (EK-FAC) | LoGra |
| Param + Contrastive | DDA | Contrastive-TRAK |
| Repr + Standard | RepSim, RepT | AirRep (if available) |
| Repr + Contrastive | Contrastive-RepSim | Contrastive-RepT |

### 3.3 Statistical Analysis

Per-sample attribution scores analyzed via:
- Per-sample permutation/bootstrap tests (not task-level ANOVA, which has insufficient power with only 3 tasks)
- Multi-seed protocol: 5 seeds with std and 95% CI reported
- Effect size: Cohen's d for pairwise comparisons
- Interaction magnitude: ratio of interaction effect to minimum main effect

Cross-reference: See experiment-design.md for complete experimental protocol.

## 4. 元数据

- **核心方法**: FM1/FM2 diagnostic framework + unified representation-space TDA family + 2x2 factor experiment
- **理论基础**: Signal processing (matched filtering + differential detection)
- **先验验证**: AURA CIFAR-10/ResNet-18 实验 (tau = -0.467, variance decomposition, disagreement AUROC = 0.691)
- **关键风险**: RepSim LDS performance on DATE-LM; Hessian scaling argument persuasiveness
