---
version: "1.0"
created: "2026-03-25"
last_modified: "2026-03-25"
entry_mode: "assimilated"
iteration_major: 1
iteration_minor: 0
---

> [ASSIMILATED: generated from CRA_old/research/problem-statement.md + AURA experimental data]

# Problem Statement

## 1. Gap 定义

### 1.1 现有方法概览

Training Data Attribution (TDA) for LLMs operates in two fundamentally different spaces, each with distinct failure modes:

**Parameter-space methods** (IF, TRAK, LoGra, LESS, DDA, ASTRA, SOURCE, TrackStar) compute influence scores via gradients with respect to model weights theta in R^B (B ~ billions). These methods share a common computational backbone: per-sample parameter gradients, optionally processed through an inverse Hessian approximation. The best recent advances focus on improving Hessian quality (ASTRA, Better Hessians Matter), gradient compression (LoGra, GraSS, LoRIF), or multi-checkpoint aggregation (SOURCE, DDA).

**Representation-space methods** (RepSim, RepT, In-the-Wild, Concept Influence, AirRep) compute influence scores using internal activations h^(l) and/or their gradients nabla_h L at specific hidden layers. These methods share a bilinear structure: the attribution score is fundamentally of the form phi(z_test)^T * psi(z_train), where phi and psi are feature extractors operating in the model's activation space rather than weight space:

| Method | phi (test encoding) | psi (train encoding) | Space dim |
|--------|-------------------|---------------------|-----------|
| RepSim | h^(l)(z_test) | h^(l)(z_train) | d ~ 4096 |
| RepT | concat[h^(l*), nabla_h L]_test | concat[h^(l*), nabla_h L]_train | 2d ~ 8192 |
| In-the-Wild | v_behavior (activation diff) | v_data (activation diff) | d ~ 4096 |
| Concept IF | J_l^T v (concept gradient) | nabla_theta f(z_train) | p (params) |
| AirRep | Enc(z_test) | Agg(Enc(z_i)) | d_enc ~ 384 |

**Contrastive scoring** is an orthogonal enhancement: DDA's debias component subtracts base-model IF from fine-tuned IF (IS_DDA = IS_{theta'} - IS_{theta_0}), effectively removing "common-mode" influence from pre-training knowledge. In-the-Wild uses activation differences (chosen - rejected) as a natural contrastive structure. RepT's representation gradient nabla_h L can be interpreted as an implicit contrastive signal.

### 1.2 Gap 陈述

**One-sentence Gap**: Five independently proposed representation-space TDA methods (RepSim, RepT, In-the-Wild, Concept Influence, AirRep) have each demonstrated superiority over parameter-space methods in their respective task niches, yet no work has (a) recognized these as a coherent methodological family, (b) explained *why* they work through a unified diagnostic lens, or (c) evaluated them on a common benchmark against the same parameter-space baselines -- leaving practitioners without principled guidance on when and why to use representation-space TDA.

**Detailed analysis**:

The current TDA literature exhibits a structural fragmentation: parameter-space and representation-space methods are developed by disconnected research communities with different evaluation protocols, different tasks, and different baselines. This fragmentation has three concrete consequences:

1. **No comparative evaluation exists**. RepT reports P@10 on controlled harmful-data experiments; AirRep reports LDS on DATE-LM (but only for data selection); DDA reports AUC on hallucination tracing; Concept IF reports evil-score reduction on misalignment benchmarks. There is no single evaluation that compares RepSim, RepT, TRAK, DDA on the same tasks with the same models. Critically, **no representation-space method has been evaluated on the full DATE-LM benchmark**.

2. **The "why representation space works" question is unanswered**. Li et al. (2409.19998) diagnosed one failure mode (LoRA low-rank Hessian -> iHVP degeneracy), but this explains only *why parameter-space methods fail in specific LoRA settings*, not why representation-space methods succeed across diverse settings.

3. **The relationship between Hessian improvement and representation-space superiority is unresolved**. Better Hessians Matter (2509.23437) demonstrates that improving Hessian approximation quality *consistently* improves attribution quality (LDS: H >= GGN >> EK-FAC >> K-FAC). This creates a direct tension: if better Hessians always help, why bother leaving parameter space?

**Empirical evidence from AURA experiments** (CIFAR-10/ResNet-18, full-model, 500 test points) directly supports this gap:
- IF and RepSim produce **systematically anti-correlated** rankings: mean Kendall tau = -0.467 (p < 1e-36), confirming they capture fundamentally different information.
- EK-FAC IF achieves mean LDS = 0.297 while RepSim achieves only 0.074, but RepSim outperforms IF on 29/500 points -- suggesting point-dependent complementarity.
- Disagreement predicts which method is better: AUROC = 0.691, partial rho = 0.266 after controlling for class and gradient norm (p < 1e-9) -- not a class proxy.

### 1.3 Root Cause 分析

**Root Cause Type**: Two independent signal-processing defects compound with (but are distinct from) Hessian approximation error.

**Layer 1 (surface symptom)**: Parameter-space IF performs poorly on LLM tasks (RepSim 96-100% vs IF 0-7% on harmful data identification; RepT P@10=0.97-1.00 vs LESS P@10=0.59-0.73).

**Layer 2 (FM1: Signal Dilution)**: Parameter-space gradients are computed in R^B (B ~ 10^9-10^10). Each training sample's gradient is approximately orthogonal to every other sample's gradient (Johnson-Lindenstrauss phenomenon). The task-relevant signal occupies a tiny subspace with extremely low SNR. AURA's variance decomposition confirms: gradient norm explains 40.5% of per-point LDS variance, demonstrating that gradient geometry fundamentally determines attribution quality.

**Layer 3 (FM2: Common Influence Contamination)**: Standard IF scoring measures **total** influence, dominated by shared pre-training knowledge. DDA's ablation: removing debias drops AUC by 55.2pp (93.49% -> 67.88%), while removing denoise only drops 8.71pp. AURA's finding that IF and RepSim are anti-correlated (tau = -0.467) directly supports FM2: the two methods capture systematically different aspects of influence.

**Layer 4 (structural root cause)**: FM1 and FM2 are *structurally* coupled to operating in parameter space. In parameter space, the model's entire knowledge is encoded in a single weight set with no natural decomposition separating "task-specific" from "general" influence. Representation space offers natural decomposition: layer-specific activations encode task-relevant semantics, and dimensionality is orders of magnitude lower (d ~ 4096 vs B ~ 10^9).

**Critically, FM1/FM2 are distinct from Hessian approximation error**. Better Hessians Matter experiments are on low-dimensional settings (Digits/MLP, <1M parameters) where FM1 is mild and FM2 is absent (no pre-training). At LLM scale, FM1 and FM2 become dominant bottlenecks.

AURA's Jaccard@10 variance decomposition: 77.5% residual per-point variation after controlling for class and gradient norm -- confirming Hessian sensitivity is a genuine per-sample phenomenon.

### 1.4 Gap 评价

| 维度 | 评价 | 论证 |
|------|------|------|
| **重要性** | **高** | TDA for LLMs is a high-activity area (50+ papers in 12 months). Five representation-space methods emerged independently, yet practitioners lack unified guidance. |
| **新颖性** | **中-高** | No prior work unifies these 5 methods or diagnoses FM1/FM2 as independent failure modes. Li et al. diagnoses iHVP degeneracy but not signal dilution. DDA identifies knowledge bias but not the connection to representation space. |
| **可解性** | **中** | Diagnostic framework can be argued from existing evidence + AURA data. DATE-LM evaluation is engineering-heavy but feasible. Key uncertainty: whether the 2x2 ablation produces clean results. |

### 1.5 Research Questions

**RQ1 (Systematic evaluation)**: How do representation-space TDA methods (RepSim, RepT) compare to parameter-space methods (TRAK, IF) on DATE-LM across all three tasks (data selection, toxicity filtering, factual attribution)?

- *Falsification*: RepSim < TRAK - 5pp on LDS across all three tasks.
- *Prediction*: Representation methods should excel on toxicity filtering (high FM2 bias), competitive on data selection, may struggle on factual attribution.

**RQ2 (Contrastive enhancement generality)**: Does contrastive scoring improve both parameter-space and representation-space methods additively?

- *Falsification*: Contrastive scoring improves representation methods by < 3pp on >= 2/3 tasks (FM2 already implicitly resolved).
- *Prediction*: Larger gains for parameter-space methods (suffer from both FM1 + FM2) than representation-space (partly address FM1).

**RQ3 (Additivity / Independence)**: Are FM1 and FM2 remediation gains approximately additive?

- *Falsification*: 2x2 interaction term exceeds 30% of minimum main effect.
- *Prediction*: Near-additivity (signal-processing orthogonality). AURA's IF-RepSim anti-correlation (tau = -0.467) supports this.

## 2. 攻击角度

### 2.1 候选攻击角度

| 角度 | 核心 idea | 优势 | 风险 |
|------|-----------|------|------|
| **A: Diagnostic + Systematic Benchmark** | FM1/FM2 diagnosis + 2x2 ablation on DATE-LM | Fills core gap; low engineering risk; AURA provides prior evidence | Ceiling may be poster-level |
| **B: A + Fixed-IF** | Parameter-space signal processing repair | Upgrades from analysis to method paper | Design space large; ASTRA overlap |
| **C: Sensitivity-Aware TDA** | AURA's routing approach with FM1/FM2 diagnostics | Leverages existing infrastructure | Routing tax may exceed gains |

### 2.2 选定攻击角度

**选定角度 A**: Diagnostic framework + systematic benchmark evaluation.

**核心 idea**: Propose a signal-processing diagnostic framework identifying two independent defects -- FM1 (Signal Dilution) and FM2 (Common Influence Contamination) -- and verify their independence through a 2x2 ablation {parameter-space, representation-space} x {standard scoring, contrastive scoring} on DATE-LM.

**为什么可能有效**: (1) Signal-processing perspective (matched filtering + differential detection) provides unified theory with 70+ years of orthogonality foundations. (2) AURA's evidence (tau = -0.467, variance decomposition) directly supports core claims. (3) DATE-LM evaluation gap for representation methods is a recognized community need.

### 2.3 攻击角度的局限性与风险

1. **天花板风险**: Pure diagnostic + benchmark may be empirical study level. Mitigation: AURA's prior evidence + clean ablation.
2. **RepSim 在 LDS 上可能不佳**: AURA shows IF dominates on LDS (0.297 vs 0.074), but RepSim wins on 5.8% of points.
3. **Hessian scaling 论证**: Must explain why Better Hessians Matter doesn't contradict the framework.
4. **FM1 LoRA artifact risk**: Need both LoRA and full fine-tuning on DATE-LM.
5. **Contrastive scoring construction**: Not trivial for all DATE-LM tasks.

## 3. 探针方案

### 3.1 Prior Empirical Evidence (AURA, CIFAR-10/ResNet-18)

See Section 1.2 empirical evidence and Codes/_Results/probe_result.md for full results.

Key supporting findings:
- IF-RepSim anti-correlation (tau = -0.467) supports FM1/FM2 independence
- Variance decomposition residuals (45.9-77.5%) confirm per-point structure
- Disagreement predictability (AUROC = 0.691) confirms genuine complementarity

### 3.2 DATE-LM Probe (Pending)

RepSim vs TRAK on Pythia-1B, toxicity filtering task. Total: 2.5 GPU-days.

Pass criteria: RepSim LDS >= TRAK LDS - 5pp on at least one task.

## 4. 元数据

- **Gap 来源**: 组合推导 (5 representation-space methods + DDA contrastive + Better Hessians Matter tension + AURA empirical data)
- **Root cause 类型**: 被忽视的维度 (signal processing perspective never applied to TDA failure analysis)
- **攻击角度来源**: 跨领域工具迁移 (signal processing matched filtering + differential detection)
- **关键不确定性**: RepSim on DATE-LM LDS; Hessian scaling argument; FM1 LoRA artifact
- **先验经验数据**: AURA CIFAR-10/ResNet-18 (Phase 0-2b, 500 test points, full-model)
