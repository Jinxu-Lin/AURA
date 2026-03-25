---
version: "1.0"
status: "assimilated"
decision: "pass"
created: "2026-03-25"
last_modified: "2026-03-25"
---

# Project: AURA (TDA Diagnostic Framework)

## 1. Overview

### 1.1 Topic

Signal-processing diagnostic framework for understanding Training Data Attribution (TDA) failure modes in LLMs, unifying five representation-space methods and validating two independent failure modes (FM1: Signal Dilution, FM2: Common Influence Contamination) through systematic 2x2 ablation on DATE-LM.

### 1.2 Initial Idea

Parameter-space TDA methods (IF, TRAK) systematically fail on LLM tasks, while representation-space methods (RepSim, RepT) succeed -- but no one has explained why through a unified lens. We propose that two independent signal-processing defects cause parameter-space failure: FM1 (signal dilution due to extreme dimensionality, where per-sample gradients become near-orthogonal in 10^9-dimensional space) and FM2 (common influence contamination, where pre-training knowledge dominates attribution scores). Representation-space methods implicitly remedy FM1 (dimension reduction as matched filtering), while contrastive scoring remedies FM2 (differential detection canceling common-mode interference).

The key insight, drawn from 70+ years of signal processing theory, is that these two remedies are orthogonal -- their gains should be approximately additive. A 2x2 factor experiment {parameter-space, representation-space} x {standard, contrastive scoring} on DATE-LM can directly test this thesis.

This project fuses CRA_old's theoretical framework (Noesis V1, FM1/FM2 diagnosis, strategic review passed) with AURA's experimental data (Sibyl system, CIFAR-10/ResNet-18 results showing IF-RepSim anti-correlation tau = -0.467, variance decomposition, disagreement predictability).

### 1.3 Baseline Papers

| # | Paper | Link | Relevance |
|---|-------|------|-----------|
| 1 | Li et al. 2025 "Do IF Work on LLMs?" | 2409.19998 | FM1 core evidence: RepSim 96-100% vs IF 0-7% |
| 2 | DDA 2024 "Enhancing TDA for LLMs" | 2410.01285 | FM2 core evidence: debias drops AUC by 55.2pp |
| 3 | RepT 2025 "Representation Gradient Tracing" | 2510.02334 | Repr-space method: P@10=0.97-1.00 vs LESS 0.59-0.73 |
| 4 | DATE-LM 2025 "LLM TDA Benchmark" | 2507.09424 | Primary evaluation benchmark (3 tasks) |
| 5 | Better Hessians Matter 2025 | 2509.23437 | Core tension: Hessian improvement vs FM1/FM2 |
| 6 | In-the-Wild 2026 "Concept Influence" | 2602.14869 | Repr-space + contrastive; IF-RepSim corr 0.37-0.45 |
| 7 | AirRep 2025 NeurIPS Spotlight | 2505.18513 | Learned repr-space TDA, ~100x faster |
| 8 | TRAK 2023 ICML | Park et al. | Param-space baseline, random projection |
| 9 | ASTRA 2025 | 2507.14740 | EKFAC-Preconditioned Neumann IF |
| 10 | Natural W-TRAK 2024 | 2512.09103 | SI/kappa diagnostics, spectral amplification theory |

### 1.4 Available Resources

- **GPU**: 4x RTX 4090 (xuchang0 server, SSH MCP) + 4x A6000 (jinxulin server)
- **Timeline / DDL**: NeurIPS 2026 (submission ~May 2026)
- **Existing Assets**:
  - AURA CIFAR-10/ResNet-18 experimental data (Phase 0-2b, 500 test points, full-model)
  - AURA codebase: probe experiment, dattri EK-FAC, TRAK, RepSim implementations
  - CRA_old theoretical framework (problem-statement, contribution, strategic review)
  - DATE-LM is open-source with leaderboard

---

## 2. Problem & Approach

### 2.1 Baseline Analysis

#### 它们解决了什么
Li et al. diagnosed iHVP degeneracy under LoRA. DDA introduced contrastive scoring for hallucination tracing. RepT introduced representation gradients with auto layer selection. DATE-LM provided a standard benchmark. Better Hessians Matter showed Hessian quality monotonically affects attribution.

#### 它们没解决什么
No work unifies the 5 representation-space methods. No work explains why representation methods work through a general lens (beyond LoRA-specific analysis). No representation method has been evaluated on the full DATE-LM benchmark. The tension between "better Hessians help" and "representation space is better" is unresolved.

#### 为什么没解决
Parameter-space and representation-space communities are disconnected. The signal-processing perspective (matched filtering + differential detection) has never been applied to TDA. Each representation method was developed for a different task in a different setting, obscuring their shared bilinear structure.

### 2.2 Problem Definition

- **问题一句话**: Five representation-space TDA methods independently outperform parameter-space methods in their niches, but lack unified evaluation, theoretical explanation, and practitioner guidance.
- **真实性论证**: AURA data shows IF-RepSim tau = -0.467 (anti-correlated), confirmed across 500 test points with full-model Hessian. DATE-LM leaderboard has no representation-space entries.
- **重要性论证**: 50+ TDA papers in 12 months; representation-space methods growing rapidly; practitioners need guidance.
- **问题价值层次**: "做了但有根本缺陷" (parameter-space TDA fails structurally at LLM scale) + "条件变了" (LLM scale makes FM1/FM2 dominant)

### 2.3 Root Cause Analysis

Layer 1 (symptom): Parameter-space IF fails on LLMs.
Layer 2 (FM1): Signal dilution -- p >> d causes SNR collapse.
Layer 3 (FM2): Common influence contamination -- pre-training dominates.
Layer 4 (structural): FM1/FM2 are intrinsic to parameter space (no natural task/general decomposition).

### 2.4 Proposed Approach

2x2 diagnostic framework: {parameter-space, representation-space} x {standard, contrastive scoring} on DATE-LM. Unified bilinear representation of 5 methods. Signal-processing theory provides orthogonality argument. AURA data provides small-scale validation.

### 2.5 Core Assumptions

| # | 假设 | 类型 | 来源 | 支撑强度 | 若为假会怎样 |
|---|------|------|------|---------|------------|
| H1 | FM1 (signal dilution) is the primary cause of param-space IF failure, not Hessian error | Structural | Li et al. + AURA variance decomposition | 中强 | Framework loses theoretical basis |
| H2 | Contrastive scoring is universally effective (not just for hallucination tracing) | Generality | DDA + In-the-Wild (only 2 task types) | 弱-中 | "Universal enhancement" claim downgraded |
| H3 | FM1 and FM2 remediation gains are approximately additive | Independence | Signal processing theory + AURA tau=-0.467 | 中 | 2x2 narrative weakened |
| H4 | Repr-space methods are competitive with param-space on DATE-LM LDS | Empirical | None (untested) | 无 | Core evaluation claim weakened |
| H5 | FM1 is not a LoRA artifact | Generality | Indirect | 弱 | FM1 scope must be restricted |

---

## 3. Validation Strategy

### 3.1 Idea Type Classification

Mixed: New theoretical perspective (FM1/FM2 diagnostic) + systematic empirical study (DATE-LM evaluation).

### 3.2 Core Hypothesis

H4 (repr-space competitive on DATE-LM) is the gating hypothesis. If RepSim < TRAK - 5pp on all DATE-LM tasks, the systematic superiority narrative fails.

### 3.3 Probe Experiment Design

RepSim vs TRAK on Pythia-1B, DATE-LM toxicity filtering task. 2.5 GPU-days.
Prior evidence: AURA CIFAR-10/ResNet-18 data (Phase 0-2b).

### 3.4 Pass / Fail Criteria

| 结果 | 条件 | 后续动作 |
|------|------|---------|
| Pass | RepSim LDS >= TRAK - 5pp on >= 1 DATE-LM task | Full 2x2 ablation |
| Marginal | RepSim < TRAK - 5pp on toxicity but competitive on data selection | Reframe as task-dependent study |
| Fail | RepSim < TRAK - 5pp on all tested tasks | Re-evaluate direction |

### 3.5 Time Budget & Resources

Probe: 2.5 GPU-days on RTX 4090. Full experiments: ~155 GPU-hours over 3-4 weeks.

### 3.6 Failure Diagnosis Plan

| 失败模式 | 特征 | 意味着什么 | 后续动作 |
|---------|------|----------|---------|
| RepSim fails on LDS universally | Low scores on all tasks | Repr methods don't capture counterfactual influence | Reframe as "different strengths" paper |
| RepSim fails due to layer selection | Only last layer tested | Wrong feature extraction | Try RepT (auto layer selection) |
| Interaction term too large | >30% of min main effect | FM1/FM2 entangled | Reframe as "partially overlapping" |
| DDA not applicable to data selection | No natural contrastive reference | Incomplete 2x2 | Design synthetic reference |

---

## 4. Review

### 4.1 Review History

| Round | Date | Decision | Key Changes |
|-------|------|----------|-------------|
| CRA_old RS | 2026-03-16 | Pass (with revisions) | Hessian scaling argument needed; MAGIC baseline required; statistical power concern |

### 4.2 Latest Assessment Summary

CRA_old strategic review (4 debaters: Contrarian, Comparativist, Pragmatist, Interdisciplinary) passed with "Go with focus" verdict. Key concerns: FM1 LoRA artifact risk (30-40%), Hessian scaling argument relies on inference not direct experiment, statistical power of 2x2 with only 3 tasks.

### 4.3 Decision

- **Decision**: Pass (assimilated, ready for research module)
- **Rationale**: CRA_old passed strategic review; AURA provides experimental validation; combined project is stronger than either alone.
- **Key Risks**: RepSim LDS on DATE-LM; Hessian scaling argument; FM1 LoRA artifact; concurrent work.
- **Unresolved Disputes**: Fixed-IF as extension vs scope creep; statistical analysis approach (per-sample vs task-level).

### 4.4 Conditions for Next Module

- Execute DATE-LM probe (RepSim vs TRAK on toxicity filtering)
- Address Hessian scaling argument in formalize phase
- Include MAGIC + DDA as mandatory baselines

<!-- 完整辩论记录: CRA_old/Reviews/ -->
