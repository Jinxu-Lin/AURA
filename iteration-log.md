# Iteration Log

> 本文档记录所有版本变更的结构化历史。**每次版本号变化时追加一条记录**。
> **倒序排列**（最新在最上方），方便 Agent 快速读取最近的迭代上下文。
> 排除方向必须记录——这是防止 Agent 重复已失败方向的核心数据。
> 关键洞察必须记录——失败经验是最有价值的知识资产。

---

## [1.2] -- 2026-03-25 -- Direction Pivot: Geometric Incommensurability of Knowledge Operations

- **触发**: formalize_review round-1 returned REVISE with 3 structural issues (Direction A/B inconsistency, DATE-LM probe required, RIF/BIF gaps). Combined with user strategic input identifying TECA's negative result as the core NeurIPS contribution.
- **诊断层次**: Strategic pivot — both Direction A (BSS on CIFAR-10) and Direction B (FM1/FM2 on DATE-LM) abandoned in favor of TECA geometric incommensurability.
- **变更文档**: problem-statement.md (1.0 → 1.2)
- **新方向**: Geometric Incommensurability of Knowledge Operations
  - Core finding: TECS ~ 0 (Cohen's d = 0.05), editing subspace ~40D, attribution subspace ~1D, random-level misalignment
  - Contribution: The negative result resolves Hase et al. (2023) localization-editing disconnect at parameter level
  - Evidence: TECA experiments already completed (GPT-2-XL, 100 CounterFact facts, 5 null baselines)
- **排除方向**:
  - **FM1/FM2 2x2 ablation on DATE-LM** (Direction B) -- 155 GPU-hours needed vs ~14 remaining; RepSim LDS = 0.074 in AURA data (uncertain LLM-scale performance); formalize review flagged 3 structural issues; no LLM-scale evidence for representation-space competitiveness
  - **BSS per-test-point diagnostic** (Direction A) -- Different problem scope (per-point diagnosis vs knowledge geometry); BSS-gradient norm correlation rho = 0.906 raises degeneracy risk; CIFAR-10 only, no scalability evidence
  - **TECS as TDA validation metric** -- TECS ~ 0 definitively shows editing is NOT a TDA validation channel, but this negative result IS the contribution
- **关键洞察**:
  - The 40D vs 1D subspace asymmetry is a standalone finding: ROME's constrained optimization explores a fundamentally richer parameter manifold than loss-gradient-based attribution
  - Cross-projection asymmetry (G captures 17.3% of D; D captures 1.0% of G) suggests hierarchical structure
  - MEMIT matched-layer alignment is high but misleading (shared loss function artifact, not knowledge geometry)
  - Formalize review's DATE-LM probe concern is moot for this direction — evidence comes from TECA parameter-space geometry, not benchmark performance
- **遗留问题**: Cross-model universality (GPT-J, Pythia); C^{-1} whitening ablation (H6); toy model validation; RIF effect on attribution subspace

---

## [1.1] -- 2026-03-25 -- FR-Revise: Direction A (BSS) committed, Direction B abandoned

- **触发**: formalize_review round-1 REVISE verdict
- **诊断层次**: formalize-level revision
- **变更文档**: problem-statement.md (1.0 → 1.1)
- **内容**: Chose Direction A (BSS diagnostic on CIFAR-10), abandoned Direction B (FM1/FM2 on DATE-LM) due to infeasible compute. Addressed RIF/BIF engagement, BSS-gradient norm degeneracy, scale limitation honesty, MRC soft combining.
- **状态**: Superseded by v1.2 pivot to geometric incommensurability.

---

## [1.0] -- 2026-03-25 -- Assimilation from CRA_old + AURA + CRA

- **触发**: Project assimilation -- merging three related TDA projects into unified Noesis v3 framework
- **诊断层次**: N/A (initial version)
- **变更文档**: problem-statement.md (1.0), method-design.md (1.0), experiment-design.md (1.0), contribution.md
- **来源项目**:
  - **CRA_old** (Noesis V1): FM1/FM2 diagnostic framework, signal-processing theory, 2x2 ablation design. Strategic review PASSED, no experiments executed.
  - **AURA** (Sibyl system): Spectral sensitivity analysis, variance decomposition, IF-RepSim disagreement analysis. Significant experimental results on CIFAR-10/ResNet-18 (Phases 0-2b completed).
  - **CRA** (Sibyl system): Cross-task VLA influence -- different domain (robotics), kept as methodological inspiration only.
- **合并内容**:
  - Theoretical framework: FM1 (Signal Dilution) + FM2 (Common Influence Contamination) from CRA_old
  - Empirical evidence: AURA's Phase 0-2b results (tau = -0.467, variance decomposition, disagreement AUROC = 0.691) reinterpreted as supporting evidence for FM1/FM2
  - Unified representation-space TDA family (phi^T psi bilinear form) from CRA_old
  - 2x2 ablation experiment design from CRA_old, enriched with AURA's experimental protocol
- **排除方向**:
  - **VLA cross-task influence** (CRA) -- different domain (robotics vs NLP/LLM), methodological inspiration only, kept as future work
  - **BSS/TRV per-point routing** (AURA Phase 2a/3) -- AURA showed cross-seed TRV rho ~ 0 and routing tax concerns; not part of core contribution but retained as optional C4
  - **Fixed-IF** (CRA_old angle B) -- design space too large, ASTRA overlap; kept as potential extension
- **从合并中获得的关键洞察**:
  - AURA's IF-RepSim anti-correlation (tau = -0.467) provides the strongest empirical evidence for FM1/FM2 independence -- stronger than any theoretical argument alone
  - AURA's variance decomposition (gradient norm explains 40.5% of LDS) directly validates FM1 mechanism
  - The "three complementary bottlenecks" framing (Hessian + FM1 + FM2) resolves the tension between Better Hessians Matter and the representation-space superiority claims
- **实验数据保留**: AURA iter_001/ directory preserved intact for reference
