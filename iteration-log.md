# Iteration Log

> 本文档记录所有版本变更的结构化历史。**每次版本号变化时追加一条记录**。
> **倒序排列**（最新在最上方），方便 Agent 快速读取最近的迭代上下文。
> 排除方向必须记录——这是防止 Agent 重复已失败方向的核心数据。
> 关键洞察必须记录——失败经验是最有价值的知识资产。

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
