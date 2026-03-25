---
version: "1.0"
last_updated: "2026-03-25"
---

> [ASSIMILATED: generated from CRA_old/research/contribution.md + AURA experimental findings]

# Contribution Tracker: AURA (TDA Diagnostic Framework)

> 本文档跨阶段维护，记录项目贡献的演化过程。
> 贡献必须足够支撑发表 -- Review Phase 将审查本文档判断发表价值。

---

## 贡献列表

### Assimilation 初始化 (CRA_old + AURA fusion)

| # | 贡献 | 类型 | 来源 | 状态 |
|---|------|------|------|------|
| C0 | **FM1/FM2 Diagnostic Framework**: Identify two independent signal-processing defects in parameter-space TDA -- FM1 (Signal Dilution: p >> d_h causes SNR collapse) and FM2 (Common Influence Contamination: pre-training knowledge dominates attribution scores) -- and argue they are complementary to (not replacements for) Hessian approximation error. | Gap 识别 / 问题定义 | CRA_old problem-statement + signal processing theory | 概念完成，待 DATE-LM 验证 |
| C1 | **Unified Representation-Space TDA Family**: Recognize 5 independently proposed methods (RepSim, RepT, In-the-Wild, Concept IF, AirRep) as instances of a single bilinear framework phi^T M psi, explaining their effectiveness as FM1 remediation. | 理论统一 | CRA_old + cross-paper pattern recognition | 概念完成 |
| C2 | **First Systematic DATE-LM Evaluation of Representation Methods**: Benchmark RepSim, RepT (and optionally others) against TRAK, IF, DDA on the full DATE-LM benchmark (3 tasks), filling a recognized evaluation gap. | 实验贡献 | CRA_old design + DATE-LM gap | 待实验 |
| C3 | **2x2 Ablation Verifying FM1/FM2 Independence**: {parameter-space, representation-space} x {standard, contrastive scoring} factor experiment quantifying each remediation's effect and their interaction, testing the "two independent failure modes" thesis. | 实验方法论 | CRA_old + signal processing theory | 待实验 |
| C4 | **(Optional) Sensitivity-Aware TDA**: AURA's variance decomposition and disagreement analysis as supplementary evidence for FM1/FM2 framework -- demonstrating that IF and RepSim capture systematically different information (tau = -0.467) with predictable per-sample complementarity. | 实验发现 | AURA Phase 0-2b experiments | 已有数据 (CIFAR-10/ResNet-18) |

---

## 贡献评估

### 整体发表价值评估

| 评估维度 | 评级 | 论据 |
|---------|------|------|
| Novelty | 中-高 | No prior work unifies 5 representation-space methods or diagnoses FM1/FM2 as independent failure modes. Signal-processing framing is novel in TDA. |
| Significance | 高 | Fills a recognized evaluation gap (representation methods on DATE-LM). Provides practitioner guidance for method selection. |
| 与目标会议/期刊的匹配度 | 高 | NeurIPS 2026 values systematic empirical studies with theoretical framing. DATE-LM is a NeurIPS benchmark. |

### 贡献与论文章节的映射

| 贡献 | Introduction 中的 claim | Experiments 中的验证 |
|------|------------------------|---------------------|
| C0 | "Parameter-space TDA fails at LLM scale due to two independent signal-processing defects" | 2x2 ablation main effects + Hessian scaling analysis |
| C1 | "Five representation-space methods form a unified family sharing bilinear structure" | Method comparison + correlation matrix |
| C2 | "First systematic evaluation of representation methods on DATE-LM" | Full LDS table across 3 tasks |
| C3 | "FM1 and FM2 are independent: their remediation gains are approximately additive" | 2x2 interaction analysis |
| C4 | "AURA's experiments provide small-scale validation of the framework" | Supplementary: CIFAR-10 results |

---

## Metadata
- **目标会议/期刊**: NeurIPS 2026
- **上次更新**: Assimilation (2026-03-25)
- **当前状态**: 贡献基本够 (C0-C3 collectively form a strong empirical+theoretical contribution; C4 is bonus)
