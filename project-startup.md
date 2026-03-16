# Project Startup: AURA (Adaptive Unified Robust Attribution)

> 本文档是研究项目的知识基础文档。
> 目标：将研究者的隐性洞察转化为 AI 可操作的结构化知识。
> 本文档将在整个项目周期中作为核心参考被反复调用。

**论文标题**：Beyond Point Estimates: Sensitivity-Aware Training Data Attribution

---

## 1. Research Seed (研究种子)

### 1.1 核心洞察 (Core Insight)

TDA 归因结果对 Hessian 近似质量高度敏感——不同的近似方法（H/GGN/EK-FAC/K-FAC/Identity）可以给出截然不同的 top-k 归因结果——但目前既无工具诊断这种敏感性，也无原则性方法利用诊断信号来改善归因质量。AURA 提出两层解决方案：(1) TRV 诊断工具量化每个 test point 的归因稳定性；(2) 基于 TRV 的自适应融合（RA-TDA）原则性地组合参数空间方法（IF）与表示空间方法（RepSim）。

### 1.2 洞察来源类型

- [x] 方法融合型：因果推断（Sensitivity Analysis + Doubly Robust Estimation）→ TDA
- [x] 问题驱动型：KB 54 篇论文系统暴露的评估可靠性危机

### 1.3 预期贡献 (Expected Contribution)

- **Phase 1（最小论文单元）**：TRV 作为 TDA 归因可信度的诊断工具——从"改善方法精度"到"量化结果可信度"的 meta-level 转换
- **Phase 2（条件推进）**：RA-TDA 自适应融合框架——基于稳定性信号原则性地组合两类 TDA 方法
- **工具贡献**：TRV 可直接服务于所有现有 TDA 方法的可靠性评估

### 1.4 初始假设清单

- **H1**：归因结果对 Hessian 近似的敏感性因 test point 而异，且这种差异是可预测的
- **H2**：IF 和 RepSim 的估计误差在统计上近似独立
- **H3**：低 TRV（不稳定）的 test point 上，RepSim 仍然提供有用的归因信号
- **H4**：Self-Influence (SI) 是 TRV 的有效快速代理
- **H5**：Hessian 近似误差和测试分布偏移是独立的鲁棒性维度

---

## 2. Source Materials (源材料)

### Source A: Hessian Approximation Hierarchy (2509.23437)
- **类型**: 论文
- **参考**: Hong et al., "Better Hessians Matter", arXiv 2025
- **核心要点**: 首次系统证明 LDS 严格遵循 Hessian 近似层级（H ≥ GGN ≫ Block-GGN ≫ EK-FAC ≫ K-FAC），最大误差源是 K-FAC→EK-FAC 特征值 mismatch
- **与本项目的关联**: Phase 1 基础设施——已实现完整的 Hessian 层级链代码（CIFAR-10/ResNet-18）；提供了"不同 Hessian 下 LDS 不同"的系统证据，但未量化 top-k 归因集合的变化

### Source B: Natural W-TRAK (2512.09103)
- **类型**: 论文
- **参考**: Li et al., "Natural Geometry of Robust Data Attribution", arXiv 2024
- **核心要点**: SI(z) = φ(z)^T Q^{-1} φ(z) = 归因稳定性的 Lipschitz 常数（Theorem 5.4）；谱放大 κ≈2.71×10⁵ 导致 Euclidean 认证 0%，Natural Wasserstein 达 68.7%；H-NW2 张力：Q^{-1} 可能降权语义方向
- **与本项目的关联**: Phase 0 的 SI/κ 诊断工具直接来自此论文；H-NW2 张力为 Phase 2 提供独立理论动机（RepSim 补回被 Q^{-1} 降权的语义方向）

### Source C: IF Failure on LLM (2409.19998)
- **类型**: 论文
- **参考**: Li et al., "Do Influence Functions Work on Large Language Models?", EMNLP 2025
- **核心要点**: IF 在 LoRA + 低信噪比下完全失效（0-7%），RepSim 96-100%
- **与本项目的关联**: Phase 2 的 IF 失效测试场景

### Source D: IF-RepSim Low Correlation (2602.14869)
- **类型**: 论文
- **参考**: Kowal et al., "Concept Influence", FAR.AI 2026
- **核心要点**: 向量类与梯度类方法的相关性仅 0.37-0.45——捕获不同信息
- **与本项目的关联**: 融合的前提条件——两类方法确实互补

### Source E: Doubly Robust Estimation (Chernozhukov 2018)
- **类型**: 理论框架
- **参考**: Chernozhukov et al., "Double/Debiased Machine Learning", 1608.00060
- **核心要点**: DR 估计器在两个组件中至少一个正确指定时一致；Neyman orthogonality 保证 cross-fitting 有效
- **与本项目的关联**: Phase 2 RA-TDA 的理论灵感来源（方法融合型种子的核心）

### Source F: Distributional TDA (2506.12965)
- **类型**: 论文
- **参考**: Mlodozeniec et al., "Distributional Training Data Attribution", NeurIPS 2025 Spotlight
- **核心要点**: IF 是"secretly distributional"，无需凸性假设
- **与本项目的关联**: 为 Phase 2 的 estimand 统一问题提供可能的理论框架

---

## 3. Knowledge Synthesis (知识综合)

### 3.1 源材料之间的关系

Source A (Hessian 层级) 和 Source B (Natural W-TRAK) 共同建立了"归因不稳定性是真实存在的"事实基础。Source C (IF 失效) 和 Source D (低相关性) 共同证明了"两类方法捕获不同信息，各有失效场景"。Source E (DR 框架) 提供了"如何原则性地组合两个各有缺陷的估计器"的理论工具。Source F (d-TDA) 提供了"IF 和 RepSim 可能在什么 estimand 下统一"的理论方向。

### 3.2 Gap Analysis (差距分析)

- Source A 证明了 Hessian 近似质量影响 LDS，但没有量化 top-k 归因集合在不同近似下的变化——这是 TRV 的直接空白
- Source B 做了"被动诊断"（认证区间），但没有"主动利用"稳定性信号改善归因——这是 RA-TDA 的直接空白
- TDA 社区不读因果推断文献，Source E 的 DR 框架从未被应用到 TDA

### 3.3 技术可行性初步判断

Phase 1 (TRV) 技术路径清晰——Source A 的代码已开源，主要工作是"在现有基础设施上增加 Jaccard@k 计算"。Phase 2 (RA-TDA) 的可行性中等——理论上 DR 框架成熟，但 H2（误差独立性）是核心不确定性。

---

## 3.4 多 Agent 多维压力测试 (Multi-Agent Stress Test)

### 核心假设清单

| # | 假设 | 来源 | 支撑强度 |
|---|------|------|---------|
| H1 | 归因敏感性因 test point 而异且可预测 | 研究者推断 + 2512.09103 SI 理论 | 中 |
| H2 | IF 和 RepSim 误差近似独立 | 研究者推断 | 弱 |
| H3 | 低 TRV 点上 RepSim 仍有用 | 2409.19998 实验 | 中 |
| H4 | SI 是 TRV 有效代理 | 理论推断 | 弱 |
| H5 | Hessian 近似误差与分布偏移独立 | 研究者推断 | 中 |

### 视角摘要

> 完整输出保存在 `phase-outcomes/debate/`。

#### 创新者（Innovator）
- **最强继续理由**：Meta-level 问题重定义（从 method improvement 到 reliability diagnosis）是真正的新空间
- **最危险失败点**：TRV 计算成本太高导致无实用性；DR 叙事依赖弱假设 H2
- **建议**：`Go with focus` — 聚焦 Phase 1，Phase 2 条件推进

#### 务实者（Pragmatist）
- **最强继续理由**：Phase 1 基础设施已就绪（2509.23437 代码），3-4 周可出结果
- **最危险失败点**：LLM 规模 IF 计算是工程深水区；Phase 2 需 3-4 个月
- **建议**：`Go with focus` — Phase 1 先行，Phase 2 条件推进；LLM 降级到 GPT-2

#### 理论家（Theorist）
- **最强继续理由**：TRV 在鲁棒统计学中有 breakdown point 的精确对应物
- **最危险失败点**：DR 类比缺少共同 estimand；lambda(TRV) 无理论指导可能退化为调参
- **建议**：`Go with focus` — 需在 C 阶段建立 estimand 统一或修改叙事为 "adaptive ensemble"

#### 反对者（Contrarian）
- **最强继续理由**：归因稳定性是真实的、未被量化的问题
- **最危险失败点**：H2 "值不相关 ≠ 误差不相关"；W-TRAK + naive ensemble 可能已经足够
- **建议**：`Go with focus` — Phase 1 有独立价值，但 Phase 2 必须超越 naive ensemble

#### 跨学科者（Interdisciplinary）
- **最强继续理由**：RA-TDA 与 AIPW/DML 结构同构（深层类比）；TRV 可形式化为 Hampel gross error sensitivity
- **最危险失败点**：DR 要求至少一个组件足够好，但 AURA 前提是两个都可能差
- **建议**：`Go with focus` — 采纳 Hampel 形式化 + Neyman orthogonality 检验

#### 实验主义者（Empiricist）
- **最强继续理由**：Pilot 仅需 0.5-1 GPU-hour 即可验证核心假设 H1
- **最危险失败点**：TRV 分布集中（>80% 同级）；LLM 上 TRV ground truth 不可获取
- **最小正面信号**：TRV 分布至少 3 个等级各占 >10%
- **否证条件**：相邻 Hessian 等级 avg Jaccard@10 > 0.85 → Hessian 近似对归因影响不够大
- **建议**：`Go with focus` — Pilot 增强（3-seed stability + OOD confound）

---

### 综合判定

**判定**：`Go with focus`（方向修正）

**值得启动的核心理由**：
- 填补 TDA meta-level 空白——从"改善方法精度"到"量化结果可信度"
- Phase 1 (TRV) 有独立发表价值，不受 Phase 2 风险影响
- 跨领域迁移（CI → TDA）提供差异化叙事，无直接竞争工作

**进入 C 前必须处理的优先问题**：
- TRV 的操作定义固化（k 值、Jaccard 阈值），不允许 post-hoc 调整
- W-TRAK + naive ensemble 作为 Phase 2 的必要 baseline
- Pilot 增强：3-seed stability + 20 OOD points + LLM 降级到 GPT-2

**进入 C 时带着的已知风险**：
- H2（误差独立性）弱——4/6 视角一致指向此风险
- TRV 变异度不足可能导致自适应退化为常数权重
- "稳定 ≠ 正确"悖论——TRV 高只保证"一致地"归因，不保证归因正确

**仍未消解的真实分歧**：
- Theorist 要求 DR 理论完备 vs Empiricist 认为经验先行理论后补
- Innovator 的 by-design 升维 vs 当前 post-hoc 诊断范式

---

## 4. Research Direction (研究方向)

### 4.1 核心研究问题 (Core Research Questions)

- **Q1**: TDA 归因结果在不同 Hessian 近似下的稳定性是否因 test point 而异，且可用廉价特征预测？
  - 关键否证线索：如果 >80% test point 有相同 TRV 等级，方向需收缩

- **Q2**: 基于稳定性信号的 IF+RepSim 自适应融合是否显著优于固定权重融合和单一方法？
  - 关键否证线索：如果 RA-TDA vs naive ensemble 的 LDS 提升 <2% absolute（Cohen's d <0.5），Phase 2 不成立

### 4.2 拟议方法概述

- **Phase 0**：Spectral Diagnosis（κ + SI，来自 2512.09103）
- **Phase 1**：TDA Robustness Value (TRV) = Jaccard@k 在 Hessian 层级链下的稳定性
- **Phase 2**：Residual Augmented TDA (RA-TDA) = RepSim + λ(TRV) × IF_residual
- Phase 1 和 Phase 2 显式解耦——Phase 1 是最小论文单元

### 4.3 候选攻击角度

- **攻击角度 A（核心）**：TRV 作为 TDA reliability diagnostic——量化"什么时候不该信任归因结果"，填补 meta-level 空白。以 2509.23437 的 Hessian 层级链为基础设施，TRV 作为新指标。
- **攻击角度 B（条件推进）**：RA-TDA 自适应融合——在 TRV 有差异化信息量的前提下，利用 DR 原理融合 IF 和 RepSim。需要 TRV variability + 显著超越 naive ensemble。

### 4.4 关键技术挑战

- TRV 需要在多个 Hessian 近似下重复计算归因——计算成本是否可接受？SI 代理能否降低成本？
- DR 融合的 estimand 统一问题——IF 和 RepSim 没有共同的估计目标
- LLM 规模下精确 Hessian 不可计算——TRV ground truth 断裂

### 4.5 已知风险与未解决的质疑

- **[Contrarian]** H2 误差独立性无实证——C 中需设计误差相关性的操作化指标
- **[Contrarian]** W-TRAK + naive ensemble 可能已足够——C 中需系统梳理竞争 baseline
- **[Theorist]** DR 叙事可能需修正为 "adaptive ensemble with diagnostic guidance"
- **[Empiricist]** Training seed confound——C 中 pilot 必须包含 3-seed stability 检验

---

## 5. Metadata

- **创建日期**: 2026-03-16
- **研究者**: Jinxu Lin
- **目标会议/期刊**: NeurIPS 2026 / ICML 2026
- **预计时间线**: Phase 1 (3-4 weeks) + Phase 2 (conditional, 2-3 months)
- **多 Agent 检验结论**：`Go with focus`（方向修正：Phase 1-2 解耦 + naive ensemble baseline + pilot 增强）
