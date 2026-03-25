## [Pragmatist] 务实者视角

### 工程组件拆解

**现有 AURA/CIFAR-10 基础设施（已完成部分）**
- ✓ ResNet-18 训练 + EK-FAC/K-FAC IF 计算 — 已有 500 点数据，代码在 Sibyl 格式下运行
- ✓ RepSim/TRAK 计算 — 已完成 Phase 1，指标已产出
- ✓ 方差分解 + IF-RepSim 相关性分析 — Phase 1/2b 已确认 A1/A3
- △ BSS 计算管线 — Phase 2a pilot 已有 100 点单 seed 结果，但 5 seed 交叉验证尚未完成，且 BSS-gradient norm 相关性 ρ=0.906 是未解决的根本隐患
- △ Sibyl 格式代码 → v3 重构 — 现有代码以 iter_001/exp/code/ 组织，需要迁移到 Codes/core/ + Codes/experiments/ 结构

**DATE-LM 全新工程（未开始部分）**
- ✗ Pythia-1B fine-tuning 管线 — 需从零搭建：数据加载、LoRA + full FT 两种配置、5 seeds，预计 20 GPU-hours
- ✗ DATE-LM 评测框架集成 — 需读懂 DATE-LM 代码库、适配 3 个 task 的评测接口、LDS 计算
- ✗ TRAK on Pythia-1B — trak 库对 LLM 的支持成熟度未知，10 checkpoint 聚合需大量中间存储
- ✗ EK-FAC IF on Pythia-1B — dattri IFAttributorEKFAC 是否支持 Pythia 架构？1B 参数的 Kronecker 分解内存需求未知
- ✗ DDA 实现 — 需要 base model + fine-tuned model 两套推理，contrastive scoring 构造对 data selection task 不自然
- ✗ Contrastive-RepSim / Contrastive-RepT — 新方法，无现成实现，需自行开发 + 调试
- ✗ RepT 实现 — 需 auto layer selection + activation gradient 拼接，无公开 LLM 级实现
- △ 统计分析管线 — per-sample permutation test + bootstrap CI + interaction analysis，工程量中等但需严谨

### 最小 Pilot 设计

**实验内容**：在 DATE-LM toxicity filtering 单任务上，用 Pythia-1B + LoRA fine-tuning（单 seed），比较 RepSim vs TRAK 的 LDS。

**为什么选 toxicity filtering**：
- problem-statement 预测 representation methods 在 toxicity filtering 上优势最大（高 FM2 bias）
- 如果 RepSim 连这个最有利的任务都输，整个 DATE-LM 叙事就站不住
- toxicity task 的 contrastive structure 最自然（toxic vs clean）

**缩放策略**：
1. 单 seed（seed 42）、单任务（toxicity）、单训练配置（LoRA）
2. RepSim 仅需 1 次前向传播提取 penultimate layer activations → 快速
3. TRAK 用 3 checkpoints（非完整 10 个）做快速估计
4. 如果 RepSim LDS >= TRAK LDS - 5pp → 扩展到 3 tasks × 5 seeds

**预计算力**：
- Pythia-1B LoRA fine-tuning: ~2 GPU-hours（单 seed）
- RepSim 计算: ~1 GPU-hour
- TRAK 3-checkpoint: ~3 GPU-hours
- **总计: ~6 GPU-hours（< 1 GPU-day）**

### 工程陷阱

- **DATE-LM 集成比想象中难 10 倍**: DATE-LM 是 NeurIPS 2025 benchmark，文档和 API 可能不完善。每个 task 的数据格式、评测协议、LOO ground truth 获取方式都不同。光是跑通 baseline 复现可能就要 1 周。这不是"调用 API"，是"读懂别人的 research code"。

- **Pythia-1B 内存墙**: 4× RTX 4090 各 24GB。Pythia-1B 本身约 4GB，但 per-sample gradient 计算（TRAK/IF 需要）会爆显存。TRAK 的 JL projection 帮助压缩，但 10 checkpoint 聚合的中间文件可能达 100GB+。EK-FAC IF 在 1B 模型上的 Kronecker 分解内存需求完全未知。

- **Contrastive scoring 构造不通用**: DDA 的 contrastive scoring 是为 hallucination tracing 设计的（base model vs fine-tuned）。对 data selection task，"base model"是什么？预训练 Pythia-1B vs fine-tuned Pythia-1B？但 data selection 评测的是"哪些训练数据对下游性能有帮助"，contrastive 参考点不自然。这意味着 2×2 矩阵在 data selection task 上可能有一整列是空的或勉强填充的。

- **RepSim 在 LDS 上的历史劣势**: AURA 自己的 CIFAR-10 数据已经显示 RepSim LDS = 0.074 vs IF LDS = 0.297。虽然 problem-statement 论证了 LLM 场景不同，但如果 RepSim 在 DATE-LM 上同样大幅落后，核心叙事（representation-space TDA 优于 parameter-space）就崩塌了。这不是"可以 reframe"的问题——reviewer 会直接说"你的框架预测错了"。

- **155 GPU-hours 预算 vs 42 GPU-hours 实际**: experiment-design.md 估算 DATE-LM 实验需要 ~155 GPU-hours，但 project.md 声明总预算 42 GPU-hours（已用 ~28）。即使加上 jinxulin 的 A6000，总预算缺口巨大。A6000 的可用性和排队时间也未明确。

- **代码重构开销**: 现有代码在 Sibyl 系统的 iter_001/exp/code/ 下，要迁移到 v3 结构（Codes/core/ + Codes/experiments/）需要非平凡的重构。同时要保持已有结果可复现。这个隐性开销通常被低估 2-3 倍。

### 综合预估

- **日历时间**（到 NeurIPS 投稿 ~2026 年 5 月）：
  - DATE-LM 环境搭建 + baseline 复现：2 周（乐观 1 周）
  - 最小 Pilot（RepSim vs TRAK, toxicity, 单 seed）：3-5 天
  - 如果 Pilot 通过 → 完整 2×2 实验（3 tasks × 主要方法 × 3 seeds 最少）：3 周
  - 统计分析 + 论文写作：2 周
  - **总计：~8 周，但只有 ~8 周可用** → 零容错余量

- **算力**：
  - 完整实验需 ~155 GPU-hours（experiment-design.md 估算）
  - 可用：xuchang0 4×4090 + jinxulin 4×A6000
  - AURA 原始 42h 预算已近耗尽（剩余 ~14h），DATE-LM 实验需要额外预算
  - **实际可行的最小配置**：3 tasks × {RepSim, TRAK, Contrastive-RepSim} × 3 seeds ≈ 60 GPU-hours（砍掉 IF EK-FAC 和 DDA 这两个计算最重的方法）

- **主要工程风险**（按严重程度排序）：
  1. **范围失控**（Critical）：从 CIFAR-10/ResNet-18 跳到 Pythia-1B/DATE-LM 是质的飞跃，不是量的增长。所有已验证的代码和结论都不能直接复用。这本质上是**启动一个新项目**而非扩展现有项目。
  2. **RepSim LDS 赌注**（High）：如果 RepSim 在 DATE-LM 上 LDS 不行，核心 RQ1 失败，整篇论文的叙事基础崩塌。而 AURA 自己的小规模数据暗示 RepSim LDS 表现差。
  3. **时间零余量**（High）：8 周干 8 周的活，任何一个环节延迟都导致错过 deadline。
  4. **BSS 方向被抛弃但未善后**（Medium）：problem-statement 完全转向 DATE-LM 评测叙事，但 project.md 的核心创新（BSS 诊断）被边缘化。是两条线同时推进还是放弃 BSS？需要明确决策。

### 务实建议

1. **立即做 DATE-LM Pilot**（6 GPU-hours）：在做任何进一步设计前，先跑通 RepSim vs TRAK on toxicity filtering。如果 RepSim LDS < TRAK LDS - 10pp，立即停止 DATE-LM 方向，回到 BSS 诊断路线。

2. **砍方法数量**：2×2 矩阵里 DDA 和 IF EK-FAC 是计算最重的（各 40 GPU-hours on A6000）。如果 A6000 不确定可用，直接砍掉，用 {TRAK, RepSim, Contrastive-RepSim} 三角对比替代完整 2×2。损失交互分析的严谨性，但保住核心叙事。

3. **明确 BSS vs DATE-LM 优先级**：不能两条线同时推。BSS 需要 ~14 GPU-hours 完成 Phase 2a/3，DATE-LM 最少 60 GPU-hours。建议：如果 DATE-LM Pilot 通过，全力 DATE-LM；如果失败，全力 BSS 诊断论文。

4. **5 seeds → 3 seeds**：从 5 seeds 砍到 3 seeds 可节省 40% 算力。3 seeds 足以报告 mean +/- std，peer review 可接受。

### 综合评价

**Verdict: 有条件通过，但需立即执行 kill-or-go Pilot**

当前 problem-statement 的核心赌注是"representation-space TDA 在 LLM scale 上系统性优于 parameter-space"。这个赌注既没有先验支持（AURA 的 CIFAR-10 数据反而暗示相反结论），也没有快速验证（DATE-LM 环境尚未搭建）。在投入 3-4 周工程努力之前，必须用 ≤1 GPU-day 的 Pilot 验证这个核心假设。如果 Pilot 失败，应果断回到 BSS 诊断方向——那条线有已确认的 Phase 1/2b 数据支撑，工程风险低得多，且 ~14 GPU-hours 即可完成。
