# TECA — TDA-Editing Consistency Analysis

## Core Idea

TDA 评估的根本困境是所有现有方法都依赖重训练获取 ground truth，而重训练本身不可靠。TECA 提出一个**完全独立的验证通道**：利用模型编辑（ROME/MEMIT）作为 TDA 的外部验证信号。

核心观察：**TDA 和模型编辑在参数空间中操作同一个对象，但从相反的方向。**
- TDA：给定训练样本 z，模型的哪些参数编码了 z 的影响？→ 梯度方向 $g_z$
- ROME：要编辑事实 f，需要改变哪些参数？→ 参数更新 $\Delta\theta$

如果两者指向相同的参数子空间，就相互验证。如果不一致，至少有一个是错的。

## Research Questions

- **RQ-T1**: TDA 归因方向与 ROME 编辑方向在参数空间中是否对齐？（TECS 基本验证）
- **RQ-T2**: TECS 与 LDS 对 TDA 方法的排序是否一致？（TECS 作为 LDS 替代的有效性）
- **RQ-T3**: 哪些类型的事实/知识的 TECS 高，哪些低？（TECS 的适用边界）
- **RQ-T4**: MEMIT 批量编辑 vs WSIF 子集归因的一致性如何？（集合级扩展）

## Motivation & Background

### 评估的独立性问题

当前所有 TDA 评估方法本质上是同一个实验的变体——删除数据 → 重训练 → 观察变化：

| 评估方法 | Ground Truth | 核心缺陷 |
|---------|-------------|---------|
| LDS | Leave-k-out 重训练 | 大数据集信噪比低；重训练随机性 |
| Spearman | 同上 | miss-relation 问题 (2303.12922) |
| LOO counterfactual | 单样本删除重训练 | 非凸模型 loss 震荡 |
| DATE-LM 三任务 | Heterogeneous filtering | Gumbel 温度敏感 |

**TECA 的核心价值**：提供一个不依赖重训练的独立验证信号。

### 模型编辑的理论基础

ROME (2202.05262) 的 causal tracing 发现：
1. 事实知识主要集中在中间层 MLP 的输出空间
2. 编辑通过 rank-1 更新 $W_{l^*}^{new} = W_{l^*} + \Lambda (C^{-1} k)^T$ 实现
3. 编辑的层 $l^*$ 通过 causal tracing 自动确定

IF 在层 $l^*$ 的参数上的影响力公式：

$$\mathcal{I}_{l^*}(z, z_{test}) = -\nabla_{\theta_{l^*}} L(z_{test})^T H_{l^*}^{-1} \nabla_{\theta_{l^*}} L(z)$$

两者在同一参数子空间中操作 → 一致性检查成为可能。

## Method

### TDA-Editing Consistency Score (TECS)

输入：
- 事实 $f = (s, r, o)$（如 "Eiffel Tower, located in, Paris"）
- TDA 方法 $M$（TRAK/LoGra/IF/RepSim/BM25）
- 编辑方法 $E$（ROME/MEMIT）

计算：
1. **编辑方向**：运行 $E$ 编辑事实 $f$ → 参数更新 $\Delta\theta_E$
2. **归因方向**：用 $M$ 计算 top-$k$ 训练样本 $\{z_1, ..., z_k\}$ → 聚合梯度 $g_M = \frac{1}{k}\sum_{i=1}^k \nabla_\theta L(z_i)$
3. **一致性**：$\text{TECS}_f = \cos(\Delta\theta_E^{l^*}, g_M^{l^*})$（仅在编辑层 $l^*$ 的参数上计算）

方法级汇总：$\overline{\text{TECS}}(M) = \frac{1}{N}\sum_{f} \text{TECS}_f(M)$

### TECS 的理论预期

| 方法 | 预期 TECS | 理由 |
|------|----------|------|
| IF (精确 Hessian) | 高 | 直接操作参数空间，理论上与编辑方向对齐 |
| TRAK / LoGra | 中-高 | 近似 IF，但随机投影/低秩可能丢失方向信息 |
| RepSim | 低 | 不操作参数空间，仅表示相似性 |
| BM25 | ~0 (random) | 纯词频统计，与参数空间无关 |
| Random baseline | ~0 | 高维空间随机向量余弦期望 |

如果实验结果符合此预期 → TECS 是有意义的指标。

## Experimental Plan

### Phase 1: TECS 基本验证（2 周，GPT-2-XL / GPT-J-6B）
- **数据集**：CounterFact（~21K 事实，ROME 标准 benchmark），随机采样 500 个
- **TDA 方法**：{TRAK, LoGra, IF(EK-FAC), RepSim, BM25, random baseline}
- **编辑方法**：ROME（单事实编辑）
- **指标**：每个方法的 $\overline{\text{TECS}}$
- **关键验证**：random baseline TECS ≈ 0；梯度方法 TECS 显著 > 0

### Phase 2: TECS vs LDS rank correlation（2 周，GPT-2）
- 在 CounterFact 子集上同时计算 TECS 和 LDS
- 对比两个指标对 TDA 方法的排序
- **关键阈值**：Spearman(TECS_ranking, LDS_ranking) > 0.7 → TECS 是 LDS 可靠替代

### Phase 3: TECS 分解分析（1 周）
- 按事实类型分解 TECS：哪些事实高，哪些低？
- 按层分解：编辑层 vs 非编辑层的 TECS 差异
- 假设：频繁出现的常识事实 TECS 高；罕见事实 TECS 低

### Phase 4: 集合级扩展（1 周）
- MEMIT 批量编辑 vs SIGMA/WSIF 子集归因
- 验证集合级 TECS 一致性

## Expected Results & Failure Modes

### 最佳情况
- TECS 与 LDS 排序高度一致（Spearman > 0.7），但计算成本低 100×
- 梯度方法 TECS 显著 > BM25 → 梯度方法确实定位到知识存储的参数区域
- BM25 TECS ≈ random → lexical matching 与参数空间知识定位无关

### 有价值的负面结果
- 所有方法 TECS ≈ random → TDA 和编辑定位到不同参数区域 → 知识可能是分布式存储的
- TECS 与 LDS 排序相反 → 归因方向与编辑方向测量不同的东西 → 重要理论发现

### 主要风险
- ROME 编辑有 side effects（编辑一个事实影响其他事实）
- ROME 仅适用于 factual associations，不覆盖 style/behavior 归因
- 层对齐：ROME rank-1 update 在一层，TDA 梯度跨所有层

## Connections to Existing Projects

- **CRA**: TECS 为 CRA 的表示空间 vs 参数空间方法对比提供新评估维度
- **SIGMA**: MEMIT 批量编辑 vs WSIF 子集归因 → SIGMA 的独立验证
- **AURA**: TECS 和 TRV 是两个独立的"归因质量诊断工具"，互补关系
- **Dir 7 (Model Editing)**: 直接消费 Dir 7 的方法论

## Key References

- Meng et al. (2022), 2202.05262: ROME — causal tracing + rank-one editing
- Meng et al. (2022), 2210.07229: MEMIT — mass editing memory in transformers
- CounterFact dataset: ~21K factual associations benchmark
- 2303.12922: Evaluation crisis — Spearman miss-relation
- 2506.12965: d-TDA — distributional IF theory
- 2409.18153: MISS — subset non-additivity (Phase 4 connection)
