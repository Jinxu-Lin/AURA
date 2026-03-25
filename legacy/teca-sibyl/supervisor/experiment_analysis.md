# 实验结果分析

## 实验结果概要

三个阶段的 pilot 实验已完成（Phase 1 ROME验证、Phase 2 TDA梯度计算、Phase 3 TECS核心测量），核心发现如下：

**Phase 1 — ROME 编辑验证**: GO。100/100 facts 编辑成功（efficacy = 100%），远超 75% 门槛。Mean P(new) post-edit = 0.978。Delta tensors 形状正确 (6400, 1600)。

**Phase 2 — TDA 梯度计算**: GO。100/100 facts 产生有效梯度，mean g_M norm = 0.1896 >> 1e-8 阈值，angular variance = 0.048（适中的梯度一致性），无 NaN。

**Phase 3 — TECS 核心测量**: **NEGATIVE**。这是关键结果：
- TECS 均值 = 0.000157（实质为零）
- TECS 标准差 = 0.00676
- 95% Bootstrap CI = [-0.00117, 0.00146]（跨越零）
- 56 个正值 / 44 个负值（近似随机）
- **Cohen's d vs Null-A (random fact) = 0.050**，远低于 0.2 阈值
- Cohen's d vs Null-B (cross-layer) = 0.078
- Cohen's d vs Null-C (shuffled) = 0.024
- Cohen's d vs Null-D (random direction) = 0.022
- Cohen's d vs Null-E (test gradient) = 0.211（唯一略高于阈值的比较，但 Null-E 本身的 std = 0.029 远大于 TECS 的 std = 0.007，说明 Null-E 不是合格的基线）
- **Bonferroni 校正后：所有 5 个比较均不显著**

决策门控明确判定：`NEGATIVE`，理由：`d(vs Null-A) = 0.050 <= 0.2`。

## 各方观点总结

注意：result_debate 目录为空（乐观者、怀疑论者、战略家的辩论尚未执行）。以下基于 idea 阶段各 perspective 的预判来分析。

- **乐观者（预判）**：理论预测 SNR ~ rho_k * rho_v * sqrt(d_k)，即使弱关联（rho_k * rho_v > 0.08）也应产生 Cohen's d > 0.3。实际 d = 0.05 意味着 rho_k * rho_v < 0.013，编辑和归因方向之间几乎没有真实的几何对齐信号。
- **怀疑论者（预判）**：Contrarian 早在 idea 阶段就提出三个根本挑战——(1) ROME 方向可能是统计伪迹而非知识几何，(2) TDA 梯度对 LLM 不可靠，(3) 10M 维空间中的余弦相似度近乎真空。Phase 3 结果验证了这些担忧：TECS 信号与噪声不可区分。
- **战略家（预判）**：Proposal 自身就设计了 dual-outcome 路径——TECS ~ 0 时进入 Negative Path（子空间几何分析、whitening 分解、MEMIT 对比）。这正是当前应走的路径。

## 分析

### 1. 方法可行性
核心方法（计算 ROME delta_W 与 TDA 梯度 g_M 之间的余弦相似度）在工程上完全可行——Phase 1 和 Phase 2 都顺利通过门控。问题不在于方法实现，而在于**被测量的信号不存在**。TECS 在编辑层 l*=17 处的值与所有空白基线不可区分。

### 2. 性能表现
TECS mean = 0.000157，Cohen's d vs Null-A = 0.050。这不是"接近基线"，而是"等于基线"。95% CI 跨越零，Bonferroni 校正后无一显著。编辑方向和归因方向在参数空间中**没有可检测的几何对齐**。

### 3. 改进空间
Proposal 自身的 Negative Path 提供了三条清晰的后续路线：
- **子空间几何分析**：用 principal angle analysis 区分"结构化不对齐"与"随机不对齐"
- **Whitening 分解**：比较 TECS_whitened vs TECS_unwhitened，检验 ROME 的 C^{-1} 旋转是否是几何间隙的根源
- **MEMIT 对比**：多层分布式编辑是否表现出不同的对齐模式

这些分析不需要新的大规模实验，可在已有的 delta_W 和 g_M 数据上快速完成（预计 20-40 分钟）。

### 4. 时间成本
- 已投入约 85 分钟 GPU 时间（Phase 1: 32min + Phase 2: 4min + Phase 3 ~50min）
- Negative Path 追加分析预计 20-40 分钟
- 若 pivot 到 Alternative 1（Knowledge Fingerprinting）需额外 75 分钟
- **继续 Negative Path 是最高效的选择**：利用已有数据，无需重新计算

### 5. 核心假设验证
- **H1 (Core signal) — 被否定**: Cohen's d = 0.05 << 0.3 预期值。TECS 在编辑层不显著高于机会水平。
- **H2-H5 需要正信号才能测试**：由于 H1 被否定，dose-response (H3)、layer specificity (H4)、spectral selectivity (H5) 的测试前提不成立。
- **H6 (Whitening gap) 和 H7 (Structured incommensurability) 是当前应测试的假设**：这是 Negative Path 的核心。

## 决策理由

**不需要 PIVOT 到全新课题**，原因如下：

1. **Proposal 已预设了 NEGATIVE 结果的科学价值**：风险评估中 "TECS ~ 0 with structured misalignment" 的概率被估计为 30%，"TECS ~ 0 with whitening explaining the gap" 为 15%。总计 45% 的概率会走到 Negative Path。这不是失败，而是设计内的路径分支。

2. **Negative Path 本身可产出高质量论文**：如 Proposal 所述，"Structured incommensurability of knowledge operations" 或 "ROME's whitening rotates away from natural knowledge geometry" 都是有价值的发现。知识编辑和训练数据归因在参数空间中几何不可通约——这本身就回答了 Hase et al. (2023) 的 localization-editing disconnect。

3. **数据基础已就绪**：100 个 facts 的 delta_W 和 g_M 都已保存，Negative Path 分析可直接复用这些数据。

4. **Alternatives 不适用于当前情况**：
   - Alternative 1 (Fingerprinting) 的触发条件是 "d ~ 0.1-0.2（模糊结果）"，但实际 d = 0.05，结果不模糊而是明确为负
   - Alternative 2 (Confound Detector) 的触发条件是 "d > 0.2 但无 dose-response"，但 d 远未达 0.2
   - Alternative 3 (Cross-Layer Consistency) 的触发条件是 "Phase 2 失败"，但 Phase 2 通过了

**结论**：按 Proposal 预设的 Negative Path 继续推进。执行子空间几何分析 (H7)、whitening 分解 (H6)、以及 MEMIT 对比。如果 Negative Path 分析也未能发现结构化不对齐（即概率 10% 的 "dead end" 场景），再 pivot 到 Alternative 1。

## DECISION: PROCEED
