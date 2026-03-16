# AURA — Adaptive Unified Robust Attribution

## Core Idea

TDA 方法的归因结果对 Hessian 近似质量高度敏感，但目前没有工具量化这种敏感性，也没有原则性方法利用这一信息。AURA 分两阶段解决：

1. **Phase 1 (Sensitivity Analysis)**：提出 TDA Robustness Value (TRV)，量化每个 test point 的归因结果在 Hessian 近似扰动下的稳定性。TRV 高 = 归因可信，TRV 低 = 归因不可信。
2. **Phase 2 (Doubly Robust Fusion)**：基于 TRV 信号，原则性地融合梯度方法（IF/TRAK）与表示方法（RepSim/RepT），构建 Doubly Robust TDA 估计器——即使其中一个组件失效，融合后的估计仍然有效。

## Research Questions

- **RQ-A1**: 归因结果对 Hessian 近似扰动的敏感性是否可预测？（TRV 与 test point 特征的相关性）
- **RQ-A2**: TRV 高的 test point 子集上，LDS 是否显著更高？（TRV 作为归因质量预测器的有效性）
- **RQ-A3**: Doubly Robust TDA（RA-TDA）是否在 IF 失效场景中 ≈ RepSim，在 RepSim 失效场景中 ≈ IF？（DR 性质验证）
- **RQ-A4**: TRV 能否自适应指导 DR 融合权重？（高 TRV → 信任 IF，低 TRV → 信任 RepSim）

## Motivation & Background

### 评估可靠性危机

KB 中多篇论文揭示了 TDA 评估的系统性问题：
- **Spearman miss-relation** (2303.12922): 相同 Spearman 值可掩盖质量完全不同的估计
- **LOO 信噪比低** (2506.12965, d-TDA): 大数据集中单样本删除效应接近噪声水平
- **LDS 元级脆弱性**: LDS 依赖 leave-k-out 重训练，与 retrain-from-optimal 有相同根本问题

**AURA Phase 1 的核心转换**：不问"归因对不对"（需要不可靠的 ground truth），而问"归因稳不稳定"（只需扰动实验）。

### 参数空间 vs 表示空间的张力

- **IF/TRAK**（参数空间）：有因果理论基础，但 Hessian 近似是致命瓶颈。Li et al. (2409.19998) 证明在 LoRA + 低信噪比下 IF 完全失效（0-7%）
- **RepSim/RepT**（表示空间）：无因果理论基础，但有害数据/后门检测上远超 IF（96-100%），无需 Hessian，可扩展到 70B+
- **Concept Influence** (2602.14869): 向量类与梯度类的相关性仅 0.37-0.45，说明两者捕获了不同信息

**AURA Phase 2 的核心转换**：不选择方法，而是原则性地组合，借鉴因果推断的 Doubly Robust Estimation 框架。

## Prior Art: Natural W-TRAK (2512.09103)

2512.09103 提出了与 AURA Phase 1 高度相关的理论框架，需要明确 AURA 的差异化定位：

### 2512.09103 做了什么

- **Self-Influence (SI)**：$\text{SI}(z) = \phi(z)^T Q^{-1} \phi(z)$，其中 $Q$ 是 TRAK 特征协方差矩阵
- **核心定理 (Theorem 5.4)**：归因的 Natural Lipschitz 界 $L_{Nat} \leq 2\sqrt{\text{SI}(z_{test})} \cdot \sqrt{\text{SI}(z_i)} \cdot R_{whit}$
- **首次理论连接**：leverage score (SI) = 归因稳定性的 Lipschitz 常数
- **实验结果**：Natural W-TRAK 认证率 68.7% vs 标准 TRAK 0%（CIFAR-10/ResNet-18）
- **谱放大机制**：特征协方差条件数 κ ≈ 2.71×10⁵ 导致 Euclidean 认证完全退化

### AURA 与 2512.09103 的关键区别

| 维度 | 2512.09103 (Natural W-TRAK) | AURA |
|------|---------------------------|------|
| **扰动类型** | 测试分布偏移（Wasserstein 球） | Hessian 近似误差（计算近似） |
| **稳定性度量** | SI = Lipschitz 常数（理论界） | TRV = 经验 Jaccard 稳定性（实测） |
| **适用范围** | 仅 TRAK 框架（需要 TRAK 特征 φ） | 任何 TDA 方法（IF/TRAK/RepSim） |
| **利用稳定性信号** | 构造认证区间（被动诊断） | 指导方法融合权重（主动利用） |
| **目标** | "这个归因结果在分布偏移下还对吗？" | "这个归因结果在 Hessian 不精确时还对吗？→ 如果不对，用什么补救？" |

### AURA 应吸收的洞察

1. **SI 作为 TRV 的理论锚点**：SI 提供了 TRV 的解析上界。如果 SI 高（outlier/高 leverage 点），TRV 应该低（不稳定）。AURA Phase 1 可以用 SI 作为 TRV 的快速预测器，避免昂贵的多层级扰动实验。

2. **谱放大机制解释了"何时需要 DR 融合"**：条件数 κ 高 → 低方差方向的扰动被放大 → IF 不稳定 → 需要 RepSim 补偿。κ 本身可以作为"是否启用 Phase 2 融合"的全局开关。

3. **H-NW2 张力为 Phase 2 提供理论动机**：Natural Wasserstein 的 Q⁻¹ 降权低方差方向，但 Concept Influence (2602.14869) 证明低方差方向可能是语义最重要的方向。这意味着 W-TRAK 在"稳定性"和"语义完整性"之间做了隐式取舍——而 RepSim 恰好保留了这些语义方向。**这为 DR 融合提供了不依赖于"IF 失效"的独立理论动机**：即使 IF 本身没有失效，W-TRAK 的谱校正也可能丢失语义信息，RepSim 可以补回来。

4. **条件数 κ 作为诊断工具**：在 Phase 1 之前先计算 κ。如果 κ 小（特征协方差良态），Hessian 近似误差影响小，可能不需要 AURA 的复杂融合；如果 κ 大，Phase 1 + Phase 2 的完整流程是必要的。

## Method

### Phase 0: Spectral Diagnosis（来自 2512.09103 的快速预检）

计算 TRAK 特征协方差矩阵 $Q$ 的条件数 $\kappa = \lambda_{max}/\lambda_{min}$：
- $\kappa < 10^2$：特征空间良态，标准 TDA 方法可能足够稳定，跳过 Phase 1-2
- $\kappa > 10^3$：谱放大显著，Phase 1 诊断 + Phase 2 融合是必要的
- 同时计算每个训练/测试点的 $\text{SI}(z) = \phi(z)^T Q^{-1} \phi(z)$ 作为 TRV 的快速代理

### Phase 1: TDA Robustness Value (TRV)

对 test point $z_{test}$，定义其归因结果在 Hessian 近似 $\tilde{H}$ 下为 $\mathcal{A}(z_{test}, \tilde{H}) = \text{top-}k\text{ attributed samples}$。

**层级版 TRV**：

$$\text{TRV}_k(z_{test}) = \max_{\ell} \{ \ell \in \{1,...,L\} : J_k(\mathcal{A}(z_{test}, H_\ell), \mathcal{A}(z_{test}, H_1)) \geq 0.5 \}$$

其中 $H_1 = H$（精确），$H_2 = \text{GGN}$，...，$H_L = I$（RepSim），$J_k$ 是 Jaccard@k 重叠率。

**扰动版 TRV**（更细粒度）：

$$\text{TRV}_k^\sigma(z_{test}) = \max\{ \sigma : J_k(\mathcal{A}(z_{test}, \tilde{H}_\sigma), \mathcal{A}(z_{test}, H_0)) \geq 0.5 \}$$

其中 $\tilde{H}_\sigma = \text{EK-FAC} + \sigma \cdot \mathcal{N}$。

**与 SI 的关系验证**：计算 Spearman(TRV, 1/SI)。如果高相关（> 0.7），SI 可以作为 TRV 的免费代理（零额外计算成本）。

### Phase 2: Residual Augmented TDA (RA-TDA)

$$\hat{\tau}_{RA}(z) = s_{rep}(z) + \lambda(\text{TRV}) \cdot r(z)$$

其中：
- $s_{rep}(z) = \cos(h(z), h(z_{test}))$（表示相似度）
- $r(z) = s_{if}(z) - \hat{\mathbb{E}}[s_{if} | s_{rep}]$（IF 在 RepSim 无法解释方向上的残差）
- $\lambda(\text{TRV})$：TRV 的单调递增函数（高 TRV → 多信任 IF 残差）

**Phase 2 的双重理论动机**：
1. **IF 失效补偿**（原始动机）：Hessian 近似差 → IF 不可靠 → RepSim 兜底
2. **语义信息补偿**（来自 2512.09103 H-NW2 张力）：即使 IF 本身可靠，Natural Wasserstein 校正会降权低方差语义方向 → RepSim 保留这些方向 → 融合后语义信息更完整

## Experimental Plan

### Phase 0 实验（2 天，CIFAR-10 + ResNet）
- 计算 TRAK 特征协方差 Q，报告条件数 κ
- 计算所有训练/测试点的 SI(z)
- 与 2512.09103 的 κ ≈ 2.71×10⁵ 对比（同 benchmark 应可复现）
- 确认 κ > 10³ 后进入 Phase 1

### Phase 1 实验（1-2 周，CIFAR-10 + ResNet）
- 使用 2509.23437（Turner 组）的 Hessian 层级代码
- 对 1000 个 test points 计算 TRV
- 分析 TRV 与 test point 特征的相关性（置信度、损失值、梯度范数、outlier 程度）
- **新增**：计算 Spearman(TRV, 1/SI)，验证 SI 是否是 TRV 的有效代理
- 验证：高 TRV 子集的 LDS 是否显著更高

### Phase 2 实验（2-3 周，GPT-2 / Llama-7B）
- IF 失效场景：Li et al. 有害数据识别（LoRA 低信噪比）
- RepSim 失效场景：DATE-LM factual attribution（entity corruption）
- 对比：IF-only / RepSim-only / naive average / RA-TDA
- 验证 DR 性质：RA-TDA 在两类失效场景中是否都不差于最好的单一方法

### Phase 3 理论分析（2 周）
- RA-TDA 一致性条件推导
- 与 d-TDA 分布性框架的连接
- 方差分析：DR-TDA 方差 vs max(Var(IF), Var(RepSim))

## Connections to Existing Projects

- **CRA**: AURA Phase 2 为 CRA 的表示空间方法分类提供理论升级——从"比较"到"融合"
- **SIGMA**: TRV 可诊断 Woodbury 修正是否改善归因稳定性
- **FMAS**: TRV 可诊断 FM 上 LDS 崩溃的原因（全局不稳定 vs 部分不稳定）
- **2512.09103** (Natural Geometry): Phase 0 直接使用其 SI/κ 诊断工具；H-NW2 张力为 Phase 2 提供独立理论动机
- **2509.23437** (Better Hessians): 直接使用其 Hessian 层级链代码作为 Phase 1 基础设施

## Positioning vs 2512.09103

```
2512.09103 (Natural W-TRAK)          AURA
──────────────────────────          ──────────
诊断：SI → Lipschitz 界              诊断：TRV → 经验稳定性（+ SI 作为代理）
应对：构造认证区间（被动）            应对：IF+RepSim 自适应融合（主动）
扰动：测试分布偏移                    扰动：Hessian 近似误差
框架：仅 TRAK                        框架：任何 TDA 方法
结果：知道"不可信"                    结果：知道"不可信" → 切换到更可信的方法
```

AURA 不是 2512.09103 的竞争者，而是其**下游消费者**：
- 用 SI 和 κ 作为 Phase 0 快速预检（2512.09103 的工具）
- 用 TRV 做更精细的方法级稳定性诊断（AURA 的新贡献）
- 用 DR 融合主动利用稳定性信号改善归因质量（2512.09103 未涉及）

## Key References

- Li et al. (2024), 2512.09103: Natural W-TRAK — SI/κ 诊断工具 + 谱放大理论（Phase 0 基础）
- Cinelli & Hazlett (2020), 1912.07236: Sensitivity analysis framework (Dir 6 inspiration)
- Chernozhukov et al. (2018), 1608.00060: Doubly robust / debiased ML (Dir 6 theoretical basis)
- Hong et al. (2025), 2509.23437: Hessian approximation hierarchy (Phase 1 infrastructure)
- Li et al. (2025), 2409.19998: IF failure on LLM (Phase 2 test scenario)
- Mlodozeniec et al. (2025), 2506.12965: d-TDA distributional IF (theoretical connection)
- Concept Influence (2602.14869): IF-RepSim low correlation evidence (0.37-0.45)
- Concept Influence H-NW2 tension: Natural Wasserstein 降权低方差语义方向（Phase 2 独立动机）
