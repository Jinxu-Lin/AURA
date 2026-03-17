## [Interdisciplinary] 跨学科者视角 — 第二轮

### 修订响应性评估

v1.1 采纳了第一轮的核心建议：(1) DR 叙事修正为 "diagnostic-guided adaptive ensemble"（Interdisciplinary 建议的核心被采纳）。(2) 探针增加了梯度 norm 偏相关检验和 TRV-LDS 对比。(3) §2.3 在局限性分析中更认真地对待了"稳定 ≠ 正确"悖论。但 Stability Selection 基准和 Conformal Prediction 方向未被纳入——这可以理解（被综合者裁定为"建议修改"），但值得在第二轮重新审视。

### 跨域对应物（深化分析）

#### 类比 A — 鲁棒统计学（Robust Statistics）— 深化

**第一轮结论的强化**：v1.1 的修订使 AURA 的框架更接近鲁棒统计学的规范路径。"诊断（TRV）→ 利用（adaptive ensemble）"的两阶段结构正是 Hampel (1974) 的 influence function → bounded-influence estimator 路径。Phase 2 从 DR 修正为 "diagnostic-guided adaptive ensemble" 后，理论定位更准确——这不是 doubly robust estimation，而是 **model selection with diagnostics**（根据诊断信号选择在两个模型之间分配权重）。

**新的深层对应**：v1.1 §1.2 的差异化论证（TRV vs Daunce vs BIF 度量不同来源的不确定性）有一个精确的鲁棒统计学对应物——**不确定性的分解 (uncertainty decomposition)**。在贝叶斯决策理论中，总不确定性 = 认知不确定性 (epistemic) + 随机不确定性 (aleatoric)。类比：

| 鲁棒统计学 | AURA 生态系统 |
|-----------|-------------|
| Aleatoric uncertainty（数据固有噪声） | Daunce: 训练随机性（模型扰动方差） |
| Epistemic uncertainty（模型选择不确定性） | AURA TRV: Hessian 近似选择敏感性 |
| Parameter uncertainty（参数估计不确定性） | BIF: 后验方差 |

这个映射暗示 AURA 的独特价值：TRV 捕获的是 **model specification uncertainty**（不同 Hessian 近似 = 不同模型规约），这在不确定性分解文献中是一个被明确识别的、与其他来源不同的维度。如果 AURA 能用这个框架来组织叙事——"TRV 是 TDA 中模型规约不确定性的首个操作化度量"——论文的理论定位会更精准，且与 Daunce/BIF 的差异化不再是"我们度量不同的东西"（模糊），而是"我们度量不确定性分解中的特定组件"（精确）。

**类比深度**：深层类比 — 数学映射直接可用（不确定性分解的形式化框架可以直接应用于 TDA）。

#### 类比 B — 多模型推断（Multi-Model Inference, Burnham & Anderson 2002）

**新增类比**：TRV 量化的"不同 Hessian 近似给出不同归因"本质上是一个 **模型不确定性** 问题。在生态学和流行病学中，Model Averaging（Bayesian Model Averaging, BMA）被用来处理类似问题——当多个模型都有理论依据但给出不同预测时，不选择单一最好的模型，而是根据每个模型的可信度加权平均。

**对应关系**：
- 不同 Hessian 近似（H/GGN/EK-FAC/K-FAC） ↔ 候选模型集合
- TRV（Jaccard@k 稳定性） ↔ 模型间预测一致性
- SI 作为 TRV 代理 ↔ 模型复杂度/拟合度指标（如 AIC/BIC）
- RA-TDA 自适应融合 ↔ Bayesian Model Averaging

**类比深度**：深层类比 — BMA 的理论框架（model posterior weights proportional to marginal likelihood）可以形式化 TRV-guided 融合。具体地：如果将每种 Hessian 近似视为一个"模型"，BMA 权重正比于各模型的 marginal likelihood——而 TRV 低的 test point 意味着不同"模型"的预测差异大，BMA 权重分散，应降低总体置信度（或引入其他信息源如 RepSim）。

**该领域的已有解法**：BMA、Stacking（Wolpert 1992）、Model Confidence Set（Hansen et al. 2011）。

**可借鉴的核心洞察**：BMA 文献中一个关键教训是"模型不确定性往往被低估"——如果候选模型集合不够多样（如都是 Kronecker 类近似），模型平均的覆盖率下降。这回到了 Contrarian 的"共模偏差"问题——如果所有 Hessian 近似都有相同的结构性偏差，TRV 会低估真实不确定性。

### 未被利用的工具（更新）

- **不确定性分解框架**：建议将 TRV 显式定位为 "model specification uncertainty in TDA"，与 Daunce 的 "aleatoric/training uncertainty" 和 BIF 的 "parameter uncertainty" 形成不确定性分解的三个组件。这不仅改善叙事，还提供了一个 principled 的组合框架——如果未来想组合 TRV + Daunce + BIF，不确定性分解理论提供了理论指导（各组件相加 vs 取最大值 vs 根号平方和）。

- **Stability Selection（第一轮建议的跟进）**：仍然建议在探针中增加 Stability Selection 基准（5-10 次训练子采样的 top-k 选择频率）。理由不变——它提供了与 Hessian 层级链无关的稳定性信号，可验证 TRV 捕获的是通用稳定性还是 Hessian 特异性信号。但理解这在 v1.1 中被列为可选改进而非必须。

### 跨域盲点与教训（更新）

- **BMA 中的 "M-open" 问题**：如果真实数据生成过程不在候选模型集合中（none of the Hessian approximations is correct），BMA 的理论保证不成立。类比到 AURA：如果所有 Hessian 近似（H/GGN/EK-FAC/K-FAC）都是 loss landscape 的粗糙近似（对深度网络来说这是事实），TRV 量化的只是"粗糙近似之间的分歧"，不是"近似与真实之间的距离"。这是"稳定 ≠ 正确"悖论的更精确表述。

- **不确定性分解中的"交互项"被忽视**：在完整的不确定性分解中，总方差 = 模型规约方差 + 参数方差 + 随机方差 + **交互项**。如果 TRV 和 Daunce/BIF 的不确定性信号有非零交互，简单的"正交维度"叙事就不完全准确。探针中检验相关性可以部分回答这个问题。

### 建议引入路径（更新）

1. **短期（叙事层面，无额外实验）**：在 problem-statement.md §1.2 的差异化论证中，引入不确定性分解框架——将 TRV 定位为 "model specification uncertainty"，Daunce 为 "training stochasticity"，BIF 为 "parameter uncertainty"。这不改变任何技术内容，但大幅提升叙事的精准度和理论深度。Reviewer 更容易理解为什么三者是互补而非冗余的。

2. **短期（探针层面）**：在探针 Step 5-6 的 TRV 分析中，增加 multi-seed 归因方差作为 Daunce-style 不确定性的代理（已有 3 seed，不需额外计算），检验 Spearman(TRV, cross-seed_variance)。如果相关性低，"正交性"声称获得初步实证；如果高，需要修改叙事。

3. **中期（Phase 2）**：将 RA-TDA 的融合框架定位为 Bayesian Model Averaging 的变体——不同 Hessian 近似作为候选模型，TRV 指导权重分配，RepSim 作为 "non-parametric fallback"（当所有参数模型不确定性都高时切换到非参数方法）。这比 "diagnostic-guided adaptive ensemble" 更有理论基础。
