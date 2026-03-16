## [Innovator] 创新者视角

### 当前方向天花板评估

**A 会 poster 级别，有条件冲 spotlight。** 理由：AURA 的核心叙事——从"改善归因精度"到"量化归因可信度"——确实是一个 **问题重新定义**（meta-level 从 method improvement 到 reliability diagnosis），这比"换一个更好的 Hessian 近似"有本质区别。Phase 1 (TRV) 如果能证明归因敏感性可预测且与下游质量相关，单独就值 poster。Phase 2 (RA-TDA) 的 DR 融合如果能在 IF 和 RepSim 各自失效场景中均不退化，叙事会非常完整。但天花板被两个因素压制：(1) TRV 的计算需要多层级 Hessian 评估，成本可能太高以至于实际中不会有人用它——这会让审稿人质疑 practical impact；(2) DR 融合的理论保证依赖 H2（误差近似独立），而这个假设缺乏理论和实证支撑，一旦不成立，RA-TDA 退化为启发式加权平均，失去了"doubly robust"的核心卖点。要冲 spotlight，需要 TRV 有一个足够廉价的近似（SI 如果有效就是关键突破），且 DR 性质需要在至少 3 个以上不同场景中稳健成立。

### 升维角度 A — 从 Post-hoc 诊断到 By-design 归因

**核心洞察**：AURA 目前是"先做归因，再诊断可信度，再修补"，本质上仍是 post-hoc 修复范式。真正颠覆性的方向是：如果我们已经知道 Hessian 近似误差会在特定子空间放大，为什么不在训练时就构造一个对近似误差鲁棒的特征空间？

**具体建议**：将 TRV 的洞察（哪些方向是脆弱的）反馈到特征学习阶段。具体来说，在训练 TRAK 投影矩阵时，加入一个正则化项，惩罚特征协方差矩阵的条件数 kappa——这等价于 Tikhonov 正则化，但动机完全不同（不是为了数值稳定性，而是为了归因鲁棒性）。这样训练出的特征空间中，所有 test point 的 TRV 都会天然较高，Phase 2 的融合可能根本不需要。更激进地，可以把"归因稳定性"作为辅助训练目标（类似对比学习中的 uniformity loss），使模型的特征空间 by-design 对 TDA 友好。这会将 AURA 从"TDA 的诊断工具"升级为"TDA-aware 训练范式"，影响力完全不同。

**属于哪种升维模式**：Post-hoc --> By-design。

**实现代价**：方向重构 — 需要修改训练流程，但核心组件（kappa 计算、SI 计算）已在 Phase 0 中实现，正则化项的梯度可以通过 TRAK 投影矩阵反传。风险在于这可能需要在训练时就决定投影维度，且正则化可能与下游任务性能冲突。如果仅作为 Phase 3 的理论扩展（"如果我们可以影响训练..."），代价是中改级别。

### 升维角度 B — 从 Point-wise TRV 到 Distribution-level Robustness Certificate

**核心洞察**：当前 TRV 是 per-test-point 的诊断指标，但下游应用（数据选择、unlearning）关心的是整个 test distribution 的归因质量。d-TDA (2506.12965) 已经将影响函数从 point-wise 推广到 distributional，AURA 应该走同样的路：不只诊断"这个 test point 的归因稳不稳定"，而是回答"对于这个任务（如删除有毒数据），归因驱动的数据选择策略在 Hessian 近似误差下的性能退化有多大？"这是一个 task-level robustness certificate，比 point-level TRV 有用得多。

**具体建议**：定义 Task Robustness Value (Task-RV) = 归因驱动的数据选择在不同 Hessian 近似下的下游任务性能方差。这可以通过聚合 TRV 来高效近似：高 TRV 的 test point 对 Hessian 选择不敏感 -> 选出的训练集稳定 -> 下游性能稳定。关键创新是：不需要真正跑下游任务多次，只需要 TRV 的分布就能预测 task-level robustness。这与 d-TDA 的 distributional 框架天然对接，理论上可以推导出 Task-RV 的 closed-form 界（基于 TRV 分布的矩）。如果成功，AURA 的叙事从"诊断工具"升级为"TDA 部署的 reliability engineering 框架"。

**属于哪种升维模式**：Task-specific --> General（从单 test point 诊断到任务级保证）。

**实现代价**：中改 — 理论推导需要 1-2 周，实验可以在 Phase 1 数据上直接扩展（已有 per-point TRV，只需聚合+验证与下游性能的相关性）。

### 当前设计的保守之处

- **Hessian 层级链是固定的（H -> GGN -> Fisher -> EK-FAC -> Diagonal -> Identity）**：这个层级来自 2509.23437 的经验排序，但不同模型架构、不同训练阶段的最优层级可能完全不同。为什么不让 TRV 本身来数据驱动地发现最优近似链？比如用 Transformer 时，block-diagonal 近似可能比 EK-FAC 更自然，但当前框架不会发现这一点。一个更大胆的做法是将 Hessian 近似选择本身建模为一个 hyperparameter selection 问题，用 TRV 作为 validation metric。

- **RA-TDA 只融合两类方法（IF + RepSim）**：因果推断的 DR 框架理论上支持任意数量的"半参数组件"。为什么不把 d-TDA (distributional)、TracIn (checkpoint-level)、DataInf (per-layer) 等也纳入融合？当前的二元融合是工程方便（两个组件最简单），而不是理论最优。即使不在 Phase 2 实现多组件融合，至少应该讨论扩展性，否则审稿人会问"为什么只融合这两个？"

- **TRV 的 0.5 阈值（Jaccard >= 0.5）是任意的**：这个阈值决定了哪些 test point 被判定为"归因可信"，但 0.5 没有理论依据。不同的下游任务可能需要不同的阈值（版权归因需要高精度 -> 高阈值；数据清洗只需大致方向 -> 低阈值）。一个更原则性的做法是让阈值由下游任务的 cost function 决定，类似假设检验中的显著性水平选择。
