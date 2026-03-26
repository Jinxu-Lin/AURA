## [Contrarian] 反对者视角

### 假设挑战

⚠️ **假设 H2（IF 和 RepSim 误差近似独立）**：这是整个 Phase 2（RA-TDA 融合框架）的理论基石，但支撑评级为"弱"，且种子卡自身承认 "IF and RepSim both depend on same model parameters"。这不是一个小问题——IF 通过梯度/Hessian 依赖模型参数，RepSim 通过表征空间依赖模型参数，两者的误差来源在本质上是耦合的。具体地：当模型在某个 test point 的局部损失面高度非凸时（Hessian 近似差的典型场景），该点的表征梯度同样不稳定，RepSim 的 cosine similarity 也会退化。这意味着 IF 失效的点上 RepSim 大概率也会出问题，融合的"互补性"前提不成立。种子卡引用的 IF-RepSim 相关性 0.37-0.45 看似支持低相关（独立），但**误差的相关性和值的相关性是完全不同的统计量**——值不相关不意味着误差不相关。AURA 需要在实验前就明确定义"误差独立性"的可操作化指标，否则 Phase 2 的理论分析（Phase 3 的 DR consistency）将建立在未经验证的假设之上。

⚠️ **假设 H4（SI 是 TRV 有效代理）**：Phase 0 的 Spectral Diagnosis 依赖 SI（Spectral Index）作为 TRV 的廉价预筛指标，但支撑评级同样为"弱"。SI 来自 2512.09103（Natural W-TRAK），其设计目的是度量 Hessian 的 Lipschitz 条件——这是一个全局/层级的谱属性，而 TRV 是 per-test-point 的 top-k 稳定性指标。从全局谱条件到逐点排名稳定性之间存在巨大的粒度差距。可以构造反例：一个 Hessian 整体谱条件良好（低 SI）的模型，在特定 test point 附近仍可能有 top-k 归因排名不稳定（高 TRV），因为 top-k 排名对归因分值的微小扰动极度敏感（尤其当多个训练样本的归因分值相近时）。如果 SI → TRV 的代理关系不成立，整个 Phase 0 预检模块就失去意义，变成一个增加计算开销但不提供有效信号的冗余步骤。

⚠️ **假设 H1（归因敏感性因 test point 而异且可预测）**：虽然支撑评级为"中"，但"可预测"这一部分实际上没有任何直接的文献支撑。2509.23437 证明了严格的 Hessian 层级结构（IHVP > Arnoldi > EK-FAC > Identity），但这是关于近似质量的全局排序，不是关于特定 test point 敏感性可预测性的证据。从"不同近似下 top-k 变化大"到"我们可以用某个特征预测哪些点变化大"之间，存在一个未经验证的推理跳跃。如果敏感性的决定因素是高维的、非线性的、且与 test point 的可观测特征弱相关，那么 TRV 虽然可以计算但不可预测，整个自适应融合权重 lambda(TRV) 就失去了实用价值。

### 反事实场景

**如果核心洞察是错的**：AURA 的核心洞察是"Hessian 近似敏感性是可诊断的，且诊断信号可以指导融合策略"。如果这是错的——即敏感性虽然存在但本质上不可预测（如同量子测量的随机性，只能事后计算不能事前诊断），那么 RA-TDA 退化为一个固定权重的 IF+RepSim 加权平均，与简单 ensemble baseline 无本质区别，论文的核心贡献蒸发。

**最可能的实验失败场景**：

- **场景 1（融合无增益）**：实验中发现 lambda(TRV) 的自适应效果与固定 lambda 相比无统计显著差异。具体机制：TRV 的分布高度集中（大多数 test point 的 TRV 值相近），导致自适应权重 lambda(TRV) 实际上退化为近似常数。这意味着"诊断信号"虽然存在但方差不够大，无法驱动有意义的差异化处理。这是 DL 中"超参敏感型创新"的典型模式——只有精心选择的案例才能展示自适应的优势，平均而言与 naive baseline 无异。

- **场景 2（Phase 3 理论分析无法闭合）**：DR（Doubly Robust）consistency 的理论证明需要 IF 和 RepSim 误差的独立性条件（H2）。如果该条件不成立（如上所述），理论贡献将只能退化为"在某些条件下成立"的弱结果，而这些条件在实践中是否满足需要额外验证，形成循环论证。审稿人会直接指出：如果你不能先验证误差独立性，你的理论保证就不具备实用价值。

### 被低估的竞争方法

**有** — 以下两个方向被低估：

1. **2512.09103 Natural W-TRAK + 简单 ensemble baseline**：种子卡自己提到"Could reviewers say 'just use W-TRAK'?"——答案很可能是 yes。W-TRAK 已经通过 Natural Wasserstein 度量提供了 certified robustness，如果再加上一个简单的 IF+RepSim 固定权重 ensemble（无需 TRV 诊断），这个两步方案可能在实验中与 RA-TDA 表现相当，同时概念上更简单、理论保证更强。AURA 必须在实验中明确展示 RA-TDA 超越"W-TRAK + naive ensemble"这个组合 baseline，否则贡献将被严重质疑。

2. **ASTRA (2507.14740) "just use better Hessian"路线**：如果问题是"Hessian 近似质量影响归因"，那最直接的解法不是"诊断+融合"，而是"用更好的近似"。ASTRA 的路线虽然计算更贵，但概念上更干净（直接解决根因而非做 workaround）。AURA 需要论证在计算预算约束下自己的方案更优——但如果计算预算足以跑 TRV（需要在多个 Hessian 近似下重复计算归因），那跑一个更好的 Hessian 近似可能反而更高效。这里有一个自洽性陷阱：计算 TRV 本身就需要跑多个 Hessian 近似下的归因，其计算开销可能与直接用最好的近似相当。

### 生死线评估

**如果结果上限是：RA-TDA 在 top-k overlap 指标上比 W-TRAK 和 naive ensemble 均只高 1-3%**：**不值得发表** — TDA 社区尚未成熟到对微小的 top-k 归因改进有高需求，且该改进幅度在统计噪声范围内（不同随机种子导致的 variance 通常就在这个量级）。AURA 要值得发表，需要满足以下至少一条：(a) 在至少一个重要 failure mode（如分布偏移下的归因）上展示 >10% 的 top-k overlap 改进；(b) 提供在竞争方法不具备的理论保证（但这依赖 H2，风险高）；(c) 发现并实证 TRV 作为 TDA 可信度指标本身的独立价值（即使融合不成功，"知道什么时候不该信任归因"也有发表价值——但这把贡献缩小到了 Phase 1，Phase 2-3 变成附属品）。建议认真考虑 Phase 1（TRV 诊断工具）作为独立贡献的可能性，而非强行绑定不成熟的 Phase 2-3 融合框架。
