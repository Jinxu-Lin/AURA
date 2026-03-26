## [Theorist] 理论家视角

### 理论基础审查

**核心操作的理论动机**：

本方向包含两个核心操作：

1. **TRV (TDA Robustness Value)**：通过在 Hessian 近似质量梯度（exact → GGN → ... → Identity）上测量归因排序的 Jaccard@k 稳定性，量化每个 test point 的归因鲁棒性。这本质上是对 influence function 关于 Hessian 扰动的**敏感性分析**（sensitivity analysis），在因果推断文献中有成熟的理论框架（Rosenbaum bounds, Ding & VanderWeele 2016）。将 Hessian 近似视为"扰动参数"，TRV 测量的是归因在该扰动下的断裂点（breakdown point），这在统计稳健性理论中有明确的对应物。

2. **RA-TDA**：将 IF 残差（回归掉 RepSim 后的部分）以 TRV 依赖的权重 λ(TRV) 加回 RepSim baseline。这试图借用因果推断中 Doubly Robust (DR) estimation 的思想（Chernozhukov et al. 2018, Robins et al. 1994）：如果两个估计器的误差"近似独立"，则适当加权组合可以降低总误差。

**评价**：有直觉但关键步骤未形式化 — TRV 作为诊断工具的理论动机清晰且站得住脚；RA-TDA 的 DR 类比在概念层面有吸引力，但从因果推断 DR 到 TDA 融合的映射存在严重的形式化缺口（详见下文）。

### 数学欠账

- **欠账1：DR 类比的映射不严密**。因果推断中 DR 的数学保证依赖于明确的统计模型：outcome model 和 propensity model 各自对应一个概率模型，DR 估计器在其中一个正确时保持一致性（consistency）。在 RA-TDA 中，IF 和 RepSim 并不是两个对同一因果量的独立估计——IF 估计的是 LOO 效应，RepSim 估计的是表征空间相似性，它们的"估计目标"（estimand）不同。因此 H2（"IF 和 RepSim 误差近似独立"）不仅是"弱假设"，而是在缺乏共同 estimand 的情况下，连"独立"的含义都需要重新定义。这不是一个可以通过实验验证绕过的问题——它决定了 RA-TDA 的理论叙事是否成立。

- **欠账2：TRV 的离散化和单调性假设**。TRV 定义为"最大的 ℓ 使得 Jaccard 仍 ≥ 0.5"，这隐含假设 Jaccard@k(H_ℓ, H_1) 在 ℓ 增大时**单调递减**。但 Hessian 近似的质量排序（exact > GGN > diagonal > Identity）并不对所有模型架构和数据分布都成立——例如 GGN 在高度非凸区域可能比 exact Hessian 的某些数值近似更稳定。如果单调性不成立，TRV 的 max 定义会产生不一致的行为。此外，0.5 阈值的选择缺乏理论依据。

- **欠账3：λ(TRV) 的函数形式**。RA-TDA 中 λ(TRV) 被要求是"monotone increasing function"——TRV 高时更多依赖 IF residual，TRV 低时更多依赖 RepSim baseline。这个方向正确（高 TRV 意味着 IF 可信），但具体的函数形式（线性？sigmoid？阶梯？）直接影响融合效果，目前没有理论指导选择。这容易退化为纯调参问题，削弱"原则性方法"的叙事。

- **欠账4：H5 与 2512.09103 的理论边界模糊**。2512.09103 的 Theorem 5.4 给出 SI(z) = φ(z)^T Q^{-1} φ(z) 作为归因的 Lipschitz 常数。如果 SI 是 TRV 的有效代理（H4），那么 TRV 本质上也在度量同样的量——与 2512.09103 的差异化就变得不清晰。理论上需要说明 TRV 捕获了 SI 未覆盖的什么信息（或反之），否则 Phase 0 的预检与既有工作重叠。

### 已有理论支撑或反例

- **支撑1：TRAK 的排序保持理论 (2602.01312)**。Park et al. 证明即使 IF 的绝对值有很大近似误差，排序（ranking）在一定条件下仍能保持。这为 TRV 的 Jaccard@k 度量提供了理论基础——排序的保持是比绝对值准确更弱但更实际的要求。但这同时也是一个**潜在反例**：如果排序本身对 Hessian 近似足够鲁棒（如 TRAK 所暗示），那么 TRV 的变异度可能很低，Phase 1 的诊断价值就有限。

- **支撑2：d-TDA 的分布视角 (2506.12965)**。IF "secretly distributional" 的结论意味着 IF 度量的是跨训练种子的期望 LOO，这为理解 IF 误差来源提供了新框架——Hessian 近似误差可被重新理解为对这个期望的有偏估计。这可能为 RA-TDA 提供更自然的 estimand。

- **反例/挑战1：Revisiting Fragility (2303.12922)** 中的 "retrain-from-optimal failure" 表明，即使归因排序正确，基于归因的 retrain 实验也可能失败。这意味着 TRV（基于排序稳定性的度量）可能无法预测归因在下游任务中的实际效用——排序稳定 ≠ 排序正确。

- **反例/挑战2：Q^{-1} Mahalanobis 与 Concept Influence (2602.14869) 的张力**。如果 TRV 高的点恰好是 Q^{-1} 加权后低方差方向上的点，那么高 TRV 可能系统性地偏向某类 test point（低概念多样性的"容易"点），而低 TRV 的"困难"点反而是概念归因中更有意义的。这会使 TRV 的实用价值产生悖论。

- **理论-实践 gap 评估**：当前方向的理论分析介于"指导性"和"装饰性"之间。TRV 的敏感性分析框架是真正指导性的——它直接告诉我们如何设计诊断工具。但 RA-TDA 的 DR 类比目前更接近装饰性——它提供了一个好听的叙事（"doubly robust"），但由于 estimand 不统一，这个叙事尚不能真正指导 λ(TRV) 的函数形式选择或预测融合应在何时优于单一方法。

### 理论强化建议

最关键的一步是**为 IF 和 RepSim 建立共同的 estimand**，使 DR 类比从隐喻升级为可操作的理论框架。具体建议：利用 d-TDA (2506.12965) 的分布视角，将目标 estimand 定义为"跨 Hessian 近似的期望归因排序"（类似 d-TDA 中跨训练种子的期望）。在此框架下，IF 是 exact Hessian 下的点估计（高方差，低偏差），RepSim 是所有 Hessian 下的边际估计（低方差，高偏差），RA-TDA 则自然成为方差-偏差权衡的加权组合，λ(TRV) 的选择可通过 bias-variance decomposition 原则性地确定。这一步不需要完整的数学证明，但需要形式化到足以产生可检验的预测（如"在 TRV 的中间区域，RA-TDA 的改善应最大"）。
