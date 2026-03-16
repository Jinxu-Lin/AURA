## [Interdisciplinary] 跨学科者视角

### 跨域对应物

#### 类比 A — 计量经济学 / 半参数效率理论

**对应关系**：AURA 的核心问题——"两个不完美的估计器（IF 和 RepSim），各自在不同条件下失效，如何原则性地融合？"——与半参数统计中 **doubly robust estimation** 的原始动机结构同构。具体映射：

| AURA | 半参数统计 |
|------|-----------|
| IF（一阶 Taylor 近似） | 倾向得分加权估计（propensity score） |
| RepSim（表示空间相似度） | 结果回归估计（outcome regression） |
| Hessian 近似误差 | 倾向得分模型错误指定 |
| 表示空间退化/坍缩 | 结果模型错误指定 |
| RA-TDA 融合 | AIPW（Augmented Inverse Probability Weighting） |

**类比深度**：**深层类比** — 数学映射明确：Chernozhukov et al. (2018) 的 DML (Double Machine Learning) 框架给出了 doubly robust 估计量的一般形式 $\hat{\theta} = \mathbb{E}[\hat{m}(X)] + \mathbb{E}[\hat{\alpha}(X)(Y - \hat{m}(X))]$，其中第一项对应 outcome regression（RepSim），第二项对应 bias correction via influence function（IF residual）。AURA 的 $\tau_{RA}(z) = s_{rep}(z) + \lambda(TRV) \cdot \text{residual}(IF | RepSim)$ 直接镜像了这一结构。这不仅是概念类比——Neyman orthogonality 条件和 semiparametric efficiency bound 可以直接迁移，为 RA-TDA 提供理论最优性保证。

**该领域的已有解法**：AIPW / TMLE (Targeted Maximum Likelihood Estimation, van der Laan & Rose 2011) — 在因果推断中已有成熟的 doubly robust 估计理论，包括效率界、渐近正态性证明、以及自适应权重选择方法。

**可借鉴的核心洞察**：DR 估计的关键不只是"把两个估计加起来"，而是 **Neyman orthogonality** — 融合估计量对任一组件的一阶扰动不敏感。AURA 应验证其 $\tau_{RA}$ 是否满足类似的正交性条件，否则融合可能不比单个估计器更鲁棒。

#### 类比 B — 稳健统计学 / Breakdown Point 理论

**对应关系**：TRV（Attribution Robustness Value）试图量化"归因结果对 Hessian 扰动的敏感性"，这与稳健统计中的 **influence function sensitivity curve** 和 **breakdown point** 概念直接对应：

| AURA | 稳健统计 |
|------|---------|
| TRV（Hessian 扰动下的 top-k 变化率） | Sensitivity curve（单样本扰动对估计量的影响） |
| "TRV 高 = IF 不可信" | "Gross error sensitivity 高 = 估计量不稳健" |
| λ(TRV) 权重调节 | Tuning constant in M-estimators（Huber's k） |
| breakdown point（多大比例的扰动使估计崩溃） | breakdown point（Donoho & Huber 1983） |

**类比深度**：**深层类比** — Hampel (1974) 的 influence function 在稳健统计中的原始定义 $IF(x; T, F) = \lim_{\epsilon \to 0} \frac{T((1-\epsilon)F + \epsilon \delta_x) - T(F)}{\epsilon}$ 与 TDA 中的 influence function 在数学上同源。Hampel 进而定义了 gross error sensitivity $\gamma^* = \sup_x |IF(x)|$ 和 local shift sensitivity — 这些正是 TRV 试图度量的对象，但 AURA 目前似乎在重新发明轮子。

**该领域的已有解法**：M-estimators 和 S-estimators 提供了系统的框架：给定 efficiency-robustness 的 tradeoff（由 breakdown point 参数化），自动确定最优 tuning constant。Huber (1981) 和 Maronna et al. (2019) 给出了完整理论。

**可借鉴的核心洞察**：稳健统计的核心教训是 **efficiency-robustness tradeoff 有精确的数学边界**（Hampel's optimality theory）。TRV 不应只是一个经验指标，而应能推导出：在给定 TRV 阈值下，RA-TDA 的 minimax optimal 权重 λ 是什么。

### 未被利用的工具

- **TMLE (Targeted Maximum Likelihood Estimation)**：来自生物统计/因果推断（van der Laan & Rose 2011）。核心原理：通过"targeting step"迭代修正初始估计，使最终估计同时满足 doubly robust 和 semiparametrically efficient。引入障碍：TMLE 需要定义"目标参数"（target parameter）——在 TDA 场景中，需要明确"什么是我们要估计的真实归因值"，而 TDA 的 ground truth 定义本身就是开放问题（LOO retraining 虽可作为 gold standard，但计算昂贵且有自身偏差）。

- **Finite-sample breakdown point 分析**：来自稳健统计（Donoho & Huber 1983）。核心原理：给定一组 Hessian 近似方法，计算使 top-k 归因完全翻转所需的最小扰动幅度。引入障碍：需要将 Hessian 近似误差空间形式化为一个可操作的"污染模型"——AURA 的 TRV 已部分做到这一点，但尚未建立对应的 minimax 理论。

- **Stein's Unbiased Risk Estimate (SURE)**：来自非参数统计。核心原理：在不知道真实值的情况下，无偏估计 MSE 风险，从而自动选择平滑/正则化参数。可以用来自动选择 λ(TRV)，而不需要 LOO retraining 作为 ground truth。引入障碍：SURE 要求估计量对数据可微——IF 和 RepSim 通常满足，但 top-k selection 引入离散性，需要用 soft-top-k 松弛。

### 跨域盲点与教训

- **"Superefficiency 陷阱"**：半参数理论中一个深刻教训是：在特定分布假设下可以达到超效率（super-efficient），但这种估计量在模型轻微错误指定时表现灾难性地差。AURA 的 RA-TDA 如果过度 tune λ(TRV) 使其在某些 benchmark 上"完美"，可能正在走这条路。稳健统计的建议是：宁可稍微损失效率，也要保证 breakdown point 足够高。具体对 AURA 的警示：不要在 single benchmark 上优化 λ 的选择策略，而要在 worst-case 意义上保证鲁棒性。

- **"Curse of dimensionality in semiparametric estimation"**：DML / DR 估计在高维设置下的收敛速率受 nuisance parameter 估计速率的乘积约束（$n^{-1/4}$ rate requirement）。在 TDA 场景中，"nuisance parameter"对应 Hessian 和 representation 的估计质量——如果二者都估计得不好（正是 AURA 要解决的场景），DR 估计可能没有理论保证。这意味着：**doubly robust 不等于 doubly immune** — DR 要求至少一个组件"足够好"，而 AURA 的前提恰恰是两个都可能很差，这是一个需要正面解决的理论张力。

- **稳健统计中的 "redescending" M-estimators 教训**：将异常值影响完全压到零（类似"当 TRV 极高时完全信任 RepSim、忽略 IF"）会导致估计量出现多个局部极值和不连续性。如果 λ(TRV) 在某个 TRV 阈值处急剧从 1 跳到 0，可能引入不稳定行为。应确保 λ 是 TRV 的连续单调函数。

### 建议引入路径

1. **将 TRV 形式化为 Hampel 的 gross error sensitivity 的计算版本**：不需要改变 TRV 的经验计算方式，但重新解释其数学含义——TRV 就是 influence function 在 Hessian 扰动空间上的 sensitivity curve 的离散近似。这立即给出 TRV 的理论意义（它度量的是什么量的什么性质），并且可以借用 Hampel optimality theory 来导出 λ(TRV) 的 minimax optimal 形式（连续单调递减函数，形如 Huber's ψ-function 的调优常数）。

2. **引入 Neyman orthogonality 检验作为 RA-TDA 的理论验证**：在 RA-TDA 的公式推导中，检验 $\tau_{RA}$ 对 IF 组件和 RepSim 组件的一阶偏导是否在真实值处为零。如果满足，则 RA-TDA 具有 doubly robust 性质的理论保证；如果不满足，需要调整融合公式（可能需要添加 targeting step，类似 TMLE）。这是一个纯推导层面的检验，不增加任何实验成本。

3. **用 SURE 或 cross-validation 替代 LOO retraining 来选择 λ**：当前方案中 λ(TRV) 的校准似乎依赖某种 ground truth（LOO retraining）。引入 SURE 可以在无 ground truth 的情况下自适应选择 λ，将 RA-TDA 从"需要 oracle"变为"fully self-calibrating"。这将是独立于主框架的一个增量改进，修改范围限于 λ 的选择策略。
