## [Interdisciplinary] 跨学科者视角

### 跨域对应物

#### 类比 A — 无线通信中的 MIMO 分集与最大比合并

**对应关系**：AURA 的核心发现——IF 和 RepSim 在同一测试点上产生系统性反相关排序（tau = -0.467），且不一致性可以预测哪种方法更可靠（AUROC = 0.691）——在无线通信中有精确的数学对应物。MIMO（多入多出）系统面对的问题是：同一信号经过多条独立衰落信道到达接收端，每条信道在不同时刻的可靠性不同。最优接收策略不是固定选择某条信道，而是**最大比合并**（Maximal Ratio Combining, MRC）：按瞬时信噪比加权合并所有信道。IF 和 RepSim 对应两条"信道"——参数空间和表示空间——它们对同一测试点的"衰落"（Hessian 敏感性 vs 表示层噪声）是独立的。BSS 本质上是在估计"参数空间信道"的瞬时信噪比，disagreement 是在检测两条信道的相对可靠性。

**类比深度**：深层。MRC 的最优性来自 Cauchy-Schwarz 不等式，要求各信道噪声独立——这与 FM1/FM2 独立性假说（RQ3）直接同构。若 FM1 和 FM2 确实独立，则存在一个数学上最优的加权合并方案（类似于 MRC），而非简单的"路由"二选一。分集增益的闭式下界仅依赖信道数和独立性程度，可为 RQ3 的 additivity 测试提供更强的理论预测。

**已有解法**：MRC 在实践中使用信道状态信息（CSI）进行逐符号自适应加权。在时变信道中，CSI 通过导频信号（pilot symbols）实时估计。BSS 和 disagreement 都可以看作是估计 CSI 的不同导频方案。

**可借鉴洞察**：(1) 当前 AURA 的 adaptive selection 是硬切换（route to IF or RepSim），这在通信中对应选择分集（selection diversity），是最差的分集方案。**软合并**（加权平均两种方法的归一化得分）在理论上总是优于硬切换，且实现更简单——无需精确阈值。(2) MRC 的最优权重与瞬时 SNR 成正比。如果 BSS 可以估计 IF 的"瞬时可靠性"，那么 `w_IF = f(BSS)` 和 `w_RepSim = 1 - w_IF` 的软合并策略可能 Pareto-dominate 当前的路由方案。(3) probe 数据显示 IF 在 471/500 点上优于 RepSim——在如此不平衡的场景下，硬路由的期望增益极其有限（最多在 29 个点上翻转选择），而软合并即使在 IF 占优的点上也能从 RepSim 获取互补信息。

#### 类比 B — 生态学中的 Beta 多样性与 Baselga 分解

**对应关系**：Jaccard@10 在生态学中就是 Jaccard 相似度——衡量两个群落共享物种的比例。J10 = 0.45-0.53（EK-FAC vs K-FAC）意味着两种 Hessian 近似"看到"的 top-10 训练样本只有约一半重叠，类似于两个相邻生态区域的物种周转率（beta diversity）约 50%。生态学对 beta diversity 的分解框架成熟度远超 ML：Baselga (2010) 将 beta diversity 分解为**周转组分**（turnover，物种替换）和**嵌套组分**（nestedness，物种丢失），两个组分有独立的生态驱动因子。

**类比深度**：表面。数学上使用相同的 Jaccard 指数，但生态学中的"物种"是离散实体而 TDA 的"训练样本影响"是连续得分的排序截断。嵌套/周转分解的数学形式可以直接套用，但生态学中的因果解释（环境过滤 vs 扩散限制）没有直接对应。

**已有解法**：Baselga 分解：beta_total = beta_turnover + beta_nestedness。turnover 高意味着两种 Hessian 近似在同一测试点看到完全不同的"优势物种"（训练样本）——对应严重的 Hessian 敏感性。nestedness 高意味着一种近似的 top-10 是另一种的子集——对应温和的排序扰动。

**可借鉴洞察**：对现有 J10 数据做 Baselga 分解可以揭示 Hessian 敏感性的**性质**，而非仅仅**程度**。如果某类测试点主要表现为 turnover（完全不同的 top-10），这比 nestedness（排序微调）更令人担忧，因为前者意味着归因指向完全不同的训练数据子集。这个分解几乎零成本（纯粹是对已有数据的后处理），但能为论文增加一个有区分度的分析维度。

### 未被利用的工具

- **最大比合并 / 软加权融合**（无线通信，Cauchy-Schwarz 最优性）：当前方案是硬路由（选 IF 或 RepSim），但 MRC 理论保证软加权 >= 硬切换。实现上只需 `s_fused = w * s_IF_norm + (1-w) * s_RepSim_norm`，其中 `w = sigmoid(a * BSS_residual + b * disagreement + c)`，参数通过留出验证集上的 LDS 学习。barrier：需要对两种方法的得分做归一化使其可比（quantile normalization 或 z-score 即可）。

- **Baselga beta-diversity 分解**（生态学，物种周转 vs 嵌套分解）：将 Jaccard dissimilarity 分解为 turnover + nestedness 两个正交组分。barrier 极低（纯后处理计算，Python `scipy` 即可，< 1 小时工作量）。可以揭示 Hessian 敏感性的定性类型差异。

- **Fisher 信息几何**（微分几何 / 信息论）：Hessian 在 loss = negative log-likelihood 时就是 Fisher 信息矩阵。FM1 的深层原因可能是：在欧氏度量下 R^B 中的梯度内积没有物理意义（各参数的"尺度"不同），而 representation space 自带更均匀的几何结构。barrier：这个视角主要提供理论解释深度，不直接改变实验设计，但可以大幅提升论文中 FM1 诊断的理论说服力——从"维度高所以 SNR 低"升级为"参数空间几何退化导致内积无意义"。

- **Bode 灵敏度积分**（经典控制理论）：在线性系统中，降低某个频段的灵敏度必然增加另一个频段的灵敏度（灵敏度守恒）。如果这对 BSS 成立，意味着不存在对所有光谱尺度均匀好的 Hessian 近似——这为 per-test-point 方法选择提供了理论 justification，因为不同测试点的梯度集中在不同光谱区域。验证方法：检查 BSS_outlier + BSS_edge + BSS_bulk 是否在测试点间近似常数。

### 跨域盲点与教训

- **"分集增益远大于路由增益"的教训（通信，1950s）**：通信领域早已证明，硬切换（选择最佳信道）的增益远小于软合并（加权合并所有信道）。AURA 当前的 adaptive method selection 是硬路由设计，在 IF 占优 471/500 点的不平衡场景下，硬路由的期望增益极其有限。H3（adaptive > uniform by 2% LDS）的门槛在硬路由框架下可能很难达到，但在软合并框架下更容易实现。**风险**：如果坚持硬路由设计，H3 很可能 fail，导致论文降级为 diagnostic-only。

- **"共线性不是退化，而是结构"的教训（统计物理）**：BSS_outlier 与 gradient norm 的 rho = 0.906 被标记为"退化风险"。但在统计物理中，序参量之间的强相关是常态——重要的是**剩余的独立信息**（"快模式"）。正确的 reframe 是：BSS = gradient norm（粗调） + spectral residual（精调），BSS 的价值在于 spectral residual 部分是否有独立预测力。这比试图证明 BSS "不是 gradient norm" 更诚实也更有说服力。如果 partial BSS（regress out gradient norm 后的残差）的 rho(partial_BSS, J10) > 0.15，那就够了——论文可以说 BSS 在 gradient norm 基础上提供了 spectral correction。

- **"交互效应检测需要 4-16 倍样本量"的教训（实验心理学）**：RQ3 测试 FM1/FM2 的 additivity（交互效应 < 30% of min main effect），这在实验心理学中对应交互效应检验。心理学的教训是：检测交互效应所需的样本量通常是检测主效应的 4-16 倍。DATE-LM 只有 3 个任务，任务级 ANOVA 没有统计效力。method-design.md 正确地指出应该用 per-sample permutation test，但即使如此，如果 FM1 和 FM2 的主效应各为 ~5pp，那 30% 的交互（~1.5pp）可能落在置信区间内。**建议**：明确报告检测交互效应的统计功效（power analysis），并预注册效应量阈值。

### 建议引入路径

最高优先级：将 Phase 3 的 adaptive method selection 从硬路由改为**软加权合并**（MRC 框架）。具体改动：`s_fused = w(z) * s_IF_norm(z) + (1-w(z)) * s_RepSim_norm(z)`，其中权重 `w(z)` 是 BSS_residual 和/或 disagreement 的单调函数，通过留一验证学习。硬路由降为 ablation baseline。这个改动对 Phase 2a 零影响，但可能显著提升 H3 的通过概率。第二优先级：对现有 Phase 1 的 J10 数据做 Baselga 分解（turnover vs nestedness），纯后处理，< 1 小时，可以立即丰富 Phase 1 的分析深度。第三优先级：在论文 theoretical framework 部分验证 Bode 灵敏度积分是否成立（检查 total BSS 是否近似常数），若成立则提供"不存在 universally best Hessian 近似"的理论保证，大幅增强 per-test-point diagnostic 的 motivation。
