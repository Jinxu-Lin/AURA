---
version: "1.1"
created: "2026-03-16"
last_modified: "2026-03-16"
entry_mode: "rs_revise"
iteration_major: 1
iteration_minor: 1
---

# Problem Statement

## 1. Gap 定义

### 1.1 现有方法概览

Training Data Attribution (TDA) 旨在量化每个训练样本对模型行为的影响。当前方法可分为两大族群：

**参数空间方法（Influence Functions 及变体）**：通过 Hessian 逆向量积 (iHVP) 估算训练样本对模型参数的影响。代表工作包括 TRAK (2303.14186, ICML 2023)、MAGIC (2504.16430)、SOURCE (2405.12186)、ASTRA (2507.14740, NeurIPS 2025)、LoGra (2405.13954)、TrackStar (2410.17413, ICLR 2025)。这类方法有明确的 counterfactual 语义（"移除该样本后损失如何变化"），但其精度严重依赖 Hessian 近似质量——Hong et al. (2509.23437) 首次系统证明了 H >= GGN >> Block-GGN >> EK-FAC >> K-FAC 的严格层级关系，其中 EK-FAC->K-FAC 的特征值 mismatch 是最大单一误差源（占 41-65%）。

**表示空间方法（RepSim 及变体）**：通过中间层表示的相似度衡量训练样本与测试样本的关联。代表工作包括 RepSim (2409.19998)、Concept Influence (2602.14869)、AirRep (2505.18513)。这类方法不依赖 Hessian 近似，在低信噪比场景（有害样本占比 < 10%）全面优于 IF (Li et al. 2409.19998: IF 0-7% vs RepSim 96-100%)，但缺乏 counterfactual 因果解释力。

**归因不确定性量化的已有探索**：近期两个 ICML 2025 工作从不同角度触及了归因可靠性问题，虽然未显式提出 per-test-point 诊断框架：
- **Daunce (2505.23223, ICML 2025)**：通过扰动模型集成（perturbed model ensemble）的 loss 协方差进行归因，其归因方差天然包含 per-sample 不确定性信号。Daunce 的不确定性来源于模型扰动（训练随机性的一阶近似），度量的是"如果模型参数轻微扰动，归因排名是否改变"。
- **Bayesian IF (BIF, 2509.26544, ICML 2025)**：用后验协方差替代 Hessian 求逆，后验方差直接作为归因不确定性估计。BIF 的不确定性来源于参数的后验不确定性（贝叶斯框架），度量的是"给定数据的后验分布下，归因的置信区间有多宽"。

这两个方法的可靠性信号与 AURA 的 TRV 在信息维度上存在本质差异（详见 §1.2），但它们的存在意味着"归因可靠性量化"这个大问题已有人在探索——AURA 需要精确界定自己的独特贡献。

**关键观察**：两类方法在同一数据集上的归因排名相关性极低（0.37-0.45，Kowal et al. 2602.14869），说明它们捕获了训练数据影响的不同方面。IF 捕获参数空间的 loss 敏感性信号，RepSim 捕获表示空间的语义相似性信号——二者在信息论意义上互补而非竞争。

### 1.2 Gap 陈述

**一句话**：现有 TDA 方法缺乏对 Hessian 近似选择导致的 per-test-point 归因排名不稳定性的诊断工具——用户无法判断"对于这个具体的测试样本，归因排名在多大程度上取决于 Hessian 近似的选择而非真实的数据影响"。

**详细分析**：

当前 TDA 研究的核心范式是"提出新方法 A，在 LDS 上超越旧方法 B"。但这个范式忽视了一个结构性问题：**同一方法在不同测试样本上的归因可靠性可能有数量级差异，而用户对此完全无感知**。

具体表现：
1. **Hessian 近似敏感性因 test point 而异**：Hong et al. (2509.23437) 证明不同 Hessian 近似（H/GGN/EK-FAC/K-FAC）产生不同的 LDS，但论文仅报告了跨所有 test point 的平均 LDS，未量化 **per-test-point 的归因稳定性变化**。Natural W-TRAK (2512.09103) 从理论侧证明 SI(z) = phi(z)^T Q^{-1} phi(z) 是归因稳定性的 Lipschitz 常数，谱放大 kappa 约 2.71x10^5 导致 Euclidean 认证 0%，但 Natural Wasserstein 达 68.7%——归因稳定性因度量空间而剧烈变化。
2. **两类方法各有系统性失效区域**：IF 在 LoRA fine-tuning + 低信噪比下完全失效 (2409.19998)；RepSim 在 factual attribution 场景不如 BM25 (2410.17413 间接证据)。但对于一个具体的测试样本，用户不知道此时 IF 更可靠还是 RepSim 更可靠。
3. **评估指标本身的可靠性危机**：LDS (Spearman 相关) 存在 miss-relation 问题 (2303.12922)；重训练 ground truth 在非凸模型上不可靠 (2303.12922)；attribution != influence (2410.17413, G4)。Distributional TDA (2506.12965) 揭示 IF "secretly distributional"，单次运行的 LOO 相关性接近 0% 是信噪比问题而非方法失效——但社区仍在用单次运行 LDS 评估方法。

**这个 Gap 的性质是"做了但缺少关键维度"**：现有方法都在回答"谁最有影响"，但没有方法从 Hessian 近似敏感性角度回答"这个答案有多可靠"。

**与 Daunce/BIF 的差异化**：Daunce 和 BIF 确实提供了归因不确定性估计，但它们度量的是**不同来源**的不确定性：
- **Daunce**：度量模型扰动（参数噪声注入）导致的归因方差——捕获的是"如果模型稍有不同，归因是否改变"，对应**训练随机性**维度。
- **BIF**：度量参数后验分布下的归因方差——捕获的是"给定先验和数据，参数不确定性多大程度传播到归因"，对应**贝叶斯不确定性**维度。
- **AURA 的 TRV**：度量 Hessian 近似选择（H/GGN/EK-FAC/K-FAC）导致的 top-k 归因排名变化——捕获的是"同一模型、同一数据，仅因方法实现选择不同而产生的归因分歧"，对应**方法选择敏感性**维度。

这三个维度在理论上正交：即使 Daunce 的扰动方差很低（模型参数稳定），TRV 仍可能很低（因为 Hessian 近似在该 test point 上产生截然不同的归因排名）；反之，Daunce 方差高的点上 TRV 可能很高（Hessian 近似间一致但对参数扰动敏感）。AURA 的独特贡献在于诊断**方法内部的实现选择**如何影响归因结论——这是从业者在使用 IF/TRAK 等方法时面临的最直接问题（"我该用 EK-FAC 还是 K-FAC？选择不同答案会变吗？"），而 Daunce/BIF 不回答这个问题。

### 1.3 Root Cause 分析

**根因类型**：被忽视的维度 + 技术局限

**Why Layer 1**：为什么没有 per-test-point 的归因可靠性指标？
**回答**：因为 TDA 社区将 Hessian 近似误差视为一个全局性问题——"方法 A 比方法 B 在所有 test point 上平均更好"。没有人追问"方法 A 在哪些 test point 上特别不可靠？"

**Why Layer 2**：为什么社区把 Hessian 近似误差当作全局性问题？
**回答**：因为 LDS 评估本身就是全局平均的——一个标量概括了跨所有 test point 的归因质量。这个评估范式不奖励（也不能检测）per-test-point 的质量差异。更深层地，Hong et al. (2509.23437) 虽然做了近似误差分解，但分解是按近似步骤（H->GGN->Block-GGN->EK-FAC->K-FAC）而非按 test point 进行的。

**Why Layer 3**：为什么 per-test-point 的归因质量差异存在？
**回答**：因为 Hessian 近似误差在参数空间中不是均匀分布的。不同 test point 的梯度 nabla_theta L(z_test, theta*) 落在参数空间的不同子空间上。如果某个 test point 的梯度恰好对齐了 Hessian 近似误差最大的方向（如 K-FAC 特征值 mismatch 最严重的特征方向），则该点的归因估计会特别不可靠。Natural W-TRAK (2512.09103) 的 Self-Influence SI(z) = phi(z)^T Q^{-1} phi(z) 正是捕获了这种"测试梯度与 Hessian 谱结构的对齐程度"——SI 大的点意味着梯度落在谱放大严重的方向上，归因对 Hessian 近似选择高度敏感。

**Root Cause 总结**：TDA 归因结果的可靠性因 test point 而异，根本原因是不同 test point 的梯度与 Hessian 近似误差谱结构的对齐程度不同。这种差异是可诊断的（通过 test point 梯度在 Hessian 谱上的投影分布），但目前既无工具量化它，也无原则性方法利用它改善归因。

### 1.4 Gap 评价

| 维度 | 评估 | 证据 |
|------|------|------|
| 重要性 | **高** | (1) 服务于所有现有 TDA 方法——任何 IF/TRAK/SOURCE/ASTRA 的用户都需要知道"何时不该信任归因结果"。(2) 位于 TDA 评估方法论危机的交汇处——LDS miss-relation (2303.12922)、attribution-influence misalignment (G4)、distributional TDA (2506.12965) 都指向"单一标量评估不够"。(3) 随着 TDA 从学术基准走向安全关键应用（有害数据溯源、版权归因），可靠性量化从"nice to have"变为必须。 |
| 新颖性 | **中-高** | (1) "归因可靠性量化"方向已有 Daunce (模型扰动方差) 和 BIF (后验方差) 从不同角度探索——但无人从 Hessian 近似选择敏感性角度系统量化 per-test-point 的归因排名稳定性。TRV 度量的是方法选择敏感性维度，与 Daunce/BIF 度量的训练随机性/贝叶斯不确定性维度理论正交。(2) Natural W-TRAK (2512.09103) 做了"被动诊断"（认证区间），但未做"主动利用"诊断信号改善归因。(3) Hong et al. (2509.23437) 做了 Hessian 近似误差分解，但按近似步骤而非按 test point 分解。(4) 跨领域迁移（sensitivity analysis -> TDA）无直接先例。 |
| 可解性 | **中-高** | (1) Phase 1 (TRV 诊断) 技术路径清晰：Hong et al. 的 Hessian 层级链代码已开源，主要工作是在此基础上计算 per-test-point 的 Jaccard@k 稳定性指标。(2) SI (Self-Influence) 作为 TRV 的廉价代理有理论基础 (2512.09103 Theorem 5.4)。(3) 核心风险：TRV 变异度是否足够——如果 > 80% test point 的 TRV 在同一等级，自适应无用。这正是探针需要验证的。 |

### 1.5 Research Questions

- **RQ1 (Phase 1 核心)**：在 CIFAR-10 / ResNet-18 设置下，不同 Hessian 近似（H/GGN/EK-FAC/K-FAC）产生的 top-k 归因集合的 Jaccard 相似度是否因 test point 而显著不同？具体地：TRV (Jaccard@k 在 Hessian 层级链下的稳定性) 的分布是否至少覆盖 3 个可区分的等级（每个等级占比 > 10%）？
  - **可证伪条件**：如果相邻 Hessian 等级的 avg Jaccard@10 > 0.85（即近似对归因影响普遍小），或 TRV 分布中 > 80% test point 落入同一等级，则 RQ1 否定。
  - **预测力**：如果 RQ1 成立，预测 TRV-low 的 test point 上 LDS 方差显著大于 TRV-high 的 test point。

- **RQ2 (Phase 2 条件)**：Self-Influence SI(z) 是否是 TRV 的有效快速代理（Spearman(SI, TRV) > 0.6）？
  - **可证伪条件**：如果 Spearman(SI, TRV) < 0.3，则 SI 不能替代完整 TRV 计算，Phase 2 的廉价诊断路线受阻。

- **RQ3 (Phase 2 条件，依赖 RQ1+RQ2 通过)**：基于 TRV/SI 信号的 IF+RepSim 自适应融合是否显著优于固定权重融合和单一方法？
  - **可证伪条件**：RA-TDA vs naive ensemble (固定 0.5:0.5 权重) 的 LDS 提升 < 2% absolute (Cohen's d < 0.5)。

## 2. 攻击角度

### 2.1 候选攻击角度

| 编号 | 核心 Idea | 与 Root Cause 匹配 | 可行性 | 选择理由 |
|------|-----------|-------------------|--------|---------|
| A1 | TRV 诊断 + 自适应融合：量化 per-test-point 归因稳定性（TRV），然后利用 TRV 信号原则性融合 IF 和 RepSim | 直接对应 root cause：不同 test point 梯度与 Hessian 误差谱对齐不同 -> TRV 量化这种对齐 -> 低 TRV 点用 RepSim 补偿 | 高（Phase 1 基础设施就绪） | **选定** |
| A2 | 端到端 Hessian 误差校正：学习一个 per-test-point 的 Hessian 校正因子，直接降低所有 test point 上的近似误差 | 间接对应 root cause：试图消除误差而非诊断+利用 | 低：缺乏 ground truth Hessian 的训练信号 | 排除：ASTRA 已在此方向做了部分工作（预条件 Neumann 迭代），且校正本身也有近似误差 |
| A3 | Distributional TRV：利用 d-TDA (2506.12965) 的多种子集成视角，定义分布式 TRV | 对应 root cause 的补充层面（训练随机性导致的归因不确定性） | 中：需要多种子训练，计算成本增加 10x | 排除：Phase 1 先验证 Hessian 近似维度的 TRV；d-TDA 视角可在 Phase 2 后作为扩展 |
| A4 | Concept-level 稳定性：用 Concept Influence (2602.14869) 在概念方向上测量稳定性，而非实例级 | 对应不同层面的 root cause（语义维度 vs 参数空间维度的稳定性） | 中：依赖概念向量质量（SAE/probe） | 排除：概念级是正交扩展方向，核心 TRV 定义应先在实例级验证 |

### 2.2 选定攻击角度

**核心 Idea**：提出 TDA Robustness Value (TRV) 作为 per-test-point 归因可靠性的诊断指标。TRV 基于 Jaccard@k 在 Hessian 近似层级链下的稳定性——对于每个 test point，计算不同 Hessian 近似产生的 top-k 归因集合之间的 Jaccard 相似度。TRV 高的 test point 意味着其归因对 Hessian 近似选择不敏感（可信），TRV 低的点意味着不同方法给出截然不同的归因（不可信）。进一步，探索 Self-Influence SI 作为 TRV 的廉价代理（SI 是归因稳定性的理论 Lipschitz 常数），以及基于 TRV 信号的 IF+RepSim 自适应融合。

**为什么可能有效**：Root cause 是不同 test point 梯度与 Hessian 误差谱的对齐程度不同。TRV 直接操作化了这种对齐——Jaccard@k 测量的正是"Hessian 近似的选择是否改变了归因结论"。SI 理论上捕获了同一信息的谱结构形式（phi(z)^T Q^{-1} phi(z) 量化 test point 梯度在 Hessian 谱上的放大程度）。Phase 2 采用 diagnostic-guided adaptive ensemble 策略：TRV/SI 作为诊断信号指导 IF 和 RepSim 的自适应加权——在 IF 不可靠的区域（低 TRV），增加 RepSim 的表示空间信号权重；在 IF 可靠的区域（高 TRV），保留 IF 的 counterfactual 语义优势。这不是严格的 Doubly Robust 估计（IF 和 RepSim 没有共同 estimand），而是基于可靠性诊断的实用融合框架。

### 2.3 攻击角度的局限性与风险

1. **TRV 变异度不足风险（最关键）**：如果绝大多数 test point 的 TRV 集中在同一等级，TRV 作为诊断工具的信息量为零。这是探针需要验证的核心假设。Empiricist 在辩论中设定了否证条件：TRV 分布至少 3 个等级各占 > 10%。

2. **"稳定 != 正确" 悖论**：TRV 高只保证"不同 Hessian 近似给出一致的归因"，不保证归因本身正确。如果所有 Hessian 近似都系统性地偏向相同的错误方向（例如由于 first-order 线性化本身的局限），TRV 高但归因仍然错误。缓解策略：(a) Phase 2 的 RepSim 补偿可部分应对，因为 RepSim 不依赖 Hessian；(b) 在 Phase 1 探针中增加 TRV-high vs TRV-low 的 LDS 对比验证——如果 TRV-high 的 test point 确实有更高的 LDS，则"稳定 -> 正确"假设获得经验支持。

3. **IF-RepSim 误差独立性假设 (H2) 弱**：4/6 辩论视角 + 4/4 审查视角一致指出此风险。"值不相关 != 误差不相关" (Contrarian)——两类方法的归因排名相关性低 (0.37-0.45) 不等于它们的估计误差不相关。如果 IF 和 RepSim 在同一类 test point 上同时失效（例如 OOD 样本），自适应融合的有效性降低。注意：Phase 2 的理论基础已从 DR 估计修正为 diagnostic-guided adaptive ensemble（见 §2.2），不再依赖严格的误差独立性，但 H2 越成立融合效果越好。

4. **计算成本**：完整 TRV 需要在多个 Hessian 近似下重复计算归因，成本为 O(M * N_test * N_train)，其中 M 是 Hessian 近似方法数。SI 代理能否充分降低成本取决于 Spearman(SI, TRV) 的实际值。

5. **TRV 分布退化的具体机制**（Contrarian 补充）：EK-FAC/K-FAC 的 Kronecker 分解来自网络架构本身（逐层独立假设），特征值 mismatch 模式可能是架构级而非 test-point 级的——即所有 test point 的梯度在同一组 Hessian 特征方向上被放大/压缩。如果这是主导效应，TRV 分布将高度集中，自适应融合退化为常数权重。探针将直接检验这一风险。

## 3. 探针方案（Dim 0）

### 3.1 核心假设

**如果这一点不成立，整个方向就不成立**：归因结果对 Hessian 近似的敏感性因 test point 而异，且这种差异的分布具有足够的变异度（非退化分布）。

具体化：在 CIFAR-10 / ResNet-18 设置下，不同 Hessian 近似产生的 top-10 归因集合的 Jaccard 相似度在 test point 之间的标准差 > 0.1，且分布至少覆盖 3 个可区分等级（TRV-high, TRV-medium, TRV-low），每个等级占比 > 10%。

### 3.2 最小实验方案

**设置**：
- 数据集：CIFAR-10（50,000 训练 / 10,000 测试）
- 模型：ResNet-18，SGD + cosine annealing，训练至收敛
- Hessian 近似层级链：EK-FAC, K-FAC（最小对比对，利用 Hong et al. 的发现——EK-FAC->K-FAC 是最大误差源）
- 作为补充：Block-GGN vs EK-FAC（第二大误差源）
- 利用 Hong et al. (2509.23437) 已开源的 Hessian 层级链代码

**步骤**：
1. **代码可行性验证（Step 0, 1-2 天）**：优先验证 Hong et al. (2509.23437) 代码是否支持 per-test-point 归因排名提取接口（而非仅支持聚合 LDS）。如果接口不存在需要自行实现，评估工作量后再决定是否继续。
2. 训练 3 个不同种子的 ResNet-18 模型（3-seed stability 检验，应对 Empiricist 的 seed confound 质疑）
3. 对每个种子的模型，在 EK-FAC 和 K-FAC 两种近似下分别计算 test set 中 200 个随机 test point 的 top-10 归因排名
4. 计算每个 test point 的 Jaccard@10(EK-FAC, K-FAC) 作为 TRV 的操作化定义
5. 分析 TRV 分布：均值、标准差、直方图、3-bin 分布
6. 计算 Self-Influence SI(z) = phi(z)^T Q^{-1} phi(z)（使用 EK-FAC 的 Q），检验 Spearman(SI, TRV)
7. **梯度 norm 控制的偏相关检验**：计算 Spearman(SI, TRV | ||grad||)——控制梯度 norm 后 SI-TRV 相关性是否仍显著。排除"SI 和 TRV 都只是梯度 norm 的代理变量"这一 confound（Contrarian 指出的假阳性风险）。
8. **TRV-high vs TRV-low 的 LDS 对比**：将 test point 按 TRV 分为高/低两组，比较两组的 LDS（使用 EK-FAC 归因 vs ground truth）。如果 TRV-high 组的 LDS 显著高于 TRV-low 组，支持"稳定 -> 正确"假设；否则 TRV 的诊断价值需要重新评估。
9. 选取 20 个 OOD test point（高斯噪声 / 从 CIFAR-100 采样），检验 TRV 在 OOD 条件下的行为（应对 Empiricist 的 OOD confound 质疑）

**代码量估计**：
- 模型训练：复用现有 CIFAR-10/ResNet 训练管线（~100 行修改）
- Hessian 层级链 + 归因计算：基于 Hong et al. 代码（~300 行适配）
- TRV 计算 + 分析：~200 行新代码
- 总计：~600 行代码

**计算需求**：
- 模型训练：3 seeds x ~30 min = 1.5 GPU-hours
- 归因计算：2 approx x 200 test points x 3 seeds = 需要逐样本 iHVP，估计 ~4-6 GPU-hours（EK-FAC 预计算 + 归因打分）
- 总计：**~6-8 GPU-hours**

### 3.3 Pass 标准

| 指标 | Pass | Borderline | Fail |
|------|------|-----------|------|
| TRV (Jaccard@10) 标准差 across test points | > 0.15 | 0.10-0.15 | < 0.10 |
| TRV 3-bin 分布（每 bin > 10%） | 3 个 bin 均 > 10% | 2 个 bin > 10% | 仅 1 个 bin 主导 (> 80%) |
| 3-seed 一致性（TRV 排名的跨 seed Spearman） | > 0.7 | 0.5-0.7 | < 0.5 |
| Spearman(SI, TRV) | > 0.5 | 0.3-0.5 | < 0.3 |
| Spearman(SI, TRV \| \|\|grad\|\|) 偏相关 | > 0.3 | 0.15-0.3 | < 0.15 (SI 仅为梯度 norm 代理) |
| TRV-high vs TRV-low LDS 差异 | Cohen's d > 0.5 | 0.3-0.5 | < 0.3 (稳定 ≠ 正确) |
| EK-FAC vs K-FAC avg Jaccard@10 | < 0.7 (足够差异) | 0.7-0.85 | > 0.85 (差异不够) |

**核心 Pass 判定**：TRV 标准差 > 0.10 **且** 3-bin 至少 2 个 > 10% **且** 3-seed 一致性 > 0.5。

**辅助判定（影响 Phase 2 可信度但不 block Phase 1）**：梯度 norm 偏相关 > 0.15 **且** TRV-LDS 差异 Cohen's d > 0.3。

### 3.4 时间预算

**6-8 GPU-hours 计算 + 3-5 天人工时间**（含 Step 0 代码可行性验证 1-2 天 + 代码适配、调试、分析结果 2-3 天）。

### 3.5 Fail 时的信息价值

**预设失败模式及其诊断特征**：

1. **TRV 分布退化（> 80% 同级）**：
   - **Failure signature**：Jaccard@10 直方图呈尖峰分布，标准差 < 0.05
   - **诊断意义**：Hessian 近似对 top-k 归因的影响是全局均匀的，per-test-point 诊断无附加价值。**但这本身是一个有价值的发现**——证明"Hessian 近似误差虽大（Hong et al. 证明了），但对 top-k 排名的影响在 test point 间高度一致"。
   - **后续方向**：收缩到 Phase 1 的纯描述性贡献（TRV 分布的经验特征化），放弃自适应融合。或转向 distributional TRV (A3)。

2. **3-seed 不一致（TRV 排名跨 seed Spearman < 0.3）**：
   - **Failure signature**：TRV 分布形状在不同 seed 间显著不同
   - **诊断意义**：TRV 的 test-point 差异主要来自训练随机性而非 test point 内在特征。这意味着 root cause 分析有误——差异不是因为"梯度与 Hessian 谱对齐不同"，而是因为"不同训练运行的 Hessian 谱本身不同"。
   - **后续方向**：转向 distributional TRV (A3)，用多 seed 方差作为不确定性指标。

3. **SI 与 TRV 相关性极低 (< 0.2)**：
   - **Failure signature**：SI 高的 test point 不对应 TRV 低的 test point
   - **诊断意义**：SI 的理论 Lipschitz bound 是 loose 的——归因稳定性的实际瓶颈不在 SI 捕获的谱放大效应，而可能在 Hessian 基方向 mismatch 等 SI 未建模的因素上。
   - **后续方向**：Phase 2 仍可用完整 TRV（不用 SI 代理），但计算成本不降。或需要开发新的廉价代理。

## 4. 元数据

- Gap 候选数量：4（归因可靠性量化缺失、IF-RepSim 融合缺乏原则性方法、TDA 评估框架系统性偏颇、per-test-point 归因质量诊断缺失）——其中归因可靠性量化缺失和 per-test-point 诊断缺失合并为最终 Gap
- 候选攻击角度数量：4（A1 选定，A2-A4 排除）
- 知识库引用：
  - **Gaps & Assumptions**: G-BHM1 (Hessian 层级仅 MLP 验证), H-BHM1 (特征值 mismatch 架构通用性), H-BHM5 (LDS 充分代理假设), G4 (attribution-influence misalignment), H-IF-LLM1 (RepSim 泛化性), H-IF-LLM3 (LoRA 低秩 = IF 障碍 vs 机会), H-RF1 (LDS 评估可靠性元问题), G-RF2 (缺替代评估方案)
  - **Cross-Paper Connections**: C54 (IF 测量维度分歧), C55 (IF vs RepSim 有效性边界), C56 (DDA = IF 修复叙事), C57 (低秩争议)
  - **Methods Bank**: #10 RepSim, #15 TRAK, ASTRA, EK-FAC IF (#7), Hessian 层级链分解框架 (2509.23437)
  - **Source Materials**: 2509.23437 (Hessian 层级), 2512.09103 (Natural W-TRAK / SI 理论), 2409.19998 (IF 失效 / RepSim), 2602.14869 (低相关性 = 互补), 2506.12965 (distributional TDA), 2505.23223 (Daunce: 扰动模型归因不确定性), 2509.26544 (BIF: 贝叶斯归因不确定性)
