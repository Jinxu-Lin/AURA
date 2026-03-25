# Empiricist Debate — TECA

## 五项立项判断

### 1. 最强继续理由
TECS 的核心测量量（余弦相似度）在数学上是完全明确的，实验可以在 GPU-hour 量级完成 pilot，且否证条件清晰：random baseline 是否与梯度方法有统计可分的差异。这是一个"结果出来就能下结论"的实验，不存在"还需要更多实验才能判断"的模糊地带。

### 2. 最危险失败点
高维参数空间中余弦相似度的统计特性。GPT-2-XL 在编辑层 l* 的参数维度约为 d ~ 10^6-10^7。在这个维度下，任意两个随机向量的余弦期望为 0，标准差约为 1/sqrt(d) ~ 10^{-3}。如果 TECS 的信号量级也在 10^{-3}，那么即使"统计显著"也没有实际意义——你在测量噪声的结构而非知识的对齐。必须预先定义 practical significance threshold，而不仅仅是 statistical significance。

### 3. 被施压的核心假设
**假设 4**（cosine similarity 是合理度量）。g_M 是 top-k 训练样本梯度的**算术平均**，这个聚合操作假设归因信号在参数空间中是同一方向上的。但如果 k 个训练样本的梯度方向分散（高 angular variance），平均后的向量模长接近 0，余弦相似度变成噪声除以噪声。更根本的问题：即使 TDA 和 ROME 操作同一子空间，cosine 只捕捉一维方向对齐，而子空间对齐需要 principal angle 或 CKA。

### 4. 关键证据要求
在写任何代码之前，先做纯数学验证：(1) 计算编辑层 l* 的参数维度 d；(2) 对 10 个事实，计算 top-k 训练样本梯度的 angular variance（pairwise cosine 的均值和标准差）；(3) 如果 angular variance 很高（pairwise cosine < 0.1），则聚合梯度 g_M 本身就不是一个有意义的"方向"，整个 TECS 框架失效。这个检查 < 30 分钟 GPU 时间。

### 5. 本视角建议
**Go with focus** — 实验设计清晰、否证条件明确、pilot 成本低。但必须在 pilot 中增加梯度 angular variance 检查，并预定义 practical significance threshold（不仅仅是 p-value）。

---

## [Empiricist] 实验主义者视角

### 否证条件

**主要否证条件 1**：如果在 GPT-2-XL 上、500 个 CounterFact 事实中，最强梯度方法（IF with EK-FAC）的 mean TECS 与 random baseline 的差异 < 0.01（绝对值），或两组分布的 Cohen's d < 0.5，则 TECS 不具备区分能力，方向不成立。阈值依据：高维空间中 random cosine 的 std ~ 1/sqrt(d)，d ~ 10^6 时 std ~ 10^{-3}，0.01 约为 10 倍标准差，已是宽松阈值。

**主要否证条件 2**：如果 Phase 2 中 Spearman(TECS_ranking, LDS_ranking) < 0.5（不是提案中的 0.7），则 TECS 作为 LDS 替代物无效。0.5 是底线——低于 0.5 意味着两个指标基本独立，TECS 测量的不是归因质量。

**早期信号否证条件**：如果 pilot 中（10 个事实），top-k 训练样本梯度的 pairwise cosine 均值 < 0.05（即梯度方向近乎正交），则聚合梯度 g_M 不是有意义的"方向"表示，TECS 框架的数学基础不成立。这在 pilot 的前 30 分钟即可观察到。

### 最小 Pilot 设计

**实验内容**：GPT-2-XL，从 CounterFact 随机选 10 个事实，ROME 编辑，仅用 IF（EK-FAC 近似）计算 top-50 训练样本。三个测量：(a) TECS 值分布，(b) top-50 梯度的 pairwise cosine 分布（angular variance），(c) 随机选 50 个训练样本作为 random baseline 的 TECS。

**核心测量量**：
- TECS(IF) vs TECS(random) 的分离度（Cohen's d）
- top-k 梯度 pairwise cosine（如果 < 0.05，聚合无意义，框架失效）
- TECS 在 10 个事实间的 variance（如果 coefficient of variation > 2，信号不稳定）

**自我欺骗风险**：
- "10 个事实太少，500 个才有 power" — 但如果效应真实存在，10 个事实足以看到方向性信号（Cohen's d > 0.5）；如果需要 500 个才勉强显著，效应量太小没有实用价值
- "EK-FAC 近似不够好，需要精确 Hessian" — 如果核心假设成立，近似 IF 也应该有正方向；如果只有精确 IF 才有信号而所有近似方法都没有，TECS 的实用性为零
- "k=50 不是最优的 top-k" — 如果结论对 k 值敏感（k=50 vs k=100 结果反转），说明信号不鲁棒

### Confounders 审查

- **ROME 编辑失败**：CounterFact 中约 5-15% 的事实 ROME 编辑失败（edit success rate < 100%）。如果这些失败事实的 TECS 也低，会人为夸大 TECS 与编辑质量的相关性。— 控制方法：分两组报告（编辑成功 vs 编辑失败），主结果仅基于编辑成功的事实；编辑成功定义为 paraphrase prompt 下 target token probability > 0.5。

- **训练数据覆盖偏差**：CounterFact 的事实在训练数据中的出现频率差异巨大。高频事实可能有更多相关训练样本，TDA 归因信号更强，TECS 也更高——但这反映的是数据频率而非方法质量。— 控制方法：按事实在训练数据中的 BM25 检索命中数分桶（高/中/低频），检查 TECS 在各桶的分布是否一致。

- **层选择偏差**：ROME 的 causal tracing 选出的 l* 可能不是梯度信号最强的层。在 l* 上比较可能系统性地 favor ROME 方向。— 控制方法：除 l* 外，在 l*±2 的相邻层也计算 TECS profile，报告跨层趋势。如果信号仅在 l* 出现而相邻层为零，这过于 suspiciously clean。

- **k 值选择偏差**：g_M = (1/k) Σ ∇L(z_i) 中 k 的选择直接影响 TECS。小 k 可能碰巧选到与编辑方向对齐的样本；大 k 平均掉信号。— 控制方法：在 k ∈ {10, 25, 50, 100, 200} 上做 sensitivity analysis，报告 TECS vs k 的曲线。如果存在清晰的 peak 后 decay 模式，说明信号真实但集中在少数样本；如果 monotonically decreasing，可能是 cherry-picking 效应。

### 评估协议完整性

**Benchmark/Metric**：CounterFact 作为 benchmark 合理（ROME 标准数据集，社区认可）。但 TECS 本身是一个新 metric，需要额外验证其 construct validity——仅靠 TECS 与 LDS 的相关性不够，应该增加 known-answer test（人工构造梯度方向已知的 toy case 验证 TECS 计算正确性）。

**统计严谨性**：
- 必须报告 500 个事实上的 mean ± std，不接受仅报告 mean
- 方法间比较使用 paired test（Wilcoxon signed-rank，因事实配对），不接受 unpaired t-test
- 多重比较校正：6 种 TDA 方法的 pairwise 比较需 Bonferroni 或 FDR 校正
- Phase 2 的 Spearman 相关需报告 95% confidence interval（bootstrap），不接受仅报告点估计
- 不同 random seed 的 variance：ROME 编辑和 TDA 归因分别在 3 个 seed 上运行，报告 seed-level variance

**Ablation 结构**：需要以下拆解——
- (A1) TECS 在编辑层 l* vs 其他层的对比（验证层定位假设）
- (A2) 不同 k 值对 TECS 的影响曲线（验证 top-k 聚合合理性）
- (A3) 梯度聚合方式：mean vs weighted mean (按 IF score 加权) vs subspace projection（验证假设 4）
- (A4) ROME 编辑成功 vs 失败事实的 TECS 分布（验证编辑质量与 TECS 的关联）

**Cross-dataset 要求**：
- 主实验在 CounterFact 上进行
- 补充验证至少在一个其他事实性知识数据集上重复核心结果（如 zsRE 或 FEVER-derived facts），否则无法排除 CounterFact 特有的 artifacts
- 跨模型验证：GPT-2-XL 和 GPT-J-6B 至少两个模型，验证 TECS 的规律在模型间一致
