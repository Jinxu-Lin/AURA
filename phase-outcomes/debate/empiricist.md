## [Empiricist] 实验主义者视角

### 否证条件

❌ **主要否证条件 1 (Phase 1 TRV)**：如果在 CIFAR-10/ResNet-18 上计算 1000 个 test point 的 TRV 分布后，发现 >80% 的 test point 具有相同的 TRV 等级（即 TRV 几乎不因 test point 而异），则 H1 不成立，Phase 1 的核心前提——"归因敏感性因 test point 而异且可预测"——坍塌。阈值依据：如果 TRV 几乎是常数，那么它就不能作为自适应融合的路由信号，整个 AURA 框架退化为 static ensemble，失去独特价值。

❌ **主要否证条件 2 (Phase 2 RA-TDA)**：如果 RA-TDA 在 DATE-LM 三个任务上的 LDS 相对于 naive ensemble（简单平均 IF + RepSim）的提升 < 2% absolute，或 Cohen's d < 0.5（基于 5 次不同 seed 运行），则 TRV-guided fusion 相比无引导融合无显著优势。阈值依据：TDA 领域 LDS 的 run-to-run std 通常在 1-2%（参考 TRAK 论文的 variance 报告），2% absolute 提升对应 ~1 std，Cohen's d = 0.5 是中等效应量的下限。

❌ **早期信号否证条件**：在 CIFAR-10 pilot（见下文）中，如果 6 个 Hessian 近似等级之间的 Jaccard@10 差异矩阵中，相邻等级间的平均 Jaccard@10 > 0.85（即不同近似等级的归因结果高度相似），则 Hessian 近似质量对归因的影响本身就不够大，TRV 的诊断价值有限。可在 pilot 的 100 个 test point 上（~0.5 GPU-hour 内）观察到。

### 最小 Pilot 设计

🔬 **实验内容**：在 CIFAR-10 / ResNet-18 上，选取 100 个 test point（分层采样：50 个 high-confidence + 50 个 low-confidence），对每个 test point 计算 6 个 Hessian 近似等级（H, GGN, Block-GGN, EK-FAC, K-FAC, Identity）下的 top-10 attribution，产出 Jaccard@10 矩阵。复用 2509.23437 的基础设施。预计 0.5-1 GPU-hour。

📊 **核心测量量**：
1. **Jaccard@10 降级曲线**：对每个 test point，以 exact H 为 ground truth，绘制各近似等级的 Jaccard@10。如果曲线在 test point 间呈现高方差（std > 0.15），则 TRV per-point 分异成立；如果方差低（std < 0.05），则 TRV 作为路由信号无意义。
2. **TRV 等级分布**：100 个 test point 的 TRV 等级直方图。需要至少 3 个等级有 >10% 的 test point 占比，否则 TRV 缺乏区分度。

⚠️ **自我欺骗风险**：
- **"可能是模型太小/数据集太简单"**——这是最大风险。如果 ResNet-18/CIFAR-10 上 TRV 分异不明显，研究者会说"大模型上肯定更明显"。但注意：如果在小规模上核心现象都不存在，大规模上的额外 confounders（更大 noise、更高计算成本）只会让信号更难观测，不会更容易。
- **"Jaccard@10 阈值 0.5 可能不对，换个 k 或换个 metric 就好了"**——不断调整 TRV 的操作定义来匹配想要的结果，属于 post-hoc metric gaming。TRV 的定义应该在 pilot 前固定。
- **"Exact Hessian 计算不够精确"**——如果连 exact H 都不可靠，那整个 TDA 敏感性分析的 ground truth 就不存在，问题比 AURA 能解决的更深层。

### Confounders 审查

- **Training variance confound**：ResNet-18 的不同训练 seed 会产生不同的 Hessian landscape，导致 TRV 分布变化。这可能让 TRV 看起来有 per-point 变异性，但实际只是 training randomness。
  — **控制方法**：在 3 个不同 training seed 的 ResNet-18 上分别计算 TRV，检查 TRV 等级的 rank correlation（Spearman ρ > 0.6 across seeds 才可接受）。如果 TRV 等级在不同 seed 间不稳定，则 TRV 本身就不是一个可靠的诊断信号。

- **Compute budget confound (Phase 2)**：RA-TDA 的 τ_RA(z) 需要同时计算 IF score 和 RepSim score，计算量约为单一方法的 2x。如果 RA-TDA 优于 IF-only 或 RepSim-only，可能仅仅是因为使用了更多信息/计算，而非 TRV 引导的功劳。
  — **控制方法**：必须与 naive ensemble（无 TRV 引导的简单平均 IF + RepSim）做严格对比。RA-TDA 必须显著优于 naive ensemble 才能 claim TRV 引导有贡献。此外，与 compute-matched TRAK（给 TRAK 同样多的 random projection 预算）对比。

- **SI 作为 TRV 代理的循环论证风险 (H4)**：SI(z) 基于 feature covariance Q 的条件数 κ 计算。如果 κ 本身与 TRV 的 correlation 是通过精心选择的 SI 公式 calibrated 出来的，则 SI → TRV 的映射可能是 overfit 到 CIFAR-10/ResNet-18 上的。
  — **控制方法**：SI → TRV 的映射关系必须在一个数据集/模型上拟合（如 CIFAR-10/ResNet-18），在另一个数据集/模型上验证（如 CIFAR-100/ResNet-50 或 GPT-2）。禁止 in-sample validation。

- **Hessian 近似误差与数据分布的交互 (H5)**：如果 Hessian 近似误差在 OOD test point 上系统性增大（而非与分布偏移独立），那么 TRV 与 OOD detection 高度相关，AURA 的贡献退化为"一种 OOD detector"而非"归因诊断工具"。
  — **控制方法**：在 pilot 中额外加入 20 个 OOD test point（如 SVHN 图片用 CIFAR-10 模型评估），检查 TRV 与 OOD score 的 correlation。如果 Spearman ρ > 0.8，则需要重新审视 AURA 的独立贡献。

### 评估协议完整性

**Benchmark/Metric**：LDS 和 Jaccard@k 是 TDA 领域标准 metric，DATE-LM 是新兴但被接受的 benchmark。但存在 **metric gaming 风险**：LDS 衡量的是 linear fit quality，RA-TDA 的加权融合公式本身就是线性形式 `s_rep + λ·r`，天然有利于 LDS metric。建议补充非线性 metric（如 top-k precision for data removal、counterfactual evaluation）来检验是否真正改善了归因质量而非仅优化了线性拟合。

**统计严谨性**：当前设计未提及多次运行。要求：Phase 1 在 3 个 training seed 上运行；Phase 2 在 5 个 seed 上运行，报告 mean ± std，并做 paired t-test 或 Wilcoxon signed-rank test（n=5 时推荐非参数检验）。DATE-LM 的三个任务应分别报告，不能只报告平均值。

**Ablation 结构**：需要以下关键 ablation：
1. **TRV 引导 vs 无引导**：RA-TDA vs naive ensemble（固定 λ vs TRV-adaptive λ）
2. **SI 代理 vs exact TRV**：用 SI(z) 近似 TRV 时的性能损失量化
3. **λ(TRV) 函数形式**：线性 vs 阶梯函数 vs learned mapping
4. **RepSim 选择**：不同 representation layer 的 RepSim 对融合效果的影响

**Cross-dataset 要求**：当前设计覆盖了 CIFAR-10（CV, small）和 GPT-2/Llama-7B（NLP, large）。但 CV 侧仅有 CIFAR-10 太弱——ResNet-18/CIFAR-10 是 TDA 领域的"MNIST"，几乎所有方法在这里都 work。**必须至少增加一个 CV 中规模 benchmark**（如 CIFAR-100/ResNet-50 或 ImageNet subset/ViT）。NLP 侧 DATE-LM 覆盖面可接受，但需注意 Llama-7B 的 exact Hessian 不可计算——Phase 1 的 TRV ground truth 如何建立？这是一个 **未解决的实验设计缺口**：对于大模型，TRV 的 ground truth 无法获取，SI 作为 TRV 代理的验证 loop 断裂。建议在 GPT-2 small (124M) 上做 TRV ground truth 验证，然后在 Llama-7B 上仅验证 SI → downstream performance 的端到端关系。
