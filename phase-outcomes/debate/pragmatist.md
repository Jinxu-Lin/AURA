## [Pragmatist] 务实者视角

### 工程组件拆解

**Phase 0: Spectral Diagnosis**
- ✓ TRAK 特征提取 — `github.com/MadryLab/trak`，已有 CIFAR-10/ResNet-18 的标准 pipeline
- ✓ 条件数 κ 计算 — NumPy/SciPy SVD，协方差矩阵规模 = (特征维度)²，标准操作
- △ SI 计算 — 2512.09103 论文有公式但需确认是否有公开代码。若无，需自行实现 Q⁻¹ 的高效计算（Q 可能很大，需 Woodbury/CG）。估计 3-5 天改造/实现；改造点：TRAK codebase 不直接暴露 Q 矩阵，需要 hook 进内部状态提取协方差

**Phase 1: TRV 计算**
- △ Hessian 层级链 — 2509.23437 Turner 组代码。**关键风险**：论文声称实现了 H/GGN/Block-GGN/EK-FAC/K-FAC 完整层级，但需验证代码是否 (a) 公开可用，(b) 接口统一到可以循环调用不同近似，(c) 支持 ResNet-18 以外的模型。估计 5-7 天集成+调试；改造点：不同 Hessian 近似的实现可能散落在不同分支/脚本中，需要统一接口封装
- △ top-k 归因计算 — 对每个 Hessian 近似分别计算 iHVP → 影响分数 → 排序取 top-k。每种近似需要独立的 iHVP solver（显式逆/CG/K-FAC 逆等）。估计 3-5 天，因为不同近似的 iHVP 计算方式完全不同
- ✓ Jaccard@k 计算 — 纯集合操作，< 1 天
- ✓ Spearman 相关计算 — SciPy，trivial

**Phase 2: RA-TDA**
- ✓ RepSim (cosine similarity) — HuggingFace Transformers 中间层提取 + cosine，标准操作
- △ IF on LLM (GPT-2/Llama-7B) — 需要在 LoRA fine-tuned 模型上计算 influence functions。TRAK 有 LLM 支持但质量有限；Li et al. (2409.19998) 的代码需要确认公开程度。估计 7-10 天复现+适配；改造点：LoRA 参数空间上的 iHVP 计算、显存管理、gradient checkpointing 与 IF 计算的兼容性
- △ 残差回归 r(z) = s_if(z) - E[s_if | s_rep] — 需要设计条件期望估计器。最简单用分箱或核回归，但需要足够样本。估计 2-3 天
- △ lambda(TRV) 融合权重函数 — 需要设计函数形式（sigmoid? 分段线性?）+ 调参。估计 2-3 天
- ✗ DATE-LM benchmark 适配 — 2507.09424 是 2025 年 7 月的论文，benchmark 可能尚未完全公开或文档不全。需从零搭建 entity corruption 评估 pipeline。估计 5-7 天；原因：LLM factual attribution 评估涉及 counterfactual retraining，计算开销大且细节多
- ✗ Li et al. IF 失效场景复现 — 2409.19998 的有害数据识别实验设置（LoRA + 低信噪比），需完整复现其 baseline 才能公平对比。估计 5-7 天；原因：需要精确复现 LoRA 训练配置、毒化数据生成、多种 TDA 方法的计算

### 最小 Pilot 设计

**实验内容**：在 CIFAR-10 / ResNet-18 上，用 TRAK codebase 计算 500 个 test points 的 top-100 归因，分别使用 K-FAC 和 Identity (即 RepSim) 两个极端近似，计算 Jaccard@100 分布。如果 Jaccard 分布呈现明显的双峰（部分 test points 高重叠、部分低重叠），则 H1（"敏感性因 test point 而异"）得到初步验证。

**缩放策略**：
- 模型/数据：CIFAR-10 / ResNet-18（与 2512.09103 和 2509.23437 同 benchmark，结果可直接对比）
- 只用两个 Hessian 近似极端（K-FAC vs Identity），不需要完整层级链
- 500 test points（非全量 10000），足以看到分布形状
- 这个 scale 足够，因为 2512.09103 已在同 scale 下观察到 κ ≈ 2.71×10⁵ 和显著的认证率差异

**所需已就位组件**：
- TRAK codebase（公开可用）
- CIFAR-10 预训练 ResNet-18 checkpoint（TRAK repo 自带或快速训练 ~30 min）
- K-FAC 近似下的归因计算能力（TRAK 内置 K-FAC 近似？需确认；若不内置，用 EK-FAC 替代，2509.23437 代码）

**预计算力**：4-8 GPU-hours (A100)。TRAK 在 CIFAR-10/ResNet-18 上的 featurize 步骤约 10 min；500 test points × 2 种近似 × top-100 归因排序，瓶颈在 iHVP 计算。

### 工程陷阱

**陷阱 1：Hessian 层级链的"统一接口"幻觉**。2509.23437 论文展示了 H/GGN/Block-GGN/EK-FAC/K-FAC 的完整实验，但其代码实现很可能是为每种近似分别写的脚本，不是一个统一的 `compute_ihvp(method="ekfac")` 接口。将 5-6 种 Hessian 近似封装成可循环调用的统一 API，需要深入理解每种方法的参数化方式（Block-GGN 需要指定 block 划分，EK-FAC 需要 Kronecker 因子缓存，K-FAC 需要 damping 调参）。这不是简单的 wrapper 工作，每种方法的 numerical stability 处理方式不同，damping/regularization 的选择直接影响归因结果，调不好会导致 TRV 测量的是 damping 差异而非 Hessian 近似误差。预估这一步会消耗 Phase 1 总工程时间的 40-50%。

**陷阱 2：Phase 2 的 LLM IF 计算是工程深水区**。在 GPT-2（124M 参数）上计算 influence function 本身就需要 careful engineering（gradient accumulation、mixed precision 下的数值稳定性、activation checkpointing 与 per-sample gradient 的冲突）。在 Llama-7B 上更是需要 multi-GPU + FSDP/DeepSpeed，per-sample gradient 的提取在分布式环境下有已知 edge cases（gradient accumulation boundary 问题）。TRAK 的 LLM 支持文档有限，可能需要大量 monkey-patching。**一个看似简单的 "用 TRAK 计算 Llama-7B 的 influence score" 可能单独就需要 2-3 周调通。**

**陷阱 3：RA-TDA 的 "DR 性质" 验证需要 careful experimental design**。声称 "即使一个组件失效，融合仍有效" 需要构造 IF 明确失效和 RepSim 明确失效的场景，且这两个场景不能是 cherry-picked。Li et al. 的 IF 失效场景是已知的，但 "RepSim 失效场景" 目前只有 DATE-LM 的 factual attribution 作为候选，这个 benchmark 是否真的让 RepSim 失效（而非只是表现较差）需要先验证。如果找不到清晰的 RepSim 失效场景，DR 性质的一半就无法验证，论文的核心 claim 会受质疑。

**陷阱 4：H4 假设（SI 是 TRV 有效代理）的失败后果被低估**。如果 SI 和 TRV 的 Spearman 相关低于 0.5，Phase 0 的快速预检就失效了，AURA 的实用性叙事（"用 SI 做免费预检，避免昂贵的 TRV 计算"）要重写。但更关键的是：如果 SI（理论界）和 TRV（经验度量）不相关，说明 2512.09103 的理论对 Hessian 近似扰动可能不适用，这会动摇 AURA 与 2512.09103 之间的理论连接——而这个连接是 positioning 的核心。

### 综合预估

**到第一个有意义结果（Phase 1 on CIFAR-10, TRV 分布 + SI 相关性验证）**：
- 日历时间：3-4 周（含 1 周 Hessian 层级代码集成调试 + 1 周 TRV 计算 + 1 周分析和 SI 对比 + buffer）
- 算力：20-40 GPU-hours (A100)。CIFAR-10/ResNet-18 本身计算量小，但 1000 test points × 6 种 Hessian 近似 × top-k 归因 = 6000 次 iHVP 求解，每次 ~10-30 秒（取决于近似方法）
- 主要工程风险：2509.23437 代码的可用性和集成难度。如果代码不可用或接口不统一，Phase 1 的基础设施搭建时间将翻倍

**到完整论文结果（Phase 1 + Phase 2 + 理论分析）**：
- 日历时间：3-4 个月
- 算力：200-400 GPU-hours (A100)。Phase 2 的 LLM 实验（GPT-2 fine-tuning + influence 计算 + Llama-7B）是主要消耗
- 主要工程风险：Phase 2 的 LLM IF 计算工程量。如果 TRAK 在 Llama-7B 上不稳定，可能需要降级到 GPT-2 only，这会削弱论文的 scalability story

### 可行性总评

**Phase 1 可行性：中高**。核心依赖（TRAK codebase + 2509.23437 Hessian 代码）是公开的或可预期公开的基础设施，CIFAR-10/ResNet-18 的计算量可控。主要不确定性在代码集成。

**Phase 2 可行性：中低**。LLM 上的 IF 计算是已知的工程难题，DR 融合的实验设计需要同时构造两种互补失效场景，DATE-LM benchmark 的成熟度存疑。建议将 Phase 2 的实验 scale 从 Llama-7B 降级到 GPT-2 only 作为保底方案。

**最大务实建议**：采用"Phase 1 先行 + Phase 2 条件推进"的策略。如果 Phase 1 的 TRV 分布没有呈现预期的 test-point-specific 变异性（即 H1 失败），则 Phase 2 的 TRV-guided 融合没有基础，应及时止损。Phase 1 本身作为 "TDA robustness diagnostic tool" 已经有独立发表价值（诊断论文也是有用的贡献），不必绑定 Phase 2。
