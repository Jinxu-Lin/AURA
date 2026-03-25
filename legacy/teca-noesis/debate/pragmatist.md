# Pragmatist Debate — TECA

## 立项判断五项

### 1. 最强继续理由

所有组件都有现成开源实现（ROME 库、TRAK 库、LogIX、CounterFact），GPT-2-XL 可在单卡上跑，核心指标 TECS 就是一个余弦相似度——实现成本极低。如果 TECS 真的能区分好归因和坏归因，这是一个低成本高杠杆的贡献。

### 2. 最危险失败点

**假设 1 直接不成立的概率很高。** TDA 梯度方向（训练数据对模型参数的影响方向）和 ROME 编辑方向（rank-1 update 强制注入新知识的方向）在参数空间中没有理由对齐。ROME 的编辑方向是通过求解一个约束最小二乘问题得到的人工构造向量，不是自然学习过程的产物。TDA 梯度反映的是"如果去掉这条数据，参数怎么变"——这两个方向为什么要一致？一个是删除信号，一个是注入信号。

如果 pilot 跑出来 TECS 在所有方法上都接近 0（随机水平），整个项目就死了，没有任何补救空间。

### 3. 被施压的核心假设

**假设 1（TDA 梯度方向与 ROME 编辑方向对齐）** 和 **假设 2（ROME rank-1 update 反映知识存储位置）** 联合构成致命依赖。

假设 2 的问题：Meng et al. 的因果追踪实验显示 MLP 中间层是关键的，但 ROME 的 rank-1 update 是在这些层上做最小扰动注入，注入方向未必等于自然存储方向。这就像知道钥匙在抽屉里，但你用撬棍撬开——撬棍的方向不等于钥匙的形状。

假设 1 更根本：即使 ROME 方向"正确"，TDA 梯度也未必对齐。影响函数计算的是 loss 对参数的二阶近似，这个方向受 Hessian 的条件数、近似质量、数据集大小等因素严重影响。TRAK 用随机投影进一步扭曲了方向信息。

### 4. 关键证据要求

在投入任何正式实验之前，必须用 **1 个事实 + GPT-2-XL + ROME + 1 种 TDA 方法** 跑一个 sanity check：

- 取 CounterFact 中 1 个事实
- 跑 ROME 编辑，提取 Δθ_E^{l*}
- 用最简单的 gradient dot product（不是 IF，不是 TRAK，就是 raw gradient）计算训练数据的归因方向
- 看 top-10 归因数据的梯度方向与编辑方向的余弦分布
- 如果 top-10 和 random 的分布没有统计显著差异，**立刻停止**

这个 sanity check 不到 2 小时就能完成。

### 5. 本视角建议

**Hold** — 在 2 小时 sanity check 通过之前不应立项。核心假设太弱，没有任何先验理由相信对齐会发生。但实现成本低，如果 sanity check 过了，可以立刻转 Go。

---

## 完整务实者输出

### 一、工程组件拆解

| 组件 | 开源实现 | 状态 | 集成难度 | 风险 |
|------|---------|------|---------|------|
| ROME/MEMIT | `rome` (GitHub: kmeng01/rome) | 成熟，GPT-2-XL/GPT-J 测试充分 | 低 | 低——直接 pip install |
| TRAK | `traker` (GitHub: MadryLab/trak) | 成熟，但主要测试在分类模型上 | 中 | **高——LM 上的 TRAK 需要自定义 modelout function，官方示例有限** |
| LoGra (LogIX) | `logix` (GitHub: logix-project/logix) | 较新，文档不完整 | 中-高 | 中——API 可能变动，LM 支持不确定 |
| IF (EK-FAC) | 无现成库 | 需自行实现 | **高** | **高——EK-FAC 实现复杂，Hessian 近似在 Transformer 上不稳定** |
| RepSim | 自行实现 | 简单（forward pass + cosine） | 低 | 低 |
| BM25 | `rank_bm25` | 成熟 | 低 | 低 |
| CounterFact | `rome` 库自带 | 成熟 | 低 | 低 |
| GPT-2-XL | HuggingFace | 成熟 | 低 | 低 |
| GPT-J-6B | HuggingFace | 成熟 | 低 | 中——显存需 ~24GB，需 A100 或量化 |
| LDS (重训练) | 自行实现 | 需大量算力 | **高** | **高——GPT-2-XL 重训练 N 次，算力爆炸** |

**关键判断**：6 种归因方法中，真正能快速跑起来的只有 RepSim 和 BM25。TRAK 需要适配，LoGra 需要验证，IF(EK-FAC) 基本需要从头写。Phase 2 的 LDS 是算力黑洞。

### 二、最小 Pilot 设计（≤1 GPU-day）

**目标**：验证假设 1 是否有最基本的信号。

**设计**：

```
模型: GPT-2-XL (1.5B, 单 A100 可跑)
数据: CounterFact 前 10 个事实
编辑: ROME, 提取每个事实的 Δθ_E^{l*} (l* = layer 17, ROME 默认)
归因:
  - Raw gradient (最简单, 无近似误差)
  - RepSim (forward-only, 快)
  - BM25 (baseline, 无 GPU)
候选训练数据: WikiText-103 中与事实相关的 top-1000 段落 (BM25 预筛)
指标:
  - cos(Δθ_E^{l*}, g_train^{l*}) 对 top-K vs random 的分布差异
  - Mann-Whitney U test, p < 0.05
```

**时间预估**：
- ROME 编辑 10 个事实: ~10 min
- Raw gradient 计算 (10 facts × 1000 candidates): ~2-4 hours (单卡)
- RepSim: ~30 min
- BM25: ~5 min
- 分析: ~1 hour

**总计: ~0.5 GPU-day**

**Kill criteria**：
- 如果 raw gradient 的 TECS top-10 vs random 无显著差异 (p > 0.1) → **Stop**
- 如果 raw gradient 有信号但 RepSim 没有 → 有趣，继续但缩小范围
- 如果两者都有信号 → Go，扩展到 TRAK/LoGra

### 三、工程陷阱

1. **TRAK 在 LM 上的适配陷阱**：TRAK 库设计给分类模型（固定输出维度）。LM 的 next-token prediction 输出空间是整个 vocab，需要定义 `modelout_fn` 聚焦到特定 token position。这不是简单的配置问题，需要理解 TRAK 内部的 JL projection 是怎么作用在 gradient 上的。预计 1-2 天调试。

2. **EK-FAC 实现深坑**：EK-FAC 需要对每层的 Kronecker 因子做特征分解，对 Transformer 的 attention 层和 MLP 层需要不同处理。现有论文（Grosse et al. 2023）的实现未开源。保守估计 1-2 周实现 + 调试。**建议：Phase 1 直接砍掉 IF(EK-FAC)，用 raw gradient 代替。**

3. **LDS 算力爆炸**：Phase 2 需要对 GPT-2-XL 做 leave-k-out 重训练。即使只重训 50 个子集，每次训练 GPT-2-XL 需要 ~2-4 GPU-days（取决于数据量和 epochs）。总计 100-200 GPU-days。**这在学术预算下不现实。建议：用 last-layer 重训练或 linear probing 替代 full retrain 作为 LDS 近似。**

4. **CounterFact 训练数据归属问题**：CounterFact 定义了"事实"（如 "The Eiffel Tower is in Paris"），但 GPT-2 的训练数据是 WebText，不是公开可查的。你需要用 BM25 在某个代理语料库（如 Wikipedia）上找"相关训练数据"，但这不是真正的训练数据。这个 gap 会削弱 TECS 的解释力。

5. **余弦相似度的维度诅咒**：ROME 编辑层 l* 的参数维度约 d_model × 4*d_model = 1600 × 6400 ≈ 10M。在 10M 维空间中，任意两个向量的余弦相似度趋近 0。你需要验证信号是否真的能从噪声中脱颖而出，还是被高维稀释了。**建议：同时计算子空间（top-k SVD）上的投影余弦。**

6. **ROME 编辑方向的非唯一性**：ROME 的 rank-1 update 取决于编辑目标（"Paris" → "London"）。同一事实的不同编辑目标会产生不同的 Δθ。你测量的是"编辑到特定目标的方向"，不是"知识存储方向"。这引入了一个未讨论的自由度。

### 四、综合预估

| 项目 | 预估 |
|------|------|
| Pilot (sanity check) | 0.5 GPU-day, 1 人-天 |
| Phase 1 (砍掉 EK-FAC) | 5-7 GPU-days, 2 人-周 |
| Phase 1 (含 EK-FAC) | 10-15 GPU-days, 4-5 人-周 |
| Phase 2 (LDS, full retrain) | 100-200 GPU-days — **不推荐** |
| Phase 2 (LDS, linear probe) | 5-10 GPU-days, 1 人-周 |
| Phase 3 (分解分析) | 1-2 GPU-days, 0.5 人-周 |
| Phase 4 (MEMIT) | 3-5 GPU-days, 1 人-周 |
| **总计（务实版）** | **15-25 GPU-days, 5-6 人-周** |
| **总计（原始计划）** | **120-230 GPU-days, 10+ 人-周** |

**务实建议**：
1. 立刻做 2 小时 sanity check（raw gradient + 10 facts）
2. 如果通过，砍掉 EK-FAC，用 raw gradient / TRAK / LoGra / RepSim / BM25 五种方法
3. Phase 2 用 linear probing 替代 full retrain
4. GPT-J-6B 作为 optional extension，不作为 core contribution
