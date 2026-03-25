# Interdisciplinary Debate — TECA

## 五项立项判断

### 1. 最强继续理由

"用方法 A 验证方法 B"的结构在科学史中反复出现且极其有力：地震波层析成像 (seismic tomography) 和宇宙学 CMB 分析都依赖多条独立观测通道的几何一致性来重建不可直接观测的内部结构。TECA 本质上是在做同样的事——用两种不同的"参数空间探针"（TDA 归因 vs 模型编辑）来三角测量 LLM 内部的知识表征。如果两条独立探针在参数空间中收敛到同一子空间，这比任何单一方法的自我验证都更有说服力。这个**结构性思路**是正确的，而且在 TDA 领域尚无人尝试。

### 2. 最危险失败点

**测量仪器与被测对象的耦合问题**——量子力学中的 observer effect 的经典翻版。ROME 的编辑不是被动观测，而是主动干预：它改变了参数空间的结构。TDA 的梯度也不是被动读数，而是依赖于模型的当前参数配置。两者都在"扰动系统来推断结构"，但各自的扰动可能系统性地偏向参数空间中的同一类结构特征（例如高曲率方向、大梯度范数层），而这种偏向与"知识存储"无关。这是一个 confounding by measurement methodology 的问题，在实验物理中被称为 systematic bias。

更精确地说：如果 TDA 梯度和 ROME 编辑方向都倾向于对齐到 loss landscape 中梯度范数最大的方向（因为 loss landscape 的几何在两种操作中都起支配作用），那么 TECS > 0 反映的是 loss landscape 的局部几何特性，而非知识的空间定位。这与"两个温度计都受磁场干扰时读数一致但都不准"是同构的。

### 3. 被施压的核心假设

**假设 2（ROME 反映知识位置）+ 一个隐含假设：知识以局部化方式存储。**

从神经科学视角，这对应 localizationism vs. distributed representation 的百年争论。大脑中的 Broca's area / Wernicke's area 是局部化的经典案例，但现代神经科学已经证明即使这些区域的功能也是分布式网络的一部分。对 LLM 做类比：ROME 的 causal tracing 找到的"知识层"可能类似于 Broca's area——它是信息处理的瓶颈（bottleneck），破坏它会产生明显效果，但这不意味着知识"存储"在那里。

信息论给出了更精确的框架：Shannon 的信道容量定理告诉我们，信息可以在冗余编码（distributed）中鲁棒存储。如果 LLM 的知识采用分布式冗余编码，那么任何试图在单层/单方向上测量"知识位置"的方法都注定只能捕捉到投影 (projection)，而 TECS 测量的是两个投影之间的一致性——可能有信号但解释力有限。

### 4. 关键证据要求

1. **Confound 控制实验**（类比实验物理中的 control experiment）：计算一个"null TECS"——不对特定事实做编辑，而是对随机方向做 rank-1 update（保持相同范数），然后计算与 TDA 梯度的余弦。如果 null TECS 显著低于 real TECS，说明信号不是来自 loss landscape 的全局几何偏好。这是排除 systematic bias 的最低要求。

2. **信息论检验**：用互信息 (MI) 替代余弦相似度。将 $\Delta\theta$ 和 $g_M$ 各自投影到前 r 个主成分，计算这两组投影之间的 MI（用 k-NN MI estimator，如 KSG estimator）。MI 能捕捉非线性依赖关系，而余弦只捕捉线性对齐。如果 MI 显著但 cosine 不显著，说明关系存在但不是线性的。

3. **跨层 profile 的控制论解读**：计算所有层的 TECS profile（不仅仅 l*），观察信号是否在 causal tracing 识别的层达到峰值。这类似于控制论中的 transfer function 分析——在哪个"频率"（层）上两个信号的相干性最强。

### 5. 本视角建议

**Go with focus** — 前提是在 pilot 中增加 null TECS（随机方向编辑）控制实验。

---

## 跨学科完整分析

### I. 跨域对应物："用方法 A 验证方法 B"的结构

TECA 的核心结构——用两条独立通道交叉验证不可直接观测的内部结构——在多个领域有成熟的对应物：

**1. 地球物理学：地震层析成像 (Seismic Tomography)**

地球内部结构不可直接观测。地震学家用两种独立信号三角测量：P 波速度异常和 S 波速度异常。两者各自有噪声和偏差，但如果在同一空间位置两种波速都异常，则对地幔柱或俯冲带的定位更可信。

TECA 的结构完全类似：TDA 梯度和 ROME 编辑方向是两种"波"，参数空间是"地球内部"，知识存储位置是"地幔柱"。

**关键差异与教训**：地震层析成像成功的前提是两种波的物理机制已被独立验证（弹性波方程）。TECA 的两种"波"（TDA 和 ROME）的"物理机制"各自有争议——这是一个根本性的弱点。地震学通过 forward modeling（给定已知结构，预测波形，再反演）来验证方法。TECA 应该做类似的 forward modeling：构造一个知识存储位置已知的 toy model（如在特定层注入事实），验证 TECS 能否恢复已知位置。

**2. 天文学：多信使天文学 (Multi-Messenger Astronomy)**

2017 年 LIGO 引力波信号 + Fermi 伽马射线暴的时空一致性，证实了中子星合并事件。单一信使各自可能有假阳性，但两者独立且在同一时空窗口共现，false positive 概率从 $p_1$ 和 $p_2$ 降到 $p_1 \times p_2$。

TECA 的逻辑类似：TDA 归因可能有假阳性（指向错误的训练样本），ROME 编辑可能有假阳性（编辑了错误的方向但碰巧成功），但两者在参数空间中指向同一子空间的概率——如果信号是假的——应该接近随机。

**关键差异与教训**：多信使天文学的成功依赖于每种信使的 false positive rate 已经被独立量化。TECA 尚未量化 TDA 和 ROME 各自的 false positive rate。如果两者的 false positive 由相同的 confound 驱动（如 loss landscape 几何），则独立性假设不成立，$p_1 \times p_2$ 的估计无效。

**3. 计量经济学：工具变量法 (Instrumental Variables)**

IV 用一个与混淆因子无关但与自变量相关的工具来估计因果效应。ROME 的编辑方向可以被类比为 TDA 归因方向的"工具变量"——它提供了一个不依赖重训练的替代信号。

**教训**：IV 方法要求工具满足排除性约束 (exclusion restriction)——工具只通过自变量影响因变量。在 TECA 中，这意味着 ROME 编辑方向必须只因为"知识位置"与 TDA 方向相关，而不通过其他路径（如 loss landscape 几何）相关。这个排除性约束目前没有任何辩护。

**4. 分子生物学：X 射线晶体学 + NMR 的交叉验证**

蛋白质结构测定中，X 射线晶体学和核磁共振 (NMR) 是两种独立方法。两者给出一致结构时，可信度大增。但两者各自都有系统性偏差（晶体学受晶体堆积效应影响，NMR 受溶液动力学影响），且某些情况下会系统性地给出一致但错误的结论。

**教训**：即使两种方法给出一致结果，仍然需要第三种完全独立的验证（如冷冻电镜 cryo-EM）来确认。TECA 不应将 TECS 一致性视为"证明"，只能视为"增加可信度的证据"。

### II. 未被利用的工具

**1. 信息几何 (Information Geometry)**

Fisher 信息矩阵 $F = E[\nabla \log p \cdot \nabla \log p^T]$ 定义了参数空间上的黎曼度量。在这个度量下，参数空间的距离和方向有统计意义——沿 $F$ 大特征值方向的移动对应模型输出的大变化，沿小特征值方向的移动几乎不影响输出。

**应用到 TECA**：不应在欧氏空间中计算 $\cos(\Delta\theta, g_M)$，而应在 Fisher 度量下计算。具体地：$\text{TECS}_{Fisher} = \frac{\Delta\theta^T F g_M}{\sqrt{\Delta\theta^T F \Delta\theta} \cdot \sqrt{g_M^T F g_M}}$。这个度量在统计上更有意义，因为它加权了"有信息的方向"。这也部分解决了 Theorist 指出的 $H^{-1}$ 缺失问题——Fisher 度量自然引入了曲率信息。

实际计算：用 EK-FAC 近似 $F$，或用对角 Fisher 近似（计算成本低），甚至只用 Adam 优化器的二阶矩估计作为对角 Fisher 的代理。

**2. 随机矩阵理论 (Random Matrix Theory)**

ROME 的 $\Delta W = v k^T$ 是 rank-1 矩阵，$g_M$ 是一个随机矩阵（多个梯度的平均）。Marchenko-Pastur 律描述了随机矩阵谱的分布。可以用 RMT 来：
- 推导 TECS 的精确 null distribution（不需要 permutation test）
- 判断 $g_M$ 的有效秩（如果有效秩很低，说明梯度方向确实集中在少数方向上，聚合是合理的）
- 检测 $g_M$ 的谱中是否有超出 Marchenko-Pastur 边界的"信号"特征值

**3. 拓扑数据分析 (Topological Data Analysis)**

（注意：此处的 TDA 是 Topological Data Analysis，不是 Training Data Attribution。）

不同于余弦相似度（线性度量），持久同调 (persistent homology) 可以捕捉参数空间中"知识"的拓扑结构。具体地：取 top-k 训练样本的梯度向量集合和 ROME 编辑方向，用 Vietoris-Rips 复形构建拓扑特征，比较两者的 persistence diagram 距离（Wasserstein distance 或 bottleneck distance）。如果两组向量生成相似的拓扑结构（如相同维度的"空洞"），这比余弦对齐提供更丰富的几何信息。

但实现成本高且解释性弱，建议作为 Phase 3 的探索性分析。

**4. 因果推断框架 (Potential Outcomes / do-calculus)**

ROME 编辑本质上是一个 do-intervention：$do(\theta_{l^*} = \theta_{l^*} + \Delta\theta)$。TDA 归因本质上是一个 counterfactual：$\theta_{-z}$ vs $\theta$（删除训练数据 $z$ 后的参数 vs 原始参数）。

Pearl 的 do-calculus 提供了判断"两个干预是否等价"的形式化工具。具体地，可以构建一个 SCM (Structural Causal Model)，其中训练数据 → 参数 → 模型输出是因果链。ROME 编辑在"参数"节点做 do-intervention，TDA 在"训练数据"节点做反事实推理。两者等价当且仅当因果链中没有 confounders 或 mediators 影响。

这个框架可以精确识别 TECS 失效的条件：当参数空间中存在非因果的相关结构（如正则化引入的全局偏好方向），两种操作都会偏向这些方向，导致虚假对齐。

### III. 盲点与教训

**盲点 1：生态学中的"指标陷阱"**

生态学中曾广泛使用物种多样性指数（Shannon index, Simpson index）来评估生态系统健康。后来发现：不同指数对同一生态系统给出矛盾的健康评估，原因是每个指数对"多样性"的数学定义不同，隐含了不同的生态学假设。

TECA 面临同样的风险：TECS 是对"对齐"的一种特定数学定义（余弦相似度），但"对齐"可以有多种合法定义（子空间角、互信息、CKA、Procrustes 距离），每种定义隐含不同假设。如果只报告余弦相似度而忽略其他度量，可能因为度量选择而得出误导性结论。

**教训**：至少报告 2-3 种不同的对齐度量，讨论它们的差异及其含义。

**盲点 2：量子测量中的"互补性原理"**

Bohr 的互补性原理指出，某些物理量（如位置和动量）不能同时精确测量。TDA 和 ROME 可能也存在类似的互补性：TDA 梯度反映的是"删除数据"的参数变化方向（类似动量——训练过程的方向），而 ROME 编辑反映的是"当前状态下修改输出"所需的最小扰动（类似位置——当前状态的局部几何）。这两者在数学上可能不是同一个量的两种测量，而是两个互补量的各自测量——测量精度存在不可逾越的权衡关系。

如果这种互补性存在，TECS 的上限不是 1.0 而是某个小于 1 的值，且这个值本身携带了关于参数空间几何的信息。

**盲点 3：社会科学中的"反身性" (Reflexivity)**

Soros 的反身性理论：市场参与者的行为改变市场，市场的变化又改变参与者的行为。在 TECA 中：如果研究者根据 TECS 调整 TDA 方法的设计（如选择使 TECS 最大化的超参数），TECS 就不再是"独立评估"，而变成了"优化目标"。这会导致 Goodhart's Law（当指标成为目标时，它不再是好的指标）。

**教训**：论文中应明确声明 TECS 不应作为 TDA 方法的优化目标，仅作为诊断工具。

### IV. 建议引入路径

**路径 1：Forward Modeling 先行（来自地球物理学）**

在跑真实实验之前，先构建一个 toy model 做 forward modeling：
- 训练一个 2-layer MLP 记忆 100 个 key-value 对
- 人工控制"知识"存储在哪一层、哪些参数
- 分别计算 ROME 式编辑方向和 IF 式归因方向
- 验证在知识位置已知的情况下，TECS 是否能恢复真实对齐度

这个 toy model 实验成本极低（CPU 上 1 小时），但能提供关于 TECS 行为的关键 ground truth——如果在 toy model 上 TECS 都不工作，在 GPT-2-XL 上更不可能工作。

**路径 2：Fisher 度量替代欧氏度量（来自信息几何）**

用对角 Fisher 信息对 TECS 做加权。实现成本低（只需对每个参数计算梯度平方的期望），但理论意义大幅提升——从"欧氏空间中的方向对齐"升级为"统计流形上的方向对齐"。建议在 pilot 中同时计算 TECS 和 TECS_Fisher，对比两者的区分能力。

**路径 3：随机矩阵理论推导 null distribution（来自 RMT）**

用 Marchenko-Pastur 律推导 $g_M$（作为随机梯度矩阵的平均）与固定 rank-1 矩阵 $\Delta\theta$ 之间余弦相似度的理论 null distribution。这比 permutation test 更优雅、更有统计力量，且可以精确量化"什么值的 TECS 是有意义的"。

**路径 4：因果推断框架形式化（来自 do-calculus）**

构建 TECA 的 SCM，明确列出独立性假设和排除性约束。这不会改变实验，但会大幅提升论文的理论深度，帮助审稿人理解为什么 TECS 的解释力有限（以及限制在哪里）。

### V. 综合判断

TECA 的"双通道交叉验证"结构是跨学科视角下最有力的论证——它在地震学、天文学、结构生物学中都有成功先例。但所有这些先例都有一个 TECA 尚不具备的条件：**每条通道的测量机制已被独立验证**。

最务实的引入路径是 **Forward Modeling**（路径 1）：在知识位置已知的 toy model 上验证 TECS 的行为。这既是 sanity check，也是理论论证的基石。如果 toy model 上 TECS 能恢复已知对齐，再扩展到 GPT-2-XL 才有意义。

**最终建议：Go with focus**——但 focus 的内容不仅是 pilot 中的 TECS 信号检测，还必须包括：(1) toy model forward modeling，(2) null TECS（随机方向编辑控制），(3) 至少一种非线性对齐度量（MI 或 CKA）作为余弦的 robustness check。这三项中任何一项的失败都应触发重新评估。
