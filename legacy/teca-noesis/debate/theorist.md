# Theorist Debate Output — TECA

## 立项判断（五项速答）

### 1. 最强继续理由

TECA 触及了一个真实的理论空白：TDA 评估全部依赖重训练 ground truth，而重训练本身是有噪声的。用模型编辑作为独立验证通道，这个**思路方向是正确的**——你在参数空间中有两个独立的操作（归因 vs 编辑），检验它们的几何一致性。这是一个合法的数学对象。

### 2. 最危险失败点

**g_M 不是 IF score，TECS 测量的不是你以为的东西。**

TECS 的"归因方向" $g_M = \frac{1}{k}\sum \nabla_\theta L(z_i)$ 是 top-k 样本的原始梯度平均。但 IF score 是 $-\nabla L(z_{test})^T H^{-1} \nabla L(z)$。两者之间差一个 $H^{-1}$。你用 IF/TRAK 选出 top-k 样本，然后丢掉 IF 的核心（Hessian 逆），只用原始梯度来构建方向向量。这等于说："我用精密仪器选了样本，然后用粗糙工具测量方向。" 这个 gap 没有理论辩护。

更严重的是：2409.19998 已经证明 LLM 上 IF 退化为梯度点积（因为阻尼 Hessian 逆趋向单位矩阵）。如果 $H^{-1} \approx \lambda^{-1} I$，那么 IF score $\propto \nabla L(z_{test})^T \nabla L(z)$，g_M 与 IF 方向的差异消失——但这同时意味着 IF 本身就是退化的，你在用两个都退化的工具互相验证。

### 3. 被施压的核心假设

**假设 1: TDA 梯度方向与 ROME 编辑方向对齐。** 支撑强度：弱。

理论上没有任何定理保证这一点。让我把"对齐"拆解成需要成立的子条件：

- (1a) 训练样本 z 对参数 $\theta_{l^*}$ 的影响方向（梯度）反映了 z 中知识的存储位置
- (1b) ROME 的 rank-1 update 方向反映了知识的存储位置
- (1c) "知识存储位置"是一个良定义的概念，两种操作指向同一个子空间

条件 (1a) 假设梯度 = 知识方向，但梯度是局部线性近似，在高维非凸 landscape 上这个等式不成立。条件 (1b) 依赖 ROME 的 causal tracing，但 causal tracing 找到的是"修改后效果最大的层"，不等于"知识存储的层"。条件 (1c) 假设存在一个共享的"知识子空间"，但这是一个未经证明的隐含假设。

### 4. 关键证据要求

1. **理论推导**：在什么条件下 $\cos(\Delta\theta_{ROME}, g_M) > 0$ 是可以从 IF 理论推出来的？需要写出完整的数学链条，明确需要哪些简化假设。如果推不出来，TECS 就是一个 ad hoc 度量。

2. **Null distribution 刻画**：高维空间中随机向量的余弦相似度分布已知（集中在 0 附近，方差 $\sim 1/d$），但 $\Delta\theta$ 和 $g_M$ 不是随机向量——它们都是结构化的、低秩的。需要推导 TECS 在 null hypothesis（归因与编辑无关）下的分布，否则无法判断观测到的 TECS 值是否显著。

3. **g_M vs IF 方向的理论辩护**：为什么用原始梯度聚合而不是 $H^{-1}$-weighted 聚合？如果理由是"计算成本"，那不是理论动机。如果理由是"Hessian 逆退化为单位矩阵"，那需要量化这个近似的误差界。

4. **rank-1 vs full gradient 的维度失配**：$\Delta\theta_{ROME} = \Lambda (C^{-1}k)^T$ 是一个 rank-1 矩阵（外积），$g_M$ 是 $\frac{1}{k}\sum \nabla_{\theta_{l^*}} L(z_i)$，也是一个矩阵（对 $W_{l^*}$ 的梯度）。余弦相似度需要把它们展平为向量。但 rank-1 矩阵展平后只有一个自由方向——你本质上在测量 $g_M$ 在这个一维子空间上的投影。这个投影有多少信息量？需要定量分析。

### 5. 本视角建议

**Hold.**

核心想法有理论吸引力，但数学基础有三个明确欠账：(a) g_M 与 IF 方向的 gap 无辩护；(b) TECS 的统计显著性无 null distribution；(c) 对齐假设无理论保证。任何一个都可能让结果不可解释。建议在继续之前补上 (a) 和 (b)，至少做到 TECS 不是一个 ad hoc 数字。

---

## 完整理论分析

### I. TECS 各组件的理论动机审查

#### 组件 1: 编辑方向 $\Delta\theta_E$

ROME 的 rank-1 update: $\Delta W_{l^*} = \Lambda (C^{-1} k)^T$

- $k$: subject token 在层 $l^*$ 的 key representation
- $C = KK^T$: 经验二阶矩矩阵（对所有 subject tokens 的 key vectors）
- $\Lambda$: 使得新事实成立的 value vector 解

**理论地位**：这是一个约束最小化问题的闭式解——在不改变其他 key 的输出的约束下，修改 $k$ 对应的 value。理论动机清晰：$C^{-1}k$ 是使修改对其他 keys 正交的投影。

**数学欠账**：ROME 假设 MLP 权重矩阵是 key-value memory 的线性读写头。这个"线性关联记忆"模型是简化的。Causal tracing 找到的层 $l^*$ 是 causal effect 最大的层，不是理论上最优的层。多个层可能联合编码同一事实。

#### 组件 2: 归因方向 $g_M$

$g_M = \frac{1}{k}\sum_{i=1}^k \nabla_{\theta_{l^*}} L(z_i)$

**严重理论问题**：这里有一个概念跳跃。TDA 方法（IF/TRAK）的核心贡献是 $H^{-1}$ 加权，它将原始梯度从欧氏空间投射到 Fisher 信息几何空间。$g_M$ 丢掉了这个加权，退化为朴素梯度平均。

数学上：
- IF 影响方向：$\propto H^{-1} \nabla L(z)$（在参数流形的自然坐标系中）
- $g_M$: $\propto \nabla L(z)$（在欧氏坐标系中）

两者在 $H = \alpha I$ 时等价，但这个条件的含义是参数空间各方向等重要——这对过参数化模型显然不成立（某些方向是 flat 的，某些是 sharp 的）。

2409.19998 的发现（阻尼 Hessian 逆趋向单位矩阵）提供了部分辩护，但这同时也是一个坏消息：如果 $H^{-1} \approx \lambda^{-1}I$，则 IF 本身退化为梯度点积，IF 选出的 top-k 样本可能本身就不可靠。

**建议**：至少做两版 TECS——一版用 $g_M$（原始梯度），一版用 $H^{-1} g_M$（IF-weighted），对比差异。如果差异很小，可以引用 2409.19998 作为辩护。如果差异很大，说明 $g_M$ 是错误的选择。

#### 组件 3: 余弦相似度

$\text{TECS} = \cos(\Delta\theta_E^{l^*}, g_M^{l^*})$

**问题 1: 维度与结构**

$\Delta\theta_E$ 是 rank-1 矩阵。将其展平为向量后做余弦，等价于测量 $g_M$（展平后）在这个一维子空间上的分量占 $g_M$ 总范数的比例。设 $\Delta\theta_E = u v^T$，展平后 $\text{vec}(\Delta\theta_E) = v \otimes u$。则：

$$\text{TECS} = \frac{\langle \text{vec}(g_M), v \otimes u \rangle}{\|\text{vec}(g_M)\| \cdot \|v \otimes u\|} = \frac{u^T g_M v}{\|g_M\|_F \cdot \|u\| \cdot \|v\|}$$

这实际上是 $g_M$ 的一个双线性形式，方向由 ROME 的 key/value 向量决定。这个量的物理意义需要阐明。

**问题 2: 高维余弦的统计特性**

对于 $d$-维随机向量，$\cos$ 集中在 $0 \pm O(1/\sqrt{d})$。ROME update 的维度是 $d_{in} \times d_{out}$（如 GPT-2-XL 的 MLP 约 1600 x 6400 = 10M 参数）。Null 分布的标准差约 $1/\sqrt{10^7} \approx 0.0003$。因此即使 TECS = 0.01 也可能在统计上显著——但是否在实际上有意义？

**建议**：报告 TECS 的效果量（相对于 null 分布标准差的倍数），而不仅仅是绝对值。

#### 组件 4: 层限制 $l^*$

只在编辑层比较。

**理论问题**：如果知识在多层分布式存储，只看 $l^*$ 会 miss 大部分信号。ROME 的 causal tracing 选择的是 causal effect 最大的单层——这是一个贪心选择，不保证最优性。

**但也有辩护**：如果 ROME 编辑只修改 $l^*$，那比较也应该限制在 $l^*$。$\Delta\theta$ 在其他层为零，余弦无法计算。因此层限制是方法决定的，不是假设。

### II. 关键理论漏洞总结

| # | 漏洞 | 严重性 | 修补方案 |
|---|------|--------|---------|
| T1 | $g_M$ 丢掉 $H^{-1}$，理论无辩护 | 高 | 做 ablation: $g_M$ vs $H^{-1}g_M$；或引用 2409.19998 + 验证 |
| T2 | TECS null distribution 未推导 | 高 | 推导或 permutation test |
| T3 | 对齐假设无理论保证 | 高 | 明确声明为实证假设，降低 claim 强度 |
| T4 | rank-1 结构使余弦退化为双线性形式 | 中 | 阐明物理意义，考虑投影距离替代 |
| T5 | top-k 选择引入 selection bias | 中 | 分析 k 的敏感性，报告 k 从 10 到 1000 的 TECS 变化 |

### III. 可能的理论强化路径

1. **从 d-TDA (2506.12965) 出发**：d-TDA 证明 IF 是 unrolled differentiation 的分布性极限，不需要凸性。能否利用这个框架推导 TECS 在某些条件下的理论下界？具体地：如果训练数据 $z$ 确实是事实 $f$ 的来源，且模型完美记忆了 $z$，那么 $\nabla_{\theta_{l^*}} L(z)$ 与 $\Delta\theta_{ROME}(f)$ 的余弦是否有一个可推导的下界？

2. **Representer 定理类比**：如果 MLP 层的输出可以写成训练样本 key 的线性组合（类似 kernel 的 representer theorem），那么 ROME 的 update 方向与 top-k 训练样本梯度的对齐有一个自然的解释。探索这条路径。

3. **投影度量替代余弦**：考虑用 principal angle 或 subspace distance 替代余弦。$\Delta\theta$ 是 rank-1，但 $g_M$ 的有效维度可能更高。用 $\Delta\theta$ 定义的一维子空间与 $g_M$ 定义的子空间之间的 principal angle 可能是更 principled 的度量。

### IV. 理论可行但需要诚实的 claim

TECA 的理论基础不足以支撑"TECS 是 TDA 评估的理论替代指标"这样的强 claim。但足以支撑更谦虚的 claim："TECS 是一个参数空间几何一致性的实证度量，用于探索 TDA 归因方向与模型编辑方向的关系。" 后者是一个合法的实证研究问题，理论漏洞变成了需要回答的研究问题而非致命缺陷。
