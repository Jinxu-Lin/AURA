# Synthesis — TECA Startup Debate

## I. 分歧地图

### 共识点（六方一致）

1. **问题本身合法**：TDA 缺乏不依赖重训练的外部验证通道，这是一个真实的方法论空白。所有辩论者都承认这一点。
2. **假设 1 是最脆弱的假设**：TDA 梯度方向与 ROME 编辑方向在参数空间中对齐——六方一致认为这是整个项目的地基，且没有理论保证。
3. **Pilot 成本极低**：GPT-2-XL + 10 个事实 + raw gradient，0.5 GPU-day 可出 go/no-go 信号。
4. **"失败也有价值"**：即使 TECS 全面失败（所有方法 cosine ~ 0），这本身是关于知识分布式存储的实验证据。
5. **ROME 可靠性有争议**：side effects、causal tracing 定位是"瓶颈"还是"存储位置"的争论，所有人都视为重大风险。

### 主要分歧

| 分歧点 | 乐观方（Innovator/Interdisciplinary/Empiricist） | 悲观方（Contrarian/Theorist/Pragmatist） |
|--------|--------------------------------------------------|------------------------------------------|
| 循环论证是否致命 | 可通过 null TECS 控制实验和 toy model forward modeling 缓解 | 根本性缺陷——没有外部锚点，两端各自不可靠，互验无意义 |
| TECS 能否超越 ad hoc | Empiricist: 否证条件明确即可；Interdisciplinary: Fisher 度量可 principled 化 | Theorist: g_M 丢掉 H^{-1} 无辩护，null distribution 未推导，理论上是 ad hoc |
| 项目天花板 | Innovator: 可升维至"知识探测框架"达 spotlight；Interdisciplinary: 多信使结构有独立价值 | Contrarian: 最好情况也是"两个不可靠工具碰巧对齐"；Pragmatist: 评估指标论文天花板 poster |
| confound 是否可控 | Interdisciplinary: 随机方向编辑控制实验可排除 systematic bias | Contrarian: loss landscape 几何是 confound，控制梯度范数后信号可能消失 |

### 独特洞察（仅单方提出但有重大价值）

1. **Theorist — rank-1 结构使余弦退化为双线性形式**：TECS 实际上是 $u^T g_M v / (\|g_M\|_F \cdot \|u\| \cdot \|v\|)$，物理意义需要阐明。这是一个数学事实，而非观点，必须在设计中正面处理。
2. **Empiricist — 梯度 angular variance 检查**：如果 top-k 梯度 pairwise cosine < 0.05，聚合梯度 g_M 本身不是有意义的"方向"，TECS 框架的数学基础直接崩塌。这是一个 < 30 分钟的前置检查，必须先做。
3. **Contrarian — "behavioral probing" 替代方案**：不在参数空间比较，而比较行为变化（ROME 编辑后 vs TDA top-k 删除后），绕开参数空间对齐假设。这是一条完全不同且更稳健的路径。
4. **Interdisciplinary — toy model forward modeling**：在知识位置已知的 toy model 上验证 TECS 能否恢复真实对齐度，成本极低（CPU 1 小时），提供 ground truth 锚点。这直接回应了 Contrarian 的循环论证质疑。
5. **Pragmatist — ROME 编辑方向的非唯一性**：同一事实的不同编辑目标（Paris→London vs Paris→Berlin）产生不同的 $\Delta\theta$，引入未讨论的自由度。
6. **Innovator — 升维到"参数空间知识探针矩阵"**：横轴多种知识定位方法，纵轴多种归因方法，TECS 是矩阵中的一个 cell。

---

## II. 优先级排序

### MUST — 必须处理（不处理则不应启动）

1. **梯度 angular variance 前置检查**（Empiricist）。如果 top-k 梯度方向近乎正交，g_M 作为"方向"就是 noise，TECS 在数学上无意义。这是 30 分钟的检查，在任何 pilot 之前必须完成。

2. **Null TECS 控制实验**（Interdisciplinary/Contrarian）。用随机方向 rank-1 update（保持相同范数）计算 null TECS。如果 null TECS 与 real TECS 无统计差异，信号来自 loss landscape 几何而非知识对齐，项目终止。这直接回应 Contrarian 的 confound 质疑，是区分"真实对齐"和"系统性偏差"的最低标准。

3. **Practical significance threshold 预定义**（Empiricist）。不仅要 p < 0.05，还要 Cohen's d > 0.5 且 TECS 绝对值 > 0.01（10 倍随机标准差）。在高维空间中，"统计显著"但效应量微小的结果没有任何价值。

### SHOULD — 应该处理（显著提升可信度和天花板）

4. **Toy model forward modeling**（Interdisciplinary）。构造知识位置已知的小模型，验证 TECS 在 ground truth 存在时的行为。成本极低，直接打破循环论证——如果在已知答案的情况下 TECS 都不工作，大模型上更没希望。

5. **g_M 聚合方式的 ablation**（Theorist/Empiricist）。至少对比：raw mean、IF-score-weighted mean、subspace projection（top-r SVD of gradient matrix）。特别是 subspace 方法可以绕开"单向量聚合不合理"的问题。

6. **跨层 TECS profile**（Innovator/Interdisciplinary/Empiricist）。不仅在 l* 计算，在所有层计算，观察信号是否在 causal tracing 层达峰值。这同时是对 ROME 层定位的独立验证。

### COULD — 可选处理（Phase 2+ 探索性扩展）

7. Fisher 度量替代欧氏度量（Interdisciplinary）
8. 随机矩阵理论推导 null distribution（Interdisciplinary/Theorist）
9. 知识探针矩阵升维（Innovator）
10. 动态一致性追踪（Innovator）
11. Behavioral probing 替代路径（Contrarian）——如果参数空间路径失败，这是 pivot 方向

### PARK — 暂时搁置

12. EK-FAC IF 实现——成本高、近似质量有争议（Pragmatist），pilot 用 raw gradient 足够
13. LDS full retrain 验证——100-200 GPU-days 不现实（Pragmatist），用 linear probing 替代
14. 拓扑数据分析方法（Interdisciplinary）——成本高、解释性弱

---

## III. 判定

### 结论：Go with focus

**理由**：

方向本身合法，pilot 成本极低（< 1 GPU-day），否证条件清晰可操作，且"失败也有价值"的特性提供了下限保护。六方中无人认为存在 deal-breaker 级别的风险（核心假设未被文献反驳——只是未被证实；所有组件有可行实现；pilot 可在合理时间给出早期信号）。三位建议 Hold 的辩论者（Innovator/Pragmatist/Theorist）的 Hold 条件都是"先做 sanity check"，而非"根本不应做"。

**但必须 focus**：Contrarian 和 Empiricist（加权更高）都指出了非平凡的系统性风险——循环论证和 confound。这些不是"可以事后处理的细节"，而是决定项目生死的前置条件。因此 focus 的核心是：在 pilot 中同时验证信号存在性和排除 confound。

### 值得启动的核心理由

1. **真实的方法论空白**：TDA 评估完全依赖重训练 ground truth，一个参数空间的独立验证通道——即使是弱的——也有明确的学术贡献。
2. **极低的验证成本**：0.5 GPU-day 的 pilot 即可给出 go/no-go 信号，风险-收益比极其有利。
3. **失败保底**：即使 TECS 全面失败，"TDA 归因方向与模型编辑方向不对齐"本身是关于知识分布式存储的实证证据，可以构成独立贡献。

### 必须优先处理的 3 个问题

1. **g_M 的数学合理性**：top-k 梯度是否在参数空间中指向一致方向？（angular variance 检查，30 分钟）。如果梯度近乎正交，聚合后的 g_M 是 noise vector，整个框架在数学上无意义。这是最早的 kill gate。

2. **信号 vs confound 的区分**：TECS > 0 是因为知识对齐还是因为 loss landscape 几何偏好？（null TECS 控制实验 + 控制梯度范数后的 residual TECS）。Contrarian 提出的最坏情况——"控制层级梯度范数后 TECS 消失"——必须在 pilot 中直接检验。

3. **效应量的实际意义**：在 10^6-10^7 维参数空间中，统计显著不等于有意义。必须预定义 practical significance threshold（Cohen's d > 0.5, |TECS| > 0.01），否则可能自欺。

### 进入 C 时必须带着的风险

- **循环论证风险**（Contrarian，高权重）：即使 pilot 通过，TECS 仍然面临"两个不可靠工具的一致性不能证明两者都对"的认识论质疑。论文的 claim 必须降级为"实证一致性度量"而非"验证工具"。
- **贡献天花板风险**：如果止步于"评估指标"，天花板是 poster。升级到 spotlight 需要揭示知识表征的几何结构（Innovator 的升维路径），这需要远超 TECS 本身的额外分析。
- **ROME 可靠性风险**：ROME 的 side effects 和 causal tracing 争议无法在本项目中解决，只能承认并控制（编辑成功/失败分组报告）。

### 哪条假设最值得先验证

**假设 1（TDA 梯度方向与 ROME 编辑方向对齐）**，六方一致。但验证方式需要分层：

- **第 0 步**（前置条件）：验证 g_M 本身是否是有意义的方向（angular variance 检查）。如果 g_M 是 noise，假设 1 甚至无法被有意义地测试。
- **第 1 步**：在 10 个事实上计算 TECS(real) vs TECS(null)（随机方向编辑），检验信号是否超过 confound 基线。
- **第 2 步**（如果第 1 步通过）：toy model forward modeling，在知识位置已知时验证 TECS 的行为。

### 六方意见中仍未消解的真实分歧

1. **循环论证是否可以被实验打破**。Interdisciplinary 认为 toy model forward modeling + null TECS 可以缓解；Contrarian 认为这些控制实验只能排除部分 confound，无法从根本上解决"没有外部 ground truth"的问题。**我的判断**：Contrarian 在认识论上是对的——循环论证不能被完全打破。但 Interdisciplinary 在实践上也是对的——部分控制 > 完全不控制。解决方案：承认 claim 的边界，不声称 TECS "验证" TDA，只声称 TECS 提供"参数空间几何一致性的实证度量"。

2. **贡献天花板是 poster 还是 spotlight**。Innovator 认为升维到知识探针矩阵可达 spotlight；Contrarian/Pragmatist 认为评估指标天花板就是 poster。**我的判断**：初始目标定为 poster+，以 TECS 作为核心贡献；升维路径（知识探针矩阵）作为 Phase 2 的条件扩展——仅在 Phase 1 结果强阳性时启动。不应在立项时就承诺 spotlight。

3. **Behavioral probing 是否是更好的路径**。Contrarian 提出在行为空间（而非参数空间）比较。这绕开了参数空间对齐的所有假设，但也失去了参数空间几何分析的独特贡献。**我的判断**：不作为主线替代，但作为 pilot 失败后的 pivot 方向保留。

---

## IV. 决策摘要

| 项目 | 决策 |
|------|------|
| **最终结论** | **Go with focus** |
| **核心理由** | 真实方法论空白 + 极低验证成本 + 失败保底 |
| **Focus 内容** | Pilot 中必须同时做：(1) angular variance 前置检查，(2) null TECS 控制，(3) practical significance 预设 |
| **必须先验证的假设** | 假设 1，分三步：g_M 合理性 → 信号 vs confound → toy model ground truth |
| **进入 C 的风险** | 循环论证无法完全消解（降级 claim）；天花板受限（不承诺 spotlight）；ROME 可靠性争议（分组控制） |
| **Kill gates** | (a) angular variance < 0.05 → Stop；(b) TECS(real) vs TECS(null) Cohen's d < 0.5 → Stop；(c) 控制梯度范数后 TECS 消失 → Stop |
| **Pivot 备选** | Behavioral probing（行为空间比较，绕开参数空间假设） |

### Pilot 执行序列（进入 C 后的第一优先级）

```
Step 0 (30 min): Angular variance check
  → top-k 梯度 pairwise cosine < 0.05? → STOP

Step 1 (4 hours): Core pilot
  → GPT-2-XL, 10 facts, ROME edit
  → Raw gradient TECS + Random direction null TECS
  → Cohen's d < 0.5 或 |TECS| < 0.01? → STOP
  → 控制层级梯度范数后 TECS 消失? → STOP

Step 2 (1 hour, 可并行): Toy model forward modeling
  → 2-layer MLP, 100 key-value pairs, known storage location
  → TECS 能否恢复已知对齐? → 如果不能, HOLD + 诊断

Step 3 (如果 Step 1-2 通过): 扩展到 500 facts + 多 TDA 方法
```
