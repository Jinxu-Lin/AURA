# Project Startup: TECA — TDA-Editing Consistency Analysis

> 本文档是研究项目的知识基础文档。
> 目标：将研究者的隐性洞察转化为 AI 可操作的结构化知识。
> 本文档将在整个项目周期中作为核心参考被反复调用。

---

## 1. Research Seed (研究种子)

### 1.1 核心洞察 (Core Insight)

TDA 评估的根本困境是所有现有方法都依赖重训练获取 ground truth，而重训练本身不可靠（Spearman miss-relation、重训练噪声）。TECA 提出一个完全独立的验证通道：利用模型编辑（ROME/MEMIT）的参数更新方向作为 TDA 归因方向的外部验证信号。

核心观察：TDA 和模型编辑在参数空间中操作同一个对象，但从相反的方向——TDA 追溯"哪些训练样本影响了这个知识"，ROME 主动修改"知识存储在哪些参数中"。如果两者指向相同的参数子空间，则相互验证。

### 1.2 洞察来源类型

- [x] 方法融合型：TDA + Model Editing 的跨方法桥接
- [x] 问题驱动型：TDA 评估独立性缺失这一真实问题

### 1.3 预期贡献 (Expected Contribution)

1. 提出 TECS（TDA-Editing Consistency Score），一个不依赖重训练的 TDA 评估指标
2. 实证揭示 TDA 归因方向与模型编辑方向在参数空间中的几何关系
3. 如果 TECS 全面失败（所有方法 ≈ random），这本身是关于知识分布式存储的实证证据

### 1.4 初始假设清单

- **假设1**：TDA 梯度方向与 ROME 编辑方向在参数空间中指向相同子空间（至少在编辑层 l*）——支撑弱，无直接实验证据；若为假则 TECS 失去意义
- **假设2**：ROME 的 rank-1 update 足够忠实地反映"知识存储位置"，而非编辑方法的人工 artifact——支撑中，causal tracing 有争议；若为假则 TECS 测量 ROME 偏置
- **假设3**：IF/TRAK 等梯度方法的归因方向确实反映了训练样本对特定知识的影响——支撑中，大模型上 IF 近似质量存疑；若为假则两端都不可靠

---

## 2. Source Materials (源材料)

### Source A: Revisiting the Fragility of IF (2303.12922)
- **类型**: 论文
- **核心要点**: 证明 Spearman 相关作为 IF 评估指标存在 miss-relation 问题；从最优点重训练产生的"ground truth"本身是噪声
- **与本项目的关联**: 直接支持 TECA 的动机——现有 TDA 评估方法论有系统性缺陷，需要独立验证通道

### Source B: Do Influence Functions Work on LLMs? (2409.19998)
- **类型**: 论文
- **核心要点**: IF 在 LLM 上三重失效：iHVP 近似退化、收敛不确定性、参数-行为断层（ASR 差 400 倍但 Δθ 差 1.6 倍）
- **与本项目的关联**: 严重反证——如果 IF 梯度方向在 LLM 上不可靠，TECS 的归因端可能是噪声。但本项目使用 GPT-2-XL/GPT-J（较小模型），问题可能不如 7B 严重

### Source C: Infusion (2602.09987)
- **类型**: 论文
- **核心要点**: IF 逆向优化在视觉域有效（100% 成功率）但在语言模型上失败（0.1% rank flip），说明参数空间方向语义在 LM 上可能较弱
- **与本项目的关联**: 桥接 IF 与参数空间操作的直接先例；语言模型上的失败是重要警示

### Source D: MDA (2601.21996)
- **类型**: 论文
- **核心要点**: 在参数子空间上计算 IF 比全参数空间更有意义；联合 Q-K 子空间 EK-FAC 修改
- **与本项目的关联**: 支持在编辑层 l* 子空间上比较的设计思路

### Source E: ROME (2202.05262) & MEMIT (2210.07229)
- **类型**: 方法
- **核心要点**: Causal tracing 定位知识层 → rank-1 update 编辑事实。MEMIT 扩展到批量编辑
- **与本项目的关联**: TECS 的编辑端基础；causal tracing 争议和 side effects 是核心风险

---

## 3. Knowledge Synthesis (知识综合)

### 3.1 源材料之间的关系

Source A 和 B 共同构成对现有 TDA 评估的双重质疑：A 质疑评估方法论（Spearman/retrain），B 质疑 IF 方法本身在 LLM 上的有效性。Source C 是 IF 与参数空间操作的桥接先例，其语言模型上的失败与 B 的发现一致。Source D 提供了子空间 IF 的成功案例。Source E 是 TECS 的另一端基础。

核心张力：A 支持 TECA 的动机（需要新评估方式），但 B+C 质疑 TECA 的手段（参数空间方向在 LM 上可能不可靠）。

### 3.2 Gap Analysis (差距分析)

现有 TDA 评估方法（LDS、LOO、Spearman）全部基于"删除→重训练→观察"范式，共享系统性偏差。无人提出过参数空间几何一致性作为独立验证通道。这是一个真实且未被填补的方法论空白。

### 3.3 技术可行性初步判断

所有核心组件（ROME、TRAK、GPT-2-XL、CounterFact）有成熟开源实现。Pilot 成本极低（< 1 GPU-day）。主要不确定性在于信号是否存在，而非能否实现。

---

## 3.4 多 Agent 多维压力测试 (Multi-Agent Stress Test)

### 核心假设清单

| # | 假设 | 来源 | 支撑强度 |
|---|------|------|---------|
| 1 | TDA 梯度方向与 ROME 编辑方向在 l* 层对齐 | 研究者推断 | 弱 |
| 2 | ROME rank-1 update 反映知识存储位置 | ROME 论文 | 中（有争议） |
| 3 | IF/TRAK 归因方向可靠 | TDA 文献 | 中 |
| 4 | 余弦相似度是合理的对齐度量 | 研究者选择 | 中 |
| 5 | 仅编辑层 l* 比较足够 | ROME 假设 | 中（有争议） |

---

### 视角摘要

#### 创新者（Innovator）
- **最强继续理由**：参数空间几何一致性作为评估维度是全新的，未有先例
- **最危险失败点**：两端同时不可靠导致四路混淆，任何结果都难以下结论
- **建议**：`Hold`（先做 sanity check；可升维到"知识探针矩阵框架"提高天花板）

#### 务实者（Pragmatist）
- **最强继续理由**：Pilot 成本极低（0.5 GPU-day），工程组件全有开源实现
- **最危险失败点**：ROME rank-1 update 是约束最小二乘解，与 TDA 梯度无理论对齐理由
- **建议**：`Hold`（先做 2 小时 sanity check；砍掉 EK-FAC，用 raw gradient 即可）

#### 理论家（Theorist）
- **最强继续理由**：如果对齐存在，将为知识表征的参数空间几何提供新实证
- **最危险失败点**：g_M 丢掉 H^{-1} 无理论辩护；ROME rank-1 结构使余弦退化为双线性形式；无 null distribution
- **建议**：`Hold`（需先推导 null distribution 和阐明 rank-1 余弦的物理意义）

#### 反对者（Contrarian）
- **最强继续理由**：方法论空白真实存在
- **最危险失败点**：循环论证——两个各自不可靠的工具互验无法产生可靠结论
- **建议**：`Hold`（除非能设计打破循环论证的实验；behavioral probing 是更稳健的替代路径）

#### 跨学科者（Interdisciplinary）
- **最强继续理由**：双通道交叉验证在地震学、多信使天文学中有成功先例
- **最危险失败点**：两种探针可能被同一 confound（loss landscape 几何）驱动
- **建议**：`Go with focus`（toy model forward modeling + 随机方向控制实验可缓解循环论证）

#### 实验主义者（Empiricist）
- **最强继续理由**：TECS 数学精确、实验成本低、否证条件清晰
- **最危险失败点**：高维参数空间中统计显著但效应量微小
- **最小正面信号**：梯度方法 TECS Cohen's d > 0.5 vs random baseline
- **否证条件**：top-k 梯度 pairwise cosine < 0.05（g_M 是 noise）；TECS(real) vs TECS(null) Cohen's d < 0.5
- **建议**：`Go with focus`

---

### 综合判定

**判定**：`Go with focus`

**值得启动的核心理由**：
- TDA 评估的方法论空白是真实的，一个参数空间的独立验证通道——即使是弱的——也有明确学术贡献
- Pilot 成本极低（< 1 GPU-day），风险-收益比极其有利
- 失败也有价值——"TDA 归因方向与模型编辑方向不对齐"本身是关于知识分布式存储的实证证据

**进入 C 前必须处理的优先问题**：
- g_M 的数学合理性（梯度 angular variance 检查，30 分钟）
- 信号 vs confound 的区分（null TECS 控制实验）
- 效应量的实际意义（预定义 practical significance threshold）

**进入 C 时带着的已知风险**：
- 循环论证无法完全消解，claim 必须降级为"参数空间几何一致性度量"而非"验证工具"
- 贡献天花板受限（初始目标 poster+，不承诺 spotlight）
- ROME 可靠性争议无法在本项目中解决，只能分组控制

**仍未消解的真实分歧**：
- 循环论证能否被实验缓解（跨学科者 vs 反对者）
- 天花板是 poster 还是可达 spotlight（创新者 vs 反对者）

---

## 4. Research Direction (研究方向)

### 4.1 核心研究问题 (Core Research Questions)

- 问题1: TDA 归因方向与模型编辑方向在参数空间中是否存在可测量的几何对齐？
  - 关键否证线索：如果所有 TDA 方法（含梯度方法）的 TECS 与 random baseline 无统计差异，则参数空间对齐不存在
- 问题2: 如果对齐存在，TECS 能否区分不同质量的 TDA 方法？
  - 关键否证线索：如果 TECS 排序与 LDS 排序 Spearman < 0.5，则 TECS 不是 LDS 的有效替代

### 4.2 拟议方法概述

TECS = cos(Δθ_E^{l*}, g_M^{l*})：在 ROME 编辑层 l* 的参数空间上，计算编辑方向（rank-1 update）与归因方向（top-k 训练样本聚合梯度）的余弦相似度。方法级汇总：对多个事实取 TECS 均值。

### 4.3 候选攻击角度

- **攻击角度 A**: 参数空间几何一致性——直接计算 TECS 并验证其是否能区分 TDA 方法。最直接，pilot 成本最低。
- **攻击角度 B**: 如果 A 失败，pivot 到行为空间一致性——比较 ROME 编辑后 vs TDA top-k 删除后的行为变化（绕开参数空间假设）。

### 4.4 关键技术挑战

- g_M 聚合方式的选择（raw mean vs IF-weighted vs subspace projection）
- ROME rank-1 结构使余弦退化为双线性形式，物理意义需阐明
- 高维空间中信号与 confound 的区分（null TECS 控制实验设计）
- ROME 编辑失败/side effects 的处理策略

### 4.5 已知风险与未解决的质疑

- [反对者] 循环论证：两个不可靠工具互验无法得出可靠结论 — C 中降级 claim 定位
- [理论家] g_M 丢掉 H^{-1} 无理论辩护 — C 中需 ablation（raw vs IF-weighted vs subspace）
- [实验主义者] 高维空间效应量可能微小 — C 中预定义 practical significance threshold
- [务实者] ROME 编辑方向的非唯一性（不同目标事实 → 不同 Δθ） — C 中控制实验
- [跨学科者] loss landscape 几何可能是 confound — C 中设计随机方向编辑控制

---

## 5. Metadata

- **创建日期**: 2026-03-16
- **研究者**: Jinxu Lin
- **目标会议/期刊**: TBD (初始目标 poster+)
- **预计时间线**: Pilot 1 周 → 全实验 4-6 周
- **多 Agent 检验结论**：`Go with focus`

### Kill Gates（Pilot 阶段触发则终止）

1. Top-k 梯度 angular variance (pairwise cosine) < 0.05 → Stop
2. TECS(real) vs TECS(null) Cohen's d < 0.5 → Stop
3. 控制层级梯度范数后 TECS 消失 → Stop
