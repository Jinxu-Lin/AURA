# 战略审查报告（第二轮）

## 多视角辩论摘要

**辩论 Agents**：Contrarian（反对者）、Comparativist（文献对标者）、Pragmatist（务实者）、Interdisciplinary（跨学科者）

**第一轮修订响应评估**：v1.1 在结构上完成了第一轮的两个必须修改项——(1) 补充 Daunce (2505.23223) 和 BIF (2509.26544) 的引用及差异化分析，(2) 将 DR 叙事修正为 "diagnostic-guided adaptive ensemble"。所有四个辩论者确认修订已响应。

**第二轮强信号问题**（多视角共识）：

- **"理论正交"声称缺乏实证锚点**：[Contrarian + Comparativist + Interdisciplinary] v1.1 声称 TRV、Daunce 方差、BIF 后验方差度量理论正交的不确定性维度。概念框架清晰，但三个维度可能都是 loss landscape 曲率的不同表征（Contrarian），BIF 后验方差可能已隐式包含 Hessian 近似影响（Comparativist），不确定性分解中的交互项不可忽视（Interdisciplinary）。建议利用已有 3 seed 数据检验 TRV 与 cross-seed 归因方差的相关性，提供正交性的初步实证。

- **Ground truth LDS 计算成本被低估**：[Pragmatist] Step 8 的 TRV-high vs TRV-low LDS 对比若需 LOO retraining 约 50-100 GPU-hours，远超探针预算。需明确降级方案（Full Hessian 归因作为近似 ground truth，或 cross-seed 一致性作为代理）。

**重要独立发现**：

- [Interdisciplinary] 将 TRV 定位为 "model specification uncertainty"（模型规约不确定性），放入不确定性分解框架（aleatoric / epistemic / model specification），使差异化论证从概念性提升到理论框架级别。与 Bayesian Model Averaging 文献建立了连接。
- [Contrarian] "从业者面临的最直接问题"叙事过度自信——建议调整为面向安全关键应用的可靠性需求。
- [Pragmatist] 探针范围从第一轮 6 步扩展到 9 步，实际人工时间估计从 3-5 天增至 6-10 天。建议分为 Pilot A（核心假设，1 周）和 Pilot B（完整分析，2-3 周），A 失败即止。
- [Comparativist] Daunce 框架在概念上可泛化到涵盖 AURA 的 use case——论文 related work 中需正面回应此 reviewer 质疑。

**分歧议题裁判**：

- **正交性实证检验是否必须** — 裁定为建议修改（利用已有 3 seed，零额外计算成本）。
- **Stability Selection 基准是否纳入探针** — 裁定为可选（信息量高但非核心假设验证）。

---

## 各维度评估

### 1. Gap 真实性

**判定**：Pass

**证据摘要**：与第一轮一致。TDA 领域确实缺乏基于 Hessian 近似选择敏感性的 per-test-point 归因可靠性诊断工具。v1.1 已正确引用了 Daunce 和 BIF 作为"从不同角度触及归因可靠性"的已有工作，并将 AURA 的 Gap 范围精确收窄为"Hessian 近似选择对归因排名的 per-test-point 影响"。

**与第一轮的变化**：无。Gap 真实性在第一轮已 Pass，v1.1 通过补充 Daunce/BIF 使 Gap 边界更锐利。

### 2. Gap 重要性与贡献天花板

**判定**：Pass

**证据摘要**：与第一轮一致。Phase 1 (TRV 诊断 + SI 代理 + 经验分布 + 稳定性-正确性分析) 天花板为 NeurIPS/ICML poster；Phase 1+2 天花板为 poster-to-spotlight。Interdisciplinary 的不确定性分解框架叙事建议（如果采纳）可能进一步提升贡献的理论深度。

**与第一轮的变化**：无变化。

### 3. Gap 新颖性 + 竞争态势

**判定**：Pass（从第一轮 Revise 升级）

**证据摘要**：v1.1 已建立了与 Daunce 和 BIF 的差异化论证——TRV 量化方法选择敏感性（model specification uncertainty），而非训练随机性（Daunce）或参数不确定性（BIF）。在线搜索未发现 2025Q4-2026Q1 出现新的直接竞争工作。DualXDA (TMLR 2025) 为稀疏归因方向，不构成直接竞争。竞争窗口仍估计 6-9 个月。

**保留意见**：差异化论证的实证强度仍有提升空间——"正交性"声称需要探针阶段的初步验证（利用 cross-seed 方差）。但这属于探针执行的改进方向，不构成 Revise 理由。

**与第一轮的变化**：从 Revise 升级为 Pass。第一轮 Revise 的原因是 Daunce/BIF 未被引用、差异化未建立——这两个问题已在 v1.1 中修复。

### 4. Root Cause 深度

**判定**：Pass

**证据摘要**：与第一轮一致。Root cause 分析（3 层 "why"，从缺乏指标到评估范式到 Hessian 谱结构非均匀性）有理论和实验证据支撑。v1.1 在 §1.2 中增加的 Daunce/BIF 差异化论证进一步强化了 root cause 的独特性——AURA 关注的 root cause（Hessian 近似谱与 test point 梯度的对齐）是 Daunce（模型扰动）和 BIF（后验分布）未触及的维度。

**与第一轮的变化**：无变化。

### 5. 攻击角度可信度

**判定**：Pass

**证据摘要**：攻击角度 A1（TRV 诊断 + 自适应融合）直接对应 root cause。v1.1 将 Phase 2 叙事从 DR 修正为 "diagnostic-guided adaptive ensemble"，消除了第一轮的主要保留意见（DR 表面类比）。Phase 1-2 解耦设计使 Phase 2 的 H2 风险不影响 Phase 1 的独立价值。

**残留保留意见**：(1) "稳定 ≠ 正确"悖论的 Kronecker 共模偏差风险仍存在，但 v1.1 增加的 TRV-LDS 对比（Step 8）提供了验证路径；(2) Phase 2 的 H2 假设仍然薄弱，但已降级为条件推进（不影响 Phase 1）。

**与第一轮的变化**：从 Pass（附保留意见）变为 Pass（残留保留意见减少）。DR 叙事保留意见已消除。

### 6. 探针方案合理性

**判定**：Pass

**证据摘要**：v1.1 的探针方案在第一轮基础上做了实质性改进：增加了 Step 0 代码可行性验证、Step 7 梯度 norm 偏相关、Step 8 TRV-LDS 对比、Step 9 OOD test points。Pass/Fail 标准有定量阈值（TRV 标准差、3-bin 分布、3-seed 一致性、SI-TRV 相关性、偏相关、Cohen's d）。预设失败模式分析完整。

**改进方向**（建议而非必须）：(1) Ground truth LDS 获取策略需明确（Pragmatist）；(2) 增加 cross-seed 方差与 TRV 的相关性分析（利用已有数据，零额外成本）；(3) 考虑分两阶段执行（Pilot A + Pilot B）。

**与第一轮的变化**：探针方案更全面，改进方向减少。

### 7. RQ 可回答性与可证伪性

**判定**：Pass

**证据摘要**：与第一轮一致。RQ1-3 均有明确否证条件和定量阈值。v1.1 新增的 pass 标准（偏相关 > 0.15、Cohen's d > 0.3/0.5）使可证伪性更强。

**与第一轮的变化**：增强（新增两个定量标准）。

---

## 竞争态势分析

基于第二轮 Comparativist 的在线搜索（arXiv 2025Q4-2026Q1）：

**更新**：未发现新的直接竞争工作（无人提出与 TRV 类似的 Hessian 层级链稳定性指标）。

| 工作 | 发表 | 与 AURA 的关系 | 差异化状态 |
|------|------|----------------|-----------|
| Natural W-TRAK (2512.09103) | arXiv 2024/12 | SI 理论先驱 | 已差异化（AURA: 经验 Jaccard + 主动利用 vs W-TRAK: 理论 bound + 被动认证） |
| Daunce (2505.23223) | ICML 2025 | 归因 + 不确定性（模型扰动） | 已差异化（v1.1 §1.2），需探针验证正交性 |
| BIF (2509.26544) | ICML 2025 | 归因 + 不确定性（后验） | 已差异化（v1.1 §1.2），需探针验证正交性 |
| ASTRA (2507.14740) | NeurIPS 2025 | 精度方向（正交） | 无需差异化 |
| d-TDA (2506.12965) | NeurIPS 2025 Spotlight | 分布式归因框架 | 正交但可融合 |
| DualXDA (2402.12118) | TMLR 2025 | 稀疏归因（不同方向） | 无直接竞争 |

竞争窗口估计：6-9 个月（无变化）。大组风险低。

## 贡献天花板评估

- **Phase 1 单独**：Solid NeurIPS/ICML poster — 经验分析 + 新诊断指标 + Daunce/BIF 差异化
- **Phase 1 + Phase 2**：Poster-to-spotlight — 诊断框架 + 实用融合工具
- **如果引入不确定性分解框架叙事**：理论贡献提升（"model specification uncertainty 的首个 TDA 操作化度量"），但量化影响有限
- **一句话 pitch**："We show that TDA attribution reliability varies dramatically across test points depending on Hessian approximation choice, and propose TRV — a diagnostic that tells practitioners when to trust their attribution results, capturing a dimension of model specification uncertainty orthogonal to existing uncertainty measures."

---

## 问题清单

**第一轮必须修改项（已完成）**：
1. ~~补充 Daunce 和 BIF 的引用和差异化分析~~ — 已在 v1.1 §1.1 和 §1.2 完成
2. ~~修正 DR 叙事~~ — 已在 v1.1 §2.2 完成

**第二轮建议修改**：

1. **差异化叙事引入不确定性分解框架** — [Interdisciplinary] — 在 §1.2 中将 TRV 定位为 "model specification uncertainty"，Daunce 为 training stochasticity，BIF 为 parameter uncertainty。纯叙事改进，~1 小时。
2. **探针增加 cross-seed 归因方差与 TRV 的相关性检验** — [Contrarian + Interdisciplinary] — 利用已有 3 seed 数据，零额外计算成本，~10 行分析代码。
3. **明确 Ground truth LDS 获取策略** — [Pragmatist] — 在 §3.2 Step 8 中明确：优先用 Hong et al. 的 Full Hessian 归因，否则降级为 cross-seed 一致性。
4. **调整"从业者最直接问题"叙事** — [Contrarian] — 修改为面向安全关键应用的可靠性需求，~10 分钟。
5. **探针分阶段执行（Pilot A + Pilot B）** — [Pragmatist] — Pilot A（核心假设，1 周），A Fail 即止，Pass 后执行 Pilot B（完整分析，2-3 周）。

---

## 战略预判

1. **如果探针失败，最可能的原因是什么？**
   与第一轮判断一致：TRV 分布退化（> 80% 同级）。机制：EK-FAC/K-FAC 的 Kronecker 特征值 mismatch 是架构级而非 test-point 级。Pragmatist 的 Pilot A 策略可在 1 周内以最小成本给出 go/no-go 信号。

2. **如果需要 pivot，有哪些备选攻击角度？**
   与第一轮一致，新增：(a) 基于不确定性分解框架，系统比较 TRV/Daunce 方差/BIF 后验方差三种可靠性度量的互补性和冗余度——从"提出 TRV"转向"比较不确定性度量"（评估论文而非方法论文）。

3. **此方向最大的 unknown unknown 是什么？**
   新增：TRV 与 Daunce/BIF 不确定性信号的经验相关性。如果高度相关（Spearman > 0.7），AURA 的独立贡献受到根本性威胁——此时 TRV 退化为 Daunce 方差的计算成本更高的替代。这个 unknown 可以在探针中以零额外成本解答。

4. **6 个月后回头看，最可能的 regret 是什么？**
   与第一轮一致："早知道应该直接扩展 Daunce/BIF 框架"。但 v1.1 的差异化论证降低了这个 regret 的概率——如果正交性得到实证支持，TRV 的独特价值是确立的。

---

## 整体判定：Pass

v1.1 已成功修复第一轮的两个核心问题。第二轮辩论评估了修订质量，确认差异化论证在结构上成立（TRV 度量 Hessian 近似选择敏感性，与 Daunce 的训练随机性和 BIF 的后验方差在概念上不同），虽然实证强度仍有提升空间。所有 7 个审查维度均达到 Pass 标准。探针方案设计严谨、有定量 pass/fail 标准、预设失败模式分析完整。

**Pass 的关键依据**：
- Gap 真实（Per-test-point 归因可靠性诊断缺失）+ 重要（服务所有 Hessian-based TDA 方法）+ 有差异化（vs Daunce/BIF 不同不确定性维度）
- Root cause 深度充分（Hessian 谱结构 x test point 梯度对齐）
- 攻击角度直接匹配 root cause，Phase 1-2 解耦设计合理
- 探针方案有明确的 go/no-go 标准和失败模式分析
- RQ 可证伪，否证条件定量

**Pass 携带的已知风险**：
- TRV 分布退化是最大的未验证假设（探针核心目标）
- TRV 与 Daunce/BIF 信号的正交性需要实证验证（建议在探针中低成本检验）
- "稳定 ≠ 正确"的 Kronecker 共模偏差风险需要 TRV-LDS 对比验证
- Ground truth LDS 的获取策略需要在 Step 0 后确定

**进入 P 阶段时的建议**：
- 优先执行 Pilot A（50 test points, 1 seed）快速验证 TRV 非退化
- 利用 3 seed 数据同时收集 cross-seed 归因方差，检验与 TRV 的相关性
- Step 0 的代码可行性验证应在 2 天内给出 go/no-go
- 参考 `Praxis/prompts/probe-guide.md` 执行探针实验
