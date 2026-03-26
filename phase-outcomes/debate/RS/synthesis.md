## Strategic Review (RS) 多视角辩论综合 — 第二轮

### 修订质量评估

v1.1 在结构上完成了第一轮的两个必须修改项：(1) 补充了 Daunce/BIF 的引用和差异化分析；(2) 修正了 DR 叙事。所有四个辩论者确认修订已响应。但对修订实质强度的评估存在分歧——Contrarian 和 Comparativist 认为差异化论证仍有弱点，Interdisciplinary 建议用更精准的理论框架强化叙事，Pragmatist 聚焦于探针范围膨胀的工程风险。

### 交叉验证发现

**强信号问题**（多视角共识）：

1. **"理论正交"声称缺乏实证锚点** — [Contrarian + Comparativist + Interdisciplinary]：v1.1 声称 TRV、Daunce 方差、BIF 后验方差度量"理论上正交"的不确定性维度。这个概念框架是清晰的，但三个视角同时指出它缺乏实证验证。Contrarian 指出三个维度可能都是 loss landscape 曲率的不同表征（因此高度相关）；Comparativist 指出 BIF 的后验方差可能已隐式包含 Hessian 近似影响；Interdisciplinary 从不确定性分解理论指出交互项不可忽视。**核心建议**：探针中增加 TRV 与 cross-seed 归因方差（作为 Daunce 代理）的相关性检验。这不需要额外计算（已有 3 seed），但提供了正交性的初步实证。

2. **Ground truth LDS 的计算成本被低估** — [Pragmatist]：v1.1 新增的 Step 8（TRV-high vs TRV-low 的 LDS 对比）需要 ground truth 归因。如果需要 leave-one-out retraining，成本约 50-100 GPU-hours，远超探针预算。Pragmatist 建议了三种降级方案——这个问题需要在进入探针前明确解决。

**重要独立发现**：

- **[Interdisciplinary]** 提出了一个关键的叙事改进：将 TRV 定位为"model specification uncertainty in TDA"，放入不确定性分解框架（aleatoric / epistemic / model specification）。这个框架使差异化论证从"我们度量不同的东西"（模糊）变为"我们度量不确定性分解中的特定组件"（精确），且与 Bayesian Model Averaging 文献建立了理论连接。建议在 problem-statement.md 的差异化论证中采纳。

- **[Contrarian]** 指出"从业者面临的最直接问题"叙事过度自信——大部分 TDA 从业者不会在多个 Hessian 近似之间切换。这个叙事需要调整为更诚实的表述（如"对于需要评估归因可靠性的安全关键应用"）。

- **[Pragmatist]** 建议将探针分为 Pilot A（核心假设验证，1 周）和 Pilot B（完整分析，2-3 周），Pilot A Fail 即止。这是合理的工程策略。

- **[Comparativist]** 指出 Daunce 的框架在概念上可以泛化到涵盖 AURA 的 use case（如果扰动集合包含不同 Hessian 近似变体）。这是一个需要在论文 related work 中正面回应的 reviewer 质疑。

**分歧议题及裁判**：

- **是否需要在探针中增加 Daunce-style 方差检验**：Contrarian 认为必须（否则正交性声称站不住脚），Pragmatist 认为增加探针负担。**裁判：建议修改而非必须修改。利用已有 3 seed 的 cross-seed 归因排名方差作为 Daunce 代理，不需要额外计算——只需在分析阶段增加一个 Spearman 相关系数。工程成本极低。**

- **Stability Selection 是否纳入探针**：Interdisciplinary 再次建议，综合者再次裁定为**可选**。理由：5x 计算成本增加有信息量但非核心假设验证。

---

### 修订指引

**建议修改（本轮无"必须修改"项——第一轮必须修改已完成）**：

1. **差异化叙事引入不确定性分解框架** — 来源：Interdisciplinary — 具体修改：在 §1.2 的差异化论证中，将 TRV 定位为 "model specification uncertainty"（模型规约不确定性），与 Daunce 的 training stochasticity 和 BIF 的 parameter uncertainty 形成不确定性分解的三个组件。这不改变任何技术内容，只改善叙事精准度。 — 预期 cost：~1 小时文档修改。影响：显著提升 reviewer 理解差异化的清晰度。

2. **探针增加 cross-seed 归因方差与 TRV 的相关性检验** — 来源：Contrarian + Interdisciplinary — 具体修改：在 §3.2 中增加一步：利用 3 seed 的归因排名计算 per-test-point 的 cross-seed 方差（作为 Daunce 方差的代理），检验 Spearman(TRV, cross-seed_variance)。 — 预期 cost：无额外计算成本（已有 3 seed 数据），仅需 ~10 行分析代码。

3. **明确 Ground truth LDS 的获取策略** — 来源：Pragmatist — 具体修改：在 §3.2 Step 8 中明确 ground truth 来源——优先使用 Hong et al. 的 Full Hessian 归因作为近似 ground truth（如果代码支持），否则降级为 cross-seed 一致性作为可靠性代理。 — 预期 cost：依赖 Step 0 的代码可行性验证结果。

4. **调整"从业者最直接问题"叙事** — 来源：Contrarian — 具体修改：将 §1.2 末尾的叙事从"从业者日常面临的问题"调整为"安全关键应用中需要评估归因可靠性时面临的问题"。 — 预期 cost：~10 分钟。

**已裁定可忽略**：

- Contrarian 要求探针增加 loss 值作为第二控制变量 — 忽略理由：偏相关控制变量的增加是渐进改进，梯度 norm 已是最关键的 confound。后续正式实验中可以增加更多控制。
- Interdisciplinary 的 Conformal Prediction 方向 — 忽略理由：与第一轮裁定一致，有趣但过于发散。
- Interdisciplinary 的 BMA 形式化建议 — 部分忽略：叙事层面的不确定性分解框架建议采纳（修改项 1），但 BMA 形式化留给 Phase 2。

---

### 综合判定

**小幅修订即可**

v1.1 已成功修复第一轮的两个核心问题（Daunce/BIF 差异化 + DR 叙事修正）。第二轮辩论未发现新的根本性问题——提出的均为改进建议而非阻断性缺陷。最有价值的改进建议是 Interdisciplinary 的不确定性分解框架叙事和 Contrarian 的正交性实证检验——前者是纯叙事改进（~1 小时），后者利用已有数据（零额外计算）。探针的工程风险（ground truth LDS 成本、代码适配复杂度）是真实的，但属于执行层面问题，不影响战略判断。

**Pass 条件评估**：所有 7 个审查维度在第一轮中已有 6 个 Pass、1 个 Revise（Gap 新颖性+竞争态势）。v1.1 的修订使该 Revise 维度升级为 Pass（Daunce/BIF 差异化已建立，虽然实证强度仍有提升空间）。第二轮无新的 Revise 或 Block 维度。

**建议路由**：Pass — 进入探针实验（P 阶段），携带本轮建议修改作为探针设计的改进方向。
