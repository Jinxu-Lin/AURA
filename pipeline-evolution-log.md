# Pipeline Evolution Log -- AURA

> 各阶段 X-reflect 产出追加于此。供 /praxis-evolve 汇总处理。

## Entry 0 -- Assimilation -- 2026-03-25

**执行模式**: 项目合并 (CRA_old + AURA + CRA -> unified v3 framework)

### 观察

#### 确认 (Confirm)
- **[跨项目]** -- 三个项目（CRA_old, AURA, CRA）在 TDA 方向上各自探索了不同角度，合并后的理论框架（FM1/FM2）比任何单一项目更完整。CRA_old 提供了理论骨架，AURA 提供了经验验证，CRA 提供了跨领域视角。
- **[实验]** -- AURA 的 CIFAR-10/ResNet-18 实验数据（特别是 tau = -0.467 的反相关发现和方差分解结果）在合并后获得了更强的解释力：不再是"IF 和 RepSim 为什么不一致"的诊断问题，而是"FM1/FM2 独立性的经验证据"。

#### 改进 (Improve)
- [ ] **[流程] [中]** -- 项目合并（assimilation）在 Noesis v3 中缺少结构化的"知识融合"步骤。当前是手动提取各项目的核心贡献并重组。建议在 assimilate-skill 中增加"源项目知识提取模板"，标准化合并流程。

#### 边界 (Boundary)
- [ ] **[跨项目] [低]** -- AURA 的 Sibyl 系统文件（.sibyl/, config.yaml, status.json, iter_001/）与 Noesis v3 的目录结构并存。不影响运行但增加目录复杂度。长期应考虑归档 Sibyl 文件。

---

## Entry 1 — formalize (fr_revise, v1.2 pivot) — 2026-03-25

**执行模式**: FR-Revise → Major direction pivot
**时间分配**: 读取遗留材料 ~40%, Gap 评估与方向决策 ~30%, 文档写作 ~30%

### 观察

#### 改进 (Improve)
- [ ] **[Prompt: formalize-prompt] [高]** — Formalize prompt 假设迭代是在同一方向内微调（FR-Revise = 修订弱点），但实际遇到了跨方向 pivot（从 FM1/FM2 到 TECA geometric incommensurability）。Prompt 中的 FR-Revise 指令（"不从零开始，保留已通过审查的内容"）在方向 pivot 时反而成为障碍。建议增加 pivot 判断逻辑：如果审查结论与新方向不兼容，允许重写而非修订。
- [ ] **[当前阶段] [中]** — Legacy 数据（TECA results in JSON format）需要大量解析工作才能理解 subspace analysis 的具体数值。建议在项目合并时为 legacy 实验数据创建标准化的 summary.md。

#### 确认 (Confirm)
- **[跨阶段]** — 保留 legacy 实验数据（teca-sibyl/results/）的决策被验证为正确。这些 JSON 文件中的定量结果（effective dimensionality, principal angles, cross-projection fractions）直接支持了新方向的 Gap 论证。
- **[当前阶段]** — 6 大 DL Gap 模式的系统性检查虽然在 pivot 场景下不是最核心的步骤，但"做了但有根本缺陷"分类帮助明确了 Gap 的价值层次。

#### 边界 (Boundary)
- [ ] **[跨阶段] [中] [BOUNDARY]** — formalize_review 的综合意见中 DATE-LM probe 要求与新方向不兼容。在 Direction B (FM1/FM2) 下 DATE-LM probe 是存亡级门槛，但在 Direction G1 (geometric incommensurability) 下完全不需要。这意味着 formalize_review → revise → formalize 的循环在方向 pivot 时可能产生"已解决但无关"的修订条件。Runner 无法区分"按审查意见修订"和"通过方向 pivot 使审查意见失效"。

---

## Entry 2 — design (v3.0, geometric incommensurability) — 2026-03-25

**执行模式**: First design for new direction (post-pivot from BSS/FM1FM2)
**时间分配**: Reading legacy TECA data ~30%, method component design ~40%, experiment planning ~30%

### 观察

#### 确认 (Confirm)
- **[当前阶段]** — The "method = analysis framework" pattern works well for negative-result papers. The six-component structure (TECS + SVD + PAF + whitening ablation + attribution ablation + toy model) maps naturally to six experiments, making the method-experiment coupling tight.
- **[当前阶段]** — Toy model as "oracle validation" is a powerful design pattern for theoretical claims. It provides a ground-truth anchor that purely empirical analysis cannot.

#### 改进 (Improve)
- [ ] **[Prompt: design-prompt] [中]** — Design prompt assumes the method is a MODEL or ALGORITHM (references "组件分解", "计算复杂度", "信息流分析" for learned representations). For analysis papers where the "method" is a measurement framework, many of these steps are inapplicable. Could benefit from a conditional branch: if attack angle is "characterization/analysis" rather than "new method," skip implementation-focused steps.
- [ ] **[当前阶段] [中]** — Legacy code assessment was harder than expected because TECA code is in two locations (Codes/experiments/teca/ AND legacy/teca-sibyl/scripts/) with overlapping but not identical files. Consolidation during assimilation would have saved time.

#### 边界 (Boundary)
- [ ] **[跨阶段] [中] [BOUNDARY]** — The design phase for an analysis paper is closer to "experiment planning" than "method design." The method-design.md and experiment-design.md distinction feels artificial when the method IS the experiment. Consider allowing merged documents for analysis papers.

## Entry 2 — design (dr_revise, v1.1) — 2026-03-25

**执行模式**: DR-Revise (design review round-1 returned REVISE with 4 blocking items)
**时间分配**: 读取审查意见 + 现有文档 ~30%, 理论修订 (M1-M3) ~40%, 文档对齐重写 (M4) ~30%

### 观察

#### 改进 (Improve)
- [ ] **[流程] [高]** — 三份设计文档（problem-statement, method-design, experiment-design）在 3 次独立修改中分别指向了 3 个不同方向（TECA, BSS, FM1/FM2）。当前 runner 不检查文档间一致性。建议在 design_review 阶段增加文档对齐检查：如果 3 份文档的方向关键词（Gap 名称、方法名、benchmark 名）不一致，自动标记为 blocking issue。
- [ ] **[Prompt: design-prompt] [中]** — design prompt 中的"保留未被质疑的部分"指令在 dr_revise 模式下与"重写文档以对齐方向"需求冲突。当审查意见要求全局对齐（M4），局部保留策略不适用。建议 dr_revise 在检测到 M4-type alignment issue 时自动升级为"全文重写"模式。

#### 确认 (Confirm)
- **[当前阶段]** — Design review 的 4 blocking items 质量很高：M1（TRAK paradox）是论文最明显的弱点，M2（independence framing）是数学错误，M3（SNR formalization）是诚信问题，M4（alignment）是基本功。这些都是 reviewer 会在 30 秒内发现的问题。
- **[跨阶段]** — 将 SNR 从 theorem 降级为 motivating analysis 实际上让论文更强：2x2 实验结果本身就是证据，不需要依赖有缺陷的理论论证。

#### 边界 (Boundary)
- [ ] **[流程] [低]** — 版本号语义不清晰。problem-statement 从 v1.2 回退到 v1.1（方向变回 FM1/FM2），method-design 从 v2.0 到 v1.1（跨方向重写）。是否应该用 v4.0 表示"第 4 次重写"而不是 v1.1 表示"回到 v1.x 方向的第 1 次修订"？当前选择了后者以保持与 iteration_major/minor 语义一致，但可能造成混淆。
