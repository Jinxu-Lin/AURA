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
