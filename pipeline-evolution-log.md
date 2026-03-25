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
