# Pipeline Evolution Log — AURA

> 各阶段 X-reflect 产出追加到此文件。供 `/praxis-evolve` 汇总处理。

---

## Entry 1 — C (Crystallize) — 2026-03-16

**执行模式**: 首次
**时间分配观察**: Step 1 (Gap 候选生成 + 知识库交叉搜索) 占约 40%——需要同时关联 10+ 篇论文的 Gaps & Assumptions 和 Cross-Paper Connections，从中做组合推导。Step 5 (攻击角度设计) 和 Step 6 (探针方案) 各占约 20%，互相约束使得需要反复调整。

### 观察

#### 缺失 (Missing)
- [ ] **[当前阶段] [中]** — Prompt 强调"从知识库中做组合推导"，但缺乏对"startup 辩论中已形成的攻击角度如何与 C 阶段衔接"的明确指导。project-startup.md 的 §4.3 已提供候选攻击角度 A/B，C 阶段是应该从零开始重新推导还是在 startup 基础上精化？实际执行中选择了后者（在 startup 的 TRV 方向上深化），但 prompt 未明确此预期。
  - 建议：在 C prompt 的"输入文档"部分明确说明 project-startup.md 候选攻击角度的消费方式——"作为起点精化而非从零推导，除非审查否定了方向"。

#### 改进 (Improve)
- [ ] **[Prompt: crystallize-prompt] [中]** — Step 1-2 (Gap 候选生成和评估) 与 Step 3 (Root Cause) 的顺序在实际执行中是交织的。评估 Gap 的"可解性"需要 Root Cause 分析（因为可解性取决于攻击角度是否可信，攻击角度取决于 Root Cause 类型）。Prompt 将它们呈现为线性步骤，但实际执行是循环的。
  - 建议：在 Step 1-2 的引导中更明确地承认"Gap 评估与 Root Cause 分析的循环依赖——可解性评估可在 Step 5 完成后回填"。

- [ ] **[当前阶段] [中]** — 知识库 kb-index.md 文件极大（515KB），无法一次读取。需要多次定向搜索 Gaps & Assumptions 和 Cross-Paper Connections 的相关条目。知识库结构虽然丰富，但检索效率受限于文件大小。
  - 建议：考虑将 kb-index.md 按资产类型拆分为独立文件（gaps-index.md、connections-index.md、methods-index.md、patterns-index.md），降低单次检索成本。

#### 边界 (Boundary)
- [ ] **[跨阶段] [中]** `[BOUNDARY]` — problem-statement.md §2.2 攻击角度描述需要控制在"不超过 2 段话"以防越界成方法设计。但在实际执行中，为了让 D 阶段的 agent 能高效接手，需要在攻击角度描述中提供足够的技术锚点（如 SI 公式、Jaccard@k 定义）。"2 段话"的限制在 startup 已提供丰富背景时可能太紧——后续 D 阶段 agent 需要能从 problem-statement.md 直接理解攻击角度的技术核心而无需回溯 project-startup.md。
  - 建议：将限制调整为"核心 idea 不超过 2 段话，但可附加公式定义和符号约定"。

#### 确认 (Confirm)
- **[当前阶段]** — "Gap、攻击角度、探针三者共同设计"的要求非常有效。在实际执行中，Gap 的"可解性"评估确实强制依赖攻击角度设计，探针的 Pass 标准直接来自 Gap 的否证条件。三者分离会导致大量返工。
- **[当前阶段]** — Startup 六维辩论产出的"已知风险"和"否证条件"直接为探针设计提供了约束（如 Empiricist 的 3-seed + OOD confound 要求、Contrarian 的 naive ensemble baseline 要求），减少了 C 阶段的开放度。

---

## Entry 3 — C (Crystallize, RS-Revise) — 2026-03-16

**执行模式**: rs_revise
**时间分配观察**: 审查意见定位和差异化论证设计占约 60%（Daunce/BIF 差异化论证需要精确界定三种不确定性维度的正交性）。DR 叙事修正占约 15%（需要全局搜索和替换 DR 相关表述）。探针增强占约 20%（新增 3 个验证步骤 + 2 个 pass 标准行）。剩余 5% 为元数据和一致性检查。

### 观察

#### 改进 (Improve)
- [ ] **[当前阶段] [中]** — RS-Revise 模式下最耗时的部分是"将审查发现的竞争工作整合到 Gap 定义中，同时保持差异化论证的精确性"。Daunce/BIF 的存在不否定 AURA 的 gap，但需要将 gap 从"归因可靠性量化缺失"收窄到"Hessian 近似选择敏感性的 per-test-point 诊断缺失"。这种收窄本身是有价值的——更精确的 gap 定义降低了 reviewer 质疑新颖性的风险。
  - 建议：在 C prompt 的 RS-Revise 指导中增加"竞争工作整合模板"——区分 (a) 竞争工作否定了 gap 的存在性, (b) 竞争工作覆盖了 gap 的部分范围需要收窄, (c) 竞争工作从正交角度探索了相关问题需要差异化论证。

#### 确认 (Confirm)
- **[当前阶段]** — "不从零开始"的 RS-Revise 原则非常有效。Root Cause 分析、RQ 设计、攻击角度核心逻辑均无需修改（审查已 Pass 这些维度），只需在 §1.1 补充竞争工作 + §1.2 增加差异化 + §2.2-2.3 修正 DR 叙事 + §3 增强探针。保留了约 70% 的原始内容。
- **[跨阶段]** — 审查中 Comparativist 发现的 Daunce/BIF 确实是关键遗漏——如果不补充，未来 reviewer 会直接问"为什么不用 Daunce 的方差作可靠性指标"。RS 审查在这一点上发挥了核心价值。

#### 边界 (Boundary)
- [ ] **[当前阶段] [低]** `[BOUNDARY]` — 差异化论证中详细描述了 Daunce/BIF 的技术路径（模型扰动方差 vs 后验方差 vs Hessian 近似敏感性），这接近于对竞争方法的技术分析。但这在 problem-statement 中是必要的（reviewer 需要理解差异），不应留到 D 阶段。判定为合理的边界扩展。

---

## Entry 2 — RS (Strategic Review) — 2026-03-16

**执行模式**: 首次
**时间分配观察**: Step 3 (多视角辩论) 占约 55%——需要从 4 个独立视角深度分析文档，其中 Comparativist 的在线搜索（WebSearch）占比最大（发现 Daunce 和 BIF 两个关键遗漏工作是本次审查的最重要产出）。Step 4（综合报告撰写）占约 35%。Step 2（文档读取）仅占 ~10%。

### 观察

#### 缺失 (Missing)
- [ ] **[Prompt: strategic-review-prompt] [中]** — Comparativist 的在线搜索是本次审查发现最关键遗漏工作（Daunce, BIF）的唯一途径。但 prompt 中对 Comparativist 的搜索策略缺乏"搜索什么关键词"的具体指导——本次执行中使用了 5 次 WebSearch，关键词选择依赖经验判断。对于不同领域的项目，搜索策略可能差异很大。
  - 建议：在 strategic-review-prompt 中增加"Comparativist 搜索策略模板"——至少包含 (a) 核心方法关键词变体搜索, (b) 核心问题关键词变体搜索, (c) 按作者/研究组追踪最新工作。

#### 改进 (Improve)
- [ ] **[Prompt: strategic-review-prompt] [中]** — 4 个辩论 Agents 的输出在单 Agent 执行模式下是顺序生成的（非真正并行）。由于后续 Agent 不可避免地受前序 Agent 分析的影响（共享同一上下文窗口），"独立性"保证弱于真正的多 Agent 并行。这在 Contrarian 和 Comparativist 的分析中尤为明显——两者都指向了 Daunce/BIF，但可能是同一推理链而非独立发现。
  - 建议：在单 Agent 执行模式的文档中注明"辩论独立性为近似独立"，或在多 Agent 团队可用时强制使用 SendMessage 实现真正并行。

- [ ] **[当前阶段] [低]** — 审查维度 7 项中，某些维度（如 "RQ 可回答性"）的评估相对机械——problem-statement.md 的 RQ 设计已经很严谨，审查基本是确认性的。相比之下，"Gap 新颖性 + 竞争态势"维度的审查产出了最多新信息。维度权重可以不必均等。
  - 建议：考虑在 YAML 配置中为每个维度增加 `weight` 或 `priority` 字段，使综合判定时高权重维度的影响更大。

#### 确认 (Confirm)
- **[当前阶段]** — Comparativist 角色 + WebSearch 工具的组合是审查中最有价值的部分。发现了 problem-statement.md 完全遗漏的两个 ICML 2025 工作（Daunce, BIF），这直接改变了审查判定（从可能的 Pass 变为 Revise）。如果没有在线搜索能力，这个审查会遗漏最关键的问题。
- **[跨阶段]** — Phase 1-2 显式解耦的设计在审查中被反复确认为正确决策——Phase 2 的 H2/DR 风险不会连带拖垮 Phase 1 的独立贡献。C 阶段的这个设计选择有效降低了审查的 Block 概率。

---

## Entry 4 — RS (Strategic Review, 第二轮) — 2026-03-16

**执行模式**: rs_revise（第二轮审查）
**时间分配观察**: Step 2 (文档读取 + 第一轮辩论回顾) 占约 20%——需要完整读取 v1.1 修订内容和第一轮所有辩论输出。Step 3 (多视角辩论) 占约 50%——第二轮辩论焦点转向修订质量评估和残留风险深化。Step 4（综合报告）占约 30%。

### 观察

#### 改进 (Improve)
- [ ] **[Prompt: strategic-review-prompt] [中]** — 第二轮审查的辩论 Agents 需要同时阅读第一轮辩论输出和修订后的文档，上下文窗口压力大。建议在多轮审查的 prompt 中增加结构化的"修订响应性检查清单"——对每个第一轮必须修改项，要求 Agent 先做 binary 判断（已修改/未修改），再评估修改质量。这可以减少重复分析，聚焦于增量问题。

- [ ] **[当前阶段] [低]** — 第二轮辩论产出的最有价值洞察来自 Interdisciplinary（不确定性分解框架），但这个洞察在第一轮就已部分出现（Hampel gross error sensitivity 类比）。第二轮的深化使其从"类比"变成了可操作的"叙事框架"。这说明跨学科视角在迭代中会自然深化——multi-round debate 对 Interdisciplinary 角色特别有价值。

#### 确认 (Confirm)
- **[当前阶段]** — 第二轮审查从 Revise 升级为 Pass 的决策是合理的：第一轮的两个必须修改项已完成，第二轮未发现新的根本性问题。"建议修改"（不确定性分解叙事、cross-seed 相关性检验等）是真正的改进方向而非阻断性缺陷。
- **[跨阶段]** — Pragmatist 建议的 Pilot A/B 分阶段探针执行策略非常务实——将 3-4 周的探针拆为"1 周核心验证 + 2-3 周完整分析"，确保在核心假设 Fail 时快速止损。建议在 probe-guide.md 中增加"分阶段执行"的一般性指导。
- **[当前阶段]** — Codex MCP 的 review 工具仅支持代码审查，不适用于研究方向的战略审查。对于 RS 阶段的 Codex 并行审查，需要使用 codex 的自由文本 prompt 模式而非 review 模式。当前跳过是合理的。
