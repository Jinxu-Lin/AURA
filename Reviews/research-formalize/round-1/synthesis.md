## 多维辩论综合（Debate Synthesis）

### 分歧地图

**共识点**（>=3 视角）：

1. **Direction A vs Direction B 的不一致性是首要问题**（4/4 同意）。problem-statement.md 描述的 FM1/FM2 框架（Direction B: DATE-LM + representation vs parameter space 系统评测）与 project.md 的 BSS 诊断方向（Direction A: per-test-point Hessian sensitivity on CIFAR-10）是完全不同的研究项目。method-design.md 和 experiment-design.md 已经跟随 Direction B，但 project.md 仍然描述 Direction A。必须明确选择一个方向。

2. **DATE-LM Pilot 是存亡级门槛**（4/4 同意）。所有证据来自 CIFAR-10/ResNet-18（FM1 轻微，FM2 不存在），而核心主张针对 LLM 规模。没有 LLM 规模的正面信号，论文是外推而非实证科学。RepSim 在 AURA 自身数据中 LDS 仅 0.074 vs IF 的 0.297，在 DATE-LM 上全面失败的风险真实存在。

3. **CIFAR-10 anti-correlation (tau = -0.467) 不能直接论证 LLM-scale FM1/FM2 独立性**（3/4 同意：contrarian, comparativist, pragmatist）。RepSim 在 CIFAR-10 上 LDS 极低（0.074），anti-correlation 可能反映"一个方法有信号，另一个是噪声"，而非"两种方法捕获不同信号"。

4. **时间和算力预算极度紧张**（3/4 同意：pragmatist, comparativist, contrarian）。NeurIPS 2026 约 8 周，DATE-LM 实验需 ~60-155 GPU-hours，已用预算 ~28/42 GPU-hours，零容错余量。

**主要分歧**：

1. **FM1/FM2 框架的理论深度**：
   - Contrarian：FM1/FM2 是重新包装已有观察（gradient orthogonality, DDA knowledge bias），不是新发现，根因深度评分 4/10。
   - Comparativist：FM1/FM2 的独立性测试（2x2 ablation）是真正的科学新颖点，如果交互项小则有意义。
   - Interdisciplinary：FM1/FM2 与 MIMO 分集、Fisher 信息几何有深层数学同构，理论深度被低估。

2. **BSS 方向是否应被保留**：
   - Pragmatist：如果 DATE-LM Pilot 失败，应回退到 BSS 诊断方向（已有 Phase 1/2b 数据，仅需 ~14 GPU-hours 完成）。
   - Contrarian：BSS 方向（Direction A）与 FM1/FM2 方向（Direction B）叙事逻辑不兼容，不能硬拼。
   - Interdisciplinary：BSS 残差（regress out gradient norm）可以作为 FM1 的精细诊断工具，两个方向可以桥接。

3. **representation-space "family" 的合法性**：
   - Contrarian：5 个方法差异巨大（hidden states vs learned encoders vs parameter gradients），统一为一个 family 是过度归纳。
   - Comparativist：bilinear structure phi^T psi 是有用的分类学观察，但是 survey 级别贡献而非研究级别。

**独特洞察**：

- **Contrarian**：FM1 和 FM2 可能不是独立失效模式，而是"高维近似误差"的两个症状。如果 GGN 级 Hessian 在 LLM 上可行，parameter-space 方法可能直接追平 representation-space。
- **Comparativist**：RIF (Rescaled IF, 2506.06656) 和 BIF (Bayesian IF, 2509.26544) 是关键遗漏——它们从 parameter space 内部修复 FM1，如果有效则"parameter space 结构性受限"的叙事被削弱。DDA 的 fitting error 可能是 FM3，框架需要处理。
- **Pragmatist**：Contrastive scoring 在 data selection task 上的构造不自然（"base model"是什么？），2x2 矩阵可能有一整列空缺。
- **Interdisciplinary**：硬路由（选 IF 或 RepSim）是最差的分集方案，软加权合并（MRC 框架）理论上总是优于硬切换；Baselga beta-diversity 分解可以零成本丰富现有 J10 分析。

---

### 优先级排序

**必须处理**（deal-breaker 级）：

1. **解决 Direction A / Direction B 不一致性**。problem-statement.md（Direction B: FM1/FM2 on DATE-LM）与 project.md（Direction A: BSS on CIFAR-10）描述不同项目。必须做出明确选择，并更新所有文档保持一致。判定：选择 Direction B（FM1/FM2 框架），保留 Direction A 作为 fallback。

2. **DATE-LM Pilot 必须在进入 design 前完成**。在 Pythia-1B + toxicity filtering 上跑 RepSim vs TRAK（~6 GPU-hours）。Kill criterion：RepSim LDS < TRAK LDS - 10pp on all metrics → 回退到 BSS 方向。这是存亡级验证，不可跳过。

3. **补充关键文献**。RIF (2506.06656) 是对 FM1"结构性不可修复"论述的直接反驳，必须正面回应。AirRep 的 learned representations 与 fixed representations 的区别需要讨论。BIF (2509.26544) 需要定位。

**可选处理**（提升质量但非阻断）：

4. 将 Contrastive scoring 在 data selection task 上的构造问题前置分析——如果无法自然构造，考虑将 2x2 ablation 限制在 toxicity filtering 上（contrastive 最自然的 task）。
5. 对现有 J10 数据做 Baselga 分解（turnover vs nestedness），零成本后处理。
6. 在论文理论框架中引入 Fisher 信息几何视角，提升 FM1 诊断的理论深度。
7. 收窄"representation-space family"范围：聚焦 RepSim 和 RepT 作为核心代表，AirRep 和 Concept IF 作为 discussion 中的扩展案例。

**暂时搁置**：

8. 软加权合并（MRC）替代硬路由——属于 Direction A 的 Phase 3 优化，与当前 Direction B 的 diagnostic + benchmark 路线关系不大。
9. Bode 灵敏度积分验证——理论美学贡献，非 MVP 必需。
10. FM3 (DDA fitting error) 是否纳入框架——可在论文 discussion 中处理。

---

### 判定

**[Revise]**

problem-statement.md 存在三个结构性问题，无法直接进入 design 阶段。

**第一，Direction A/B 不一致性未解决。** problem-statement.md 描述了 FM1/FM2 框架（Direction B），但 project.md 仍然描述 BSS 诊断（Direction A）。两个方向的研究问题、实验设计、评估基准完全不同。当前状态下，文档系统内部自相矛盾。必须在 problem-statement 中明确声明方向选择，更新 project.md 使其一致，并将 BSS 定位为 fallback 方案。

**第二，核心假设缺乏 LLM-scale 任何验证。** 所有经验证据来自 CIFAR-10/ResNet-18，而 problem-statement 自己承认 FM1 在低维设置中"轻微"，FM2 在无 pre-training 的设置中"不存在"。这意味着现有证据实际上不支持核心主张。DATE-LM Pilot（RepSim vs TRAK on toxicity filtering, ~6 GPU-hours）必须在 formalize → design 转换前完成，作为 kill-or-go gate。如果 Pilot 失败，应回退到 Direction A（BSS 诊断），该方向有已确认的 Phase 1/2b 数据，工程风险低得多。

**第三，关键竞争方法遗漏。** RIF (Rescaled IF) 和 BIF (Bayesian IF) 从 parameter space 内部修复 FM1 类问题，是对"parameter space 结构性受限"核心叙事的直接挑战。problem-statement 必须正面回应这些方法，解释为什么 rescaling/Bayesian 修复不能替代 representation-space 的结构性优势，否则审稿人会直接质疑。

这三个问题都是 formalize 层面的修订，不需要重新设计实验，但需要在进入 design 前解决。

---

### 修订后的研究方向

**主方向（Direction B）**：FM1/FM2 信号处理诊断框架 + DATE-LM 系统评测。

核心调整：
- 明确承认这是从 Direction A（BSS 诊断）的 pivot，而非延续。CIFAR-10 数据作为"motivating evidence"（启发性证据），不作为"supporting evidence"（支撑性证据）——两者的区别必须在论文中明确。
- DATE-LM Pilot 作为 formalize → design 的硬性 gate。
- 正面回应 RIF/BIF 对 FM1"结构性"论述的挑战：论证 rescaling 缓解 FM1 的程度有限（在 B ~ 10^9 量级，即使 rescaled，SNR 改善可能不足以匹敌 d ~ 4096 的 representation space），或者将 RIF 纳入 baseline 让数据说话。
- 收窄"representation-space family"的范围：承认 5 个方法的异质性，聚焦于 RepSim 和 RepT 作为核心代表，AirRep 和 Concept IF 作为 discussion 中的扩展案例。

**Fallback 方向（Direction A）**：如果 DATE-LM Pilot 失败（RepSim LDS < TRAK LDS - 10pp on all tasks），回退到 BSS 诊断方向。已有 Phase 1/2b 数据支撑，仅需 ~14 GPU-hours 完成 Phase 2a/3，风险可控，timeline 充裕。

---

### 下一阶段重点

1. **执行 DATE-LM Pilot**（最高优先级，~6 GPU-hours）：Pythia-1B + LoRA, toxicity filtering, RepSim vs TRAK, 单 seed。Pass: RepSim LDS >= TRAK LDS - 5pp on at least one metric。Kill: RepSim LDS < TRAK LDS - 10pp on all metrics → 回退 Direction A。
2. **修订 problem-statement.md**：(a) 明确声明 Direction B 选择和 Direction A fallback；(b) 将 CIFAR-10 证据降级为 motivating evidence；(c) 补充 RIF/BIF/AirRep 深度讨论；(d) 收窄 representation-space family 定义；(e) 同步更新 project.md 使其与 Direction B 一致。
3. **补充文献覆盖**：RIF (2506.06656), BIF (2509.26544), Zhu & Cangelosi (2508.07297), AirRep learned vs fixed representation 分析, DDA fitting error 定位。

---

### 未解决的开放问题

- **RepSim 在 LLM scale 的表现**：AURA 自身 CIFAR-10 数据中 RepSim LDS 仅 0.074（vs IF 0.297）。虽然 problem-statement 论证 LLM 场景不同（FM1/FM2 更强），但无经验验证。DATE-LM Pilot 是唯一的解答途径。
- **2x2 ablation 的可行性**：Contrastive scoring 在 data selection task 上的构造不自然（DDA 的 contrastive 是为 hallucination tracing 设计的）。如果 2x2 矩阵在部分 task 上无法完整填充，RQ3（FM1/FM2 独立性）的统计检验力不足。
- **RIF/BIF 的实际效果**：如果 parameter-space 内部修复（rescaling, Bayesian）在 LLM scale 实质性缩小与 representation-space 的差距，FM1"结构性不可修复"的论述需要重大修改——可能需要弱化为"parameter-space 修复代价高昂"而非"结构性受限"。
- **时间压力**：~8 周到 NeurIPS 2026，零容错余量。DATE-LM 环境搭建 + baseline 复现可能独占 1-2 周。
- **算力缺口**：完整实验需 60-155 GPU-hours，原始 42h 预算已近耗尽（剩余 ~14h）。需要确认 jinxulin A6000 的可用性和额外预算。
