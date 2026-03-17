## 六维辩论综合（Debate Synthesis）

### 分歧地图

**共识点（>=3个视角同时指向）**：

- **H2（IF-RepSim 误差独立性）是最大理论风险**：Contrarian 直接挑战"值相关性低 != 误差相关性低"；Theorist 指出 estimand 不统一导致"独立"本身需重新定义；Interdisciplinary 警告"doubly robust != doubly immune"——DR 要求至少一个组件足够好，而 AURA 的前提恰恰是两个都可能差；Empiricist 要求 RA-TDA 必须显著超越 naive ensemble 才能 claim TRV 引导有贡献。四个视角从不同角度指向同一结论：Phase 2 的 DR 叙事在当前形式下缺乏坚实基础。

- **Phase 1（TRV 诊断）有独立发表价值，应与 Phase 2 解耦**：Pragmatist 明确建议"Phase 1 先行 + Phase 2 条件推进"；Contrarian 建议"Phase 1 作为独立贡献"；Empiricist 的否证条件设计隐含了 Phase 1 和 Phase 2 可独立评估的前提；Innovator 也承认 Phase 1 单独值 poster。至少 4 个视角认为 Phase 1 不应被 Phase 2 的风险拖累。

- **TRV 变异度是否足够是核心经验问题**：Contrarian 质疑若 TRV 高度集中则自适应退化为常数；Empiricist 设定了明确否证条件（>80% test point 同等级则 H1 不成立）；Pragmatist 的 pilot 设计以 TRV 分布双峰性为核心验证目标。三个视角一致认为这个经验问题必须在 pilot 中回答，不能假设。

- **LLM 上的 IF 计算是工程深水区**：Pragmatist 估计 Llama-7B 的 IF 计算单独需 2-3 周调通；Empiricist 指出大模型上 TRV ground truth 不可获取；Innovator 也承认 TRV 计算成本可能过高以至于无人使用。三个视角认为 Phase 2 的 LLM 规模目标需要降级。

**主要分歧**：

- **Innovator vs Pragmatist**（升维 vs 可行性）：Innovator 提出两个升维方向——(A) 从 post-hoc 诊断到 by-design TDA-aware 训练，(B) 从 point-wise TRV 到 task-level robustness certificate。Pragmatist 的工程评估显示仅 Phase 1+2 就需要 3-4 个月。**综合判定**：升维 A（TDA-aware 训练）是方向重构，代价过大，不纳入当前项目范围；升维 B（Task-RV）代价中等且可在 Phase 1 数据上扩展，作为 C 阶段的可选探索方向保留。判定依据：升维 B 不阻塞核心 pilot，且如果 Phase 1 成功，Task-RV 是自然延伸。

- **Theorist vs Empiricist**（理论完备性 vs 经验优先）：Theorist 要求为 IF 和 RepSim 建立共同 estimand，形式化 DR 类比；Empiricist 更关心否证条件和统计严谨性，认为理论可后补。**综合判定**：站 Empiricist 一侧。DL 领域的有效研究模式是"经验发现先行 + 理论解释后跟"。但 Theorist 提出的 d-TDA 分布视角重新定义 estimand 的建议（将 IF 视为高方差低偏差估计，RepSim 视为低方差高偏差估计）应在 C 阶段纳入讨论，因为它可能为 lambda(TRV) 的选择提供原则性指导。判定依据：理论不完备不阻塞 pilot，但好的理论框架能指导实验设计。

- **Contrarian（W-TRAK + naive ensemble 可能足够）vs 项目核心 claim**：Contrarian 提出"W-TRAK + 固定权重 ensemble"可能与 RA-TDA 表现相当，AURA 必须超越这个 baseline。这不是可搁置的质疑——它直接威胁 Phase 2 的增量贡献。**综合判定**：将"W-TRAK + naive ensemble"纳入 Phase 2 的必要 baseline。如果 RA-TDA 无法显著超越，Phase 2 的贡献不成立，但 Phase 1 的 TRV 诊断工具仍有独立价值。判定依据：这是 deal-breaker 级别的竞争方法，不能回避。

**独特洞察（值得特别关注）**：

- **[Interdisciplinary]** TRV 可形式化为 Hampel 的 gross error sensitivity 的计算版本，无需改变经验计算方式，但立即获得理论意义 + Hampel 最优性理论可导出 lambda(TRV) 的 minimax optimal 形式。这是一个零成本的理论升级，应在 C 阶段直接采纳。

- **[Interdisciplinary]** Neyman orthogonality 检验：纯推导层面验证 tau_RA 是否满足对 IF 和 RepSim 组件的一阶正交性条件。如果满足，DR 性质有理论保证；如果不满足，需要添加 targeting step（类似 TMLE）。零实验成本，可在 Phase 3 理论分析中完成，且结果决定 RA-TDA 的理论叙事是否成立。

- **[Empiricist]** Training variance confound：不同训练 seed 可能让 TRV 看起来有 per-point 变异性但实际只是 training randomness。控制方法：3 个 seed 上计算 TRV，检查 rank correlation（Spearman rho > 0.6）。这是 pilot 中必须包含的控制实验。

- **[Empiricist]** OOD confound：如果 Hessian 近似误差在 OOD test point 上系统性增大，TRV 退化为 OOD detector，AURA 的独立贡献被削弱。Pilot 中加入 20 个 OOD test point 检验相关性。

- **[Innovator]** 当前 Hessian 层级链是固定的经验排序，不同架构可能有不同最优近似。这个质疑合理但不阻塞——作为 C 阶段的 discussion point 记录。

---

### 优先级排序

**必须处理（直接决定成败）**：

1. **TRV 变异度验证（H1）** — 来源：Contrarian + Empiricist — Pilot 必须证明 TRV 分布在 test point 间有足够区分度（至少 3 个等级各占 >10%），否则整个自适应框架坍塌。这是 Phase 1 的生死线。

2. **W-TRAK + naive ensemble baseline** — 来源：Contrarian — Phase 2 必须显著超越此 baseline，否则 RA-TDA 的增量贡献不成立。需在实验设计中作为核心对照组。

3. **Training variance confound 控制** — 来源：Empiricist — TRV 在不同 training seed 间的 rank stability 必须通过验证（Spearman rho > 0.6），否则 TRV 不是可靠的诊断信号。纳入 pilot 设计。

4. **RepSim 失效场景的确认** — 来源：Pragmatist — DR 性质验证需要 IF 失效和 RepSim 失效两个互补场景。IF 失效场景有 Li et al. 的现成实验，但 RepSim 失效场景目前仅有 DATE-LM 作为候选，需先验证 RepSim 确实在此场景失效（而非仅表现较差）。如果找不到清晰的 RepSim 失效场景，DR 叙事的一半无法验证。

**可选处理（有更好但非必须）**：

- **Task-level RV（升维 B）**：Innovator 的 Task-RV 建议。如果 Phase 1 成功，可作为自然延伸，将 AURA 从 point-level 诊断升级到 task-level reliability engineering。中等实现代价，可在 Phase 1 数据上直接扩展。

- **SURE 自动 lambda 选择**：Interdisciplinary 的建议，用 Stein's Unbiased Risk Estimate 替代 LOO retraining 来校准 lambda(TRV)，使 RA-TDA 变为 fully self-calibrating。独立增量改进，不阻塞核心工作。

- **d-TDA 分布视角重新定义 estimand**：Theorist 的建议，将 IF 和 RepSim 统一到"跨 Hessian 近似的期望归因排序"这个共同 estimand 下。可在 Phase 3 理论分析中完成，不增加实验成本。

**暂时搁置（附理由）**：

- **升维 A（TDA-aware 训练范式）** — 搁置理由：方向重构级别的改动，需要修改训练流程，且正则化可能与下游任务性能冲突。当前项目应聚焦 post-hoc 诊断+融合的核心叙事。如果 Phase 1 证明 TRV 有诊断价值，这可作为 future work。属于"可接受的风险"中的"工程实现有不确定性但有 fallback 方案"。

- **多组件融合（IF + RepSim + d-TDA + TracIn + DataInf）** — 搁置理由：二元融合已经是理论和工程的最小可行单元。扩展到多组件在理论上自然但工程代价巨大。在 C 阶段讨论扩展性即可，审稿人问"为什么只融合这两个"时可作为 discussion 回应。属于"有更好但没有也能发"。

- **Theorist 关于 TRV 离散化单调性假设的质疑** — 搁置理由：Hessian 近似质量的全局排序（H > GGN > diagonal > Identity）虽不对所有架构严格成立，但在标准 benchmark（CIFAR-10/ResNet-18、GPT-2）上已由 2509.23437 实证确认。Pilot 使用同一 benchmark，可直接验证。如果出现非单调情况，可改用扰动版 TRV（连续扰动而非离散等级）作为 fallback。属于"工程有不确定性但有 fallback"。

- **H4（SI 是 TRV 代理）失效的后果** — 搁置理由：Pragmatist 正确指出 H4 失败会影响 AURA 实用性叙事和 2512.09103 的理论连接，但 H4 是 Phase 0 的优化项，不是核心贡献。如果 SI-TRV 相关性低，Phase 0 可以退化为"先计算 kappa 做全局判断"（kappa 大 -> 需要 AURA，kappa 小 -> 标准 TDA 够用），TRV 仍可直接计算（只是更贵）。不影响 Phase 1-2 的核心逻辑。

---

### 判定

**方向修正**

AURA 的核心方向——TRV 诊断 + 自适应融合——在概念层面有明确的 meta-level 创新（从"改善归因精度"到"量化归因可信度"），且 Phase 1 的 TRV 诊断工具有独立发表价值。但辩论揭示了两个必须正面处理的问题：(1) Phase 2 的 DR 叙事在理论上存在 estimand 不统一的严重缺口（Theorist + Contrarian + Interdisciplinary 一致指向），且 W-TRAK + naive ensemble 是未被充分重视的竞争 baseline；(2) TRV 变异度是核心经验假设，必须在 pilot 中回答而不能假设成立。修正方向：将 Phase 1 和 Phase 2 明确解耦为可独立贡献的两个模块，Phase 1 作为最小论文单元（TDA robustness diagnostic tool），Phase 2 仅在 Phase 1 成功 + TRV 变异度充分的条件下推进。

---

### 修订后的研究方向

**核心修正 1：Phase 1 与 Phase 2 显式解耦**

原方向将 Phase 1 和 Phase 2 作为连贯整体叙事。修正后：Phase 1（TRV 作为 TDA 归因可信度诊断工具）是最小完整贡献，即使 Phase 2 不成功也可独立发表。Phase 2（RA-TDA 融合）是条件推进的升级模块，需满足：(a) TRV 分布有足够区分度（pilot 验证），(b) RA-TDA 显著超越 W-TRAK + naive ensemble baseline。

修改理由：4 个视角一致认为 Phase 1 有独立价值，且 Phase 2 的 DR 理论基础最弱。解耦后降低项目整体风险，避免 Phase 2 的理论问题拖累 Phase 1 的实际贡献。

**核心修正 2：新增必要 baseline**

实验设计中新增"W-TRAK + naive ensemble（固定权重 IF+RepSim 平均）"作为 Phase 2 的核心对照组。RA-TDA 必须在至少一个 failure mode 上展示 >5% absolute improvement over naive ensemble 才能 claim TRV-guided 融合有增量贡献。

修改理由：Contrarian 正确指出这是 AURA 最强竞争者，且计算 TRV 的成本可能与直接用最好的 Hessian 近似相当（自洽性陷阱），必须正面对比。

**核心修正 3：Pilot 设计增强**

原 pilot 仅验证 TRV 分布形状。修正后增加：(a) 3 个 training seed 的 TRV rank stability 检验（Spearman rho > 0.6）；(b) 20 个 OOD test point 的 TRV-OOD 相关性检验（Spearman rho < 0.8）；(c) Phase 2 的 LLM 规模目标降级到 GPT-2 only 作为保底方案（Llama-7B 仅作为 stretch goal）。

修改理由：Empiricist 提出的 confounders 和 Pragmatist 的 LLM 工程风险评估均为实质性问题，必须在 pilot 阶段控制。

---

### C 重点

进入 Crystallize 时，以下问题需要重点深入：

1. **TRV 的操作定义固化** — 来源：Empiricist 的"metric gaming"警示 + Theorist 的"0.5 阈值无理论依据"质疑。C 阶段必须在 pilot 前固定 TRV 的完整操作定义（k 值、Jaccard 阈值、Hessian 层级链），不允许 post-hoc 调整。同时参考 Interdisciplinary 的建议，将 TRV 形式化为 Hampel gross error sensitivity 的计算版本，获取理论锚点。

2. **Phase 2 的 estimand 统一问题** — 来源：Theorist 的"欠账1" + Interdisciplinary 的 Neyman orthogonality 建议。C 阶段需要明确回答：IF 和 RepSim 的共同估计目标是什么？初步方向：采用 d-TDA 的分布视角，定义为"跨 Hessian 近似的期望归因排序"。如果无法建立统一 estimand，Phase 2 的叙事需要从"doubly robust"修正为更诚实的"adaptive ensemble with diagnostic guidance"。

3. **Phase 1 的独立贡献叙事构建** — 来源：综合判定的"Phase 1-2 解耦"修正。C 阶段需要为 Phase 1 构建完整的独立叙事：TRV 作为"TDA reliability engineering"的诊断工具，回答"什么时候不该信任归因结果"。这个叙事不依赖 Phase 2 的融合，且填补了 TDA 领域的 meta-level 空白（从 method improvement 到 reliability diagnosis）。

4. **竞争 baseline 的系统性梳理** — 来源：Contrarian 提出的 W-TRAK + naive ensemble + ASTRA。C 阶段需要明确 AURA 在计算预算对等条件下相对于每个竞争方法的差异化价值。特别需要解决"计算 TRV 的成本 vs 直接用更好的 Hessian 近似"的自洽性问题。

5. **LLM 规模的实验策略** — 来源：Pragmatist 的工程评估 + Empiricist 的 ground truth 断裂警告。C 阶段需要决定：Phase 2 的 LLM 实验以 GPT-2 为核心还是 Llama-7B 为核心？如果 GPT-2 only，如何论证 scalability？如果包含 Llama-7B，如何解决 exact Hessian 不可计算导致的 TRV ground truth 缺失问题？

---

### 未解决的开放问题（进入 C 时带着的已知风险）

- **H2（IF-RepSim 误差独立性）**：四个视角一致指向这是最弱假设。"值不相关 != 误差不相关"。目前无法先验判断，需要 Phase 2 实验结果回答。 — 影响评估：**高** — 若 IF 和 RepSim 误差高度耦合（在同一类 test point 上同时失效），RA-TDA 的融合增益将消失，Phase 2 的核心 claim 不成立。升级条件：pilot 中若 naive ensemble 已经达到 RA-TDA 的 ~95% 性能，则 H2 事实上为假。

- **TRV 变异度不足**：如果 TRV 分布高度集中，自适应权重 lambda(TRV) 退化为近似常数，AURA 框架退化为 static ensemble。 — 影响评估：**高** — 如果 pilot 中 >80% test point 落入同一 TRV 等级，Phase 1 的诊断叙事和 Phase 2 的自适应叙事同时受损。升级条件：pilot 的 Jaccard@k 降级曲线 std < 0.05。

- **"排序稳定 != 排序正确"悖论**：Theorist 引用 2303.12922 指出，即使 TRV 高（排序稳定），归因排序本身可能就是错的。TRV 高只保证"一致地错"。 — 影响评估：**中** — 这不阻塞 Phase 1 的发表（"稳定性"本身是有信息量的诊断），但限制了 TRV 作为"归因质量"指标的解释力。C 阶段需要在叙事中区分"stability"和"correctness"。升级条件：如果高 TRV 子集的 LDS 不显著高于低 TRV 子集。

- **Phase 2 LLM 工程复杂度**：GPT-2 上的 IF 计算已经是工程挑战，Llama-7B 更是深水区。TRAK 的 LLM 支持有限，可能需要大量 monkey-patching。 — 影响评估：**中** — 有明确 fallback（GPT-2 only），但会削弱 scalability 叙事。升级条件：如果 GPT-2 上的实验结果需要 Llama-7B 验证才能说服审稿人。
