## [Contrarian] 反对者视角

### 假设挑战

- **假设 C1: problem-statement.md 的 FM1/FM2 框架与项目实际证据基础完全脱节**

  problem-statement.md 提出了一个全新的研究方向：FM1 (Signal Dilution) + FM2 (Common Influence Contamination) 信号处理诊断框架，目标是在 DATE-LM benchmark 上对比 representation-space vs parameter-space TDA 方法。然而，AURA 项目的全部实验证据（project.md, probe_result.md）来自一个完全不同的方向：BSS 诊断、Hessian 灵敏度分析、CIFAR-10/ResNet-18 上的 per-test-point 方法选择。

  具体矛盾：
  - project.md Section 2.4 的三组件设计（Variance Decomposition -> BSS -> Adaptive Selection）在 problem-statement.md 中完全消失
  - problem-statement.md 的 RQ1-RQ3 围绕 DATE-LM + 2x2 ablation，但项目没有任何 LLM 规模的实验数据
  - probe_result.md 的核心发现（BSS-gradient norm rho=0.906、TRV cross-seed instability）与 FM1/FM2 框架无直接关系

  **质疑**：problem-statement.md 实质上是在用 AURA 的 CIFAR-10 数据为一篇完全不同的论文做论证。这不是 "formalize"，而是 "pivot"。如果是 pivot，应当承认原方向（BSS 诊断）的证据不适用于新方向，而非选择性引用。

- **假设 C2: IF-RepSim anti-correlation (tau = -0.467) 不支持 FM1/FM2 独立性**

  problem-statement.md 反复引用 tau = -0.467 作为 "FM1 和 FM2 是独立失效模式" 的证据（Section 1.2: "confirming they capture fundamentally different information"; Section 1.3: "directly supports FM2"）。

  但 IF 和 RepSim 的反相关完全可以由更平凡的原因解释：
  1. IF 在 full-model CIFAR-10 上用 EK-FAC 近似，本身就有大量近似误差（probe_result.md: kappa ~ 1.2-1.4x10^6）
  2. RepSim 在 CIFAR-10 上表现很差（LDS = 0.074 vs IF 的 0.297），反相关可能仅仅反映 "一个方法有信号，另一个是噪声"
  3. 两个随机排名的 Kendall tau 期望为 0，tau = -0.467 可能反映的是 RepSim 系统性地反转了某种 artifact（如类内距离 vs 类间距离的混淆），而非 "捕获不同信号"

  **关键**：要论证 FM1/FM2 独立性，需要在两个方法都有合理表现的设置下测量相关性。在一个方法 LDS 仅 0.074 的设置下，anti-correlation 不具诊断意义。

- **假设 C3: "Five independently proposed representation-space methods form a coherent family" 是过度归纳**

  problem-statement.md 声称 RepSim、RepT、In-the-Wild、Concept IF、AirRep 构成一个 "coherent methodological family"（Section 1.2）。但 Table（Section 1.1）清楚展示了这些方法的巨大差异：
  - RepSim 和 RepT 在 hidden state 空间操作（d ~ 4096）
  - AirRep 用独立 encoder（d_enc ~ 384），与模型内部表示无关
  - Concept IF 的 psi 是 parameter gradient（`nabla_theta f(z_train)`），实质上是混合空间方法
  - In-the-Wild 用 activation difference，编码的是 behavioral 信号而非 representational 信号

  将这 5 个方法统一为 "representation-space family" 并用统一的 FM1/FM2 框架解释，需要论证它们共享相同的优势机制。problem-statement.md 没有提供这个论证——仅仅因为它们 "都不在 parameter space" 不足以构成 family。

### 反事实场景

**如果核心洞察是错的**：FM1 (Signal Dilution) 和 FM2 (Common Influence Contamination) 不是独立失效模式，而是同一个现象（高维近似误差）的两个症状。

具体失败机制：
- Better Hessians Matter 已经证明改善 Hessian 近似质量能系统性提升 LDS（H >= GGN >> EK-FAC >> K-FAC）
- 如果在 DATE-LM 上使用 GGN 级别的 Hessian（而非 LoRA + K-FAC），parameter-space IF 可能直接追平甚至超过 RepSim
- 此时 FM1 不再是 "信号稀释" 而只是 "近似质量不足"，FM2 不再是 "common influence contamination" 而只是 "Hessian error 放大了 pre-training gradient 的权重"
- 2x2 ablation 的结果将是：good Hessian + standard scoring 约等于 any method + contrastive scoring，证明 representation space 没有结构性优势，只有计算效率优势

**最可能的实验失败场景**：

- **场景 1: RepSim 在 DATE-LM 上全面输给 TRAK**。problem-statement.md 自己承认 AURA 数据中 IF 的 LDS (0.297) 远超 RepSim (0.074)。TRAK 在 DATE-LM 上已有 baseline 数据且表现良好。RepSim 从未在 DATE-LM 上被评估——原因可能恰恰是它在 LLM 规模不 work。如果 RepSim LDS < TRAK LDS - 5pp on all three tasks，RQ1 直接 falsified，整个框架的经验基础崩塌。

- **场景 2: Contrastive scoring 在 representation-space 方法上无效（RQ2 falsified），但不是因为 "FM2 already implicitly resolved"，而是因为 representation-space 方法本身太弱（LDS 太低），contrastive 修正没有足够的信号可以增强**。此时 problem-statement.md 的 prediction（"FM2 已被隐式解决"）看起来像是 post-hoc rationalization。

- **场景 3: 2x2 ablation 显示强交互效应（interaction > 30% of minimum main effect），RQ3 falsified**。FM1 和 FM2 不可分离意味着信号处理框架的核心假设（两个独立缺陷）是错的，整个理论叙事需要重写。

### 被低估的竞争方法

**有** — 至少三个方向被严重低估：

1. **Better Hessians Matter (2509.23437) 的直接扩展到 LLM 规模**。problem-statement.md 试图将其限制在 "low-dimensional settings"（Section 1.3: "Digits/MLP, <1M parameters"），但该论文的核心发现（H >= GGN >> EK-FAC >> K-FAC）是关于近似质量的序关系，没有理由认为这个序关系在 LLM 规模会反转。如果有人直接在 DATE-LM 上实现 GGN 级别 Hessian（即使是近似的），可能直接消除 "representation space 更优" 的叙事。

2. **LESS (2024) 和 DDA (2024) 的组合方法**。DDA 的 contrastive scoring 已经在 hallucination tracing 上达到 93.49% AUC。LESS 在 instruction tuning 任务上表现强劲。两者的组合（contrastive + gradient-based selection）可能已经覆盖了 problem-statement.md 想要的 2x2 矩阵中最强的那个 cell (parameter-space + contrastive)，且无需理论框架。

3. **TrackStar (2025) 和 SOURCE (2025)**。这些最新的 parameter-space 方法使用 multi-checkpoint aggregation，可能已经隐式地缓解了 FM1（通过跨 checkpoint 平均降噪）和 FM2（通过 trajectory-based scoring 去除 common mode）。problem-statement.md 在 Section 1.1 列出了它们但没有分析它们是否已经解决了 FM1/FM2。

### 生死线评估

**生死线 1: RepSim 在 DATE-LM 上的 LDS**

- 如果 RepSim LDS < TRAK LDS - 5pp on all three DATE-LM tasks：**不值得发表**。框架的核心叙事（representation-space 方法因避免 FM1/FM2 而优于 parameter-space）被实证否定。仅靠 "diagnostic framework"（无正面实验支撑）不足以支撑顶会论文。
- 如果 RepSim LDS >= TRAK on at least 1 task：**有条件值得发表**，但需要解释为什么只在部分任务上成立。

**生死线 2: 2x2 Ablation 的清晰度**

- 如果 interaction term > 30% of minimum main effect（RQ3 falsification criterion）：**不值得以 "独立失效模式" 的叙事发表**。可以退化为 empirical benchmark paper，但贡献天花板降至 workshop level。
- 如果 main effects < 3pp absolute：**不值得发表**。信号处理框架看起来正确但实际效应微不足道。

**生死线 3: Scale gap 的可信度**

- 如果论文仅在 CIFAR-10/ResNet-18 上验证 FM1/FM2 框架，而 DATE-LM 实验仅有 RepSim vs TRAK 的单一比较（无完整 2x2 ablation）：**不值得以顶会论文发表**。CIFAR-10 上的发现无法泛化到 LLM 的论证缺口太大。NeurIPS reviewer 会直接质疑 scale gap。
- 最低要求：至少在 Pythia-1B 或同等规模上完成完整 2x2 ablation + RepSim/RepT vs TRAK/IF comparison on >= 2 DATE-LM tasks。

**生死线 4: 与原 AURA 方向的关系**

- 如果 problem-statement.md 的 FM1/FM2 方向和 project.md 的 BSS 方向被当作同一篇论文提交：**不值得发表**。两个方向的叙事逻辑互不兼容（BSS 是 "per-test-point Hessian diagnostic"，FM1/FM2 是 "representation-space family unification"），硬拼会导致论文缺乏连贯性。必须明确选择一个方向。

### 维度评分

| 维度 | 评分 (1-10) | 关键理由 |
|------|------------|---------|
| 1. Gap 真实性与推导系统性 | 5 | Gap 真实存在（无统一评估）但推导依赖于 CIFAR-10 证据为 LLM-scale 结论做论证，scale gap 未被认真对待 |
| 2. Gap 重要性与贡献天花板 | 6 | TDA 是热门方向，统一评估有价值，但纯 diagnostic + benchmark 论文天花板有限 |
| 3. Gap 新颖性 + 竞争态势 | 5 | FM1/FM2 命名新颖，但 DDA (contrastive)、Li et al. (iHVP degeneracy)、Better Hessians (approximation quality) 已各自覆盖核心观察的不同侧面 |
| 4. Root Cause 深度 | 4 | FM1/FM2 的 "独立性" 未被验证（仅有 CIFAR-10 anti-correlation），"signal processing perspective" 是重新包装而非新发现 |
| 5. 攻击角度可信度 | 4 | 选择最保守角度 A（diagnostic + benchmark），但 RepSim 在 DATE-LM 上完全未验证，风险极高 |
| 6. RQ 可回答性与可证伪性 | 7 | RQ1-RQ3 定义清晰，falsification criteria 明确且合理 |
| 7. 探针结果整合质量 | 3 | 探针结果全部来自 BSS/Hessian 方向（CIFAR-10），被选择性引用支持一个完全不同的方向（FM1/FM2/DATE-LM），整合不诚实 |

### 总结判断

problem-statement.md 实质上是一次未被明确标记的方向转换（pivot）。原 AURA 方向（BSS diagnostic on CIFAR-10）的证据被选择性重新解读以支持一个新方向（FM1/FM2 framework on DATE-LM），但两个方向之间的逻辑桥梁薄弱。最大风险不是某个假设错误，而是**整个论证结构建立在 scale gap 上空**：CIFAR-10 的 anti-correlation 不能推导 LLM-scale 的 FM1/FM2 独立性。建议在推进 RQ1-RQ3 之前，先完成 DATE-LM probe（Section 3.2），获得至少一个 LLM-scale 的正面信号。
