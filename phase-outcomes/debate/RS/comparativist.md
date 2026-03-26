## [Comparativist] 文献对标者视角 — 第二轮

### 修订响应性评估

v1.1 在 §1.1 中增加了 Daunce (2505.23223) 和 BIF (2509.26544) 的专门段落（"归因不确定性量化的已有探索"），在 §1.2 中增加了"与 Daunce/BIF 的差异化"子节。DR 叙事已修正为 "diagnostic-guided adaptive ensemble"（§2.2）。第一轮的两个必须修改项在结构上已响应。

### SOTA 定位（更新）

**绝对 SOTA（归因精度）**：无变化——ASTRA (NeurIPS 2025) 和 TrackStar (ICLR 2025) 仍是精度方向的领先工作。

**最相近 approach（归因可靠性/不确定性方向）**：

v1.1 已正确引用了三个最相近工作，重新定位如下：

| 工作 | 可靠性信号来源 | 计算成本 | Per-test-point | 主动利用信号 |
|------|---------------|----------|----------------|-------------|
| Daunce (ICML 2025) | 模型扰动（参数噪声注入） | 1次集成推理（M个扰动模型） | 是（loss 协方差方差） | 否（用于归因，非诊断） |
| BIF (ICML 2025) | 后验分布（SGMCMC 采样） | 1次后验采样 | 是（后验方差） | 否（用于归因，非诊断） |
| Natural W-TRAK | 谱结构（Lipschitz bound） | 1次 SI 计算 | 是（SI 值） | 否（被动认证） |
| AURA TRV | Hessian 近似层级链 | M次归因（M=近似方法数） | 是（Jaccard@k） | 是（Phase 2 融合） |

**差异化空间的精细评估**：

v1.1 的差异化论证聚焦于"不确定性来源的维度差异"——TRV 度量方法选择敏感性，Daunce 度量训练随机性，BIF 度量贝叶斯不确定性。这个框架在概念上是清晰的。但有几个微妙问题：

1. **Daunce 的"模型扰动"可以模拟 Hessian 近似选择**：如果 Daunce 的扰动集合中包含了使用不同 Hessian 近似的变体，Daunce 的方差就可以捕获 AURA 的 TRV 信号。当然 Daunce 原文没有这样设计（它的扰动是参数噪声注入），但一个审慎的 reviewer 可能会指出"Daunce 框架的泛化性足以涵盖 AURA 的 use case"。

2. **BIF 的后验方差在概念上更"正确"**：BIF 用 Bayesian 框架将 Hessian 逆替换为后验协方差——这意味着 BIF 的不确定性估计已经隐式包含了 Hessian 近似选择的影响（因为不同的 Hessian 近似对应不同的后验近似）。TRV 用经验 Jaccard 来度量这种影响，BIF 用后验方差——后者有更强的理论基础。

3. **计算成本比较不利于 AURA**：TRV 需要 M 种 Hessian 近似 x N_test x N_train 的归因计算。Daunce 需要 M 个扰动模型的 fine-tuning + 推理。BIF 需要 SGMCMC 采样。在 LLM 规模上，Daunce（已在 GPT 模型上验证）和 BIF（架构无关）都比 TRV（需要完整 Hessian 层级链）更实用。

### 文献覆盖漏洞（更新）

⚠️ **新发现的潜在相关工作**：
- **DualXDA (2402.12118, TMLR 2025)**：SVM-based 稀疏归因。与 AURA 方向不同（效率/稀疏性），但其稀疏归因的天然稳定性可能与 TRV 讨论相关。建议在 related work 中提及。
- **Unified Attribution (2501.18887)**：提出统一 XAI、Data-Centric AI 和 Mechanistic Interpretability 的归因框架。可能影响 AURA 的 framing，但不构成直接竞争。

✅ **第一轮漏洞已修复**：
- Daunce (2505.23223) — 已在 §1.1 和 §1.2 中充分引用和差异化
- BIF (2509.26544) — 已在 §1.1 和 §1.2 中充分引用和差异化
- DR 叙事修正 — 已完成（§2.2）

### 贡献边际（更新）

**实际 delta 的重新评估**：

v1.1 的差异化论证使 Phase 1 的贡献更清晰——"TRV 量化方法选择敏感性这一特定维度"。但贡献边际仍然处于**边缘偏上**水平：

- **如果 TRV 与 Daunce/BIF 信号高度正交**（经验验证）：Phase 1 提供了独特的新信息维度，贡献从"边缘"升为"充分"。
- **如果 TRV 与 Daunce/BIF 信号高度相关**：TRV 退化为冗余信号，贡献不足。
- **无论哪种情况**，Phase 1 的经验贡献（"首次系统量化 TDA 归因在 Hessian 层级链上的 per-test-point 稳定性分布"）有描述性价值，但如果分布退化则信息量有限。

**创新类型判断不变**：有意义的增量改进——新分析视角 + 新诊断指标。

### 并发工作风险（更新）

**风险等级**：中（无变化）

**更新依据**：
- 在线搜索未发现 2025Q4-2026Q1 出现新的直接竞争工作（无人提出与 TRV 类似的 Hessian 层级链稳定性指标）。
- Daunce 和 BIF 团队的后续工作值得跟踪：如果他们在 follow-up 中显式提出 per-test-point 可靠性指标，AURA 的窗口会缩小。
- TDA 方向的 arXiv 活跃度保持稳定（未见爆发式增长），竞争窗口仍估计 6-9 个月。

**差异化策略建议**：AURA 的独特 angle 在于"经验诊断 + 主动利用（Phase 2）"的全链条。如果 Phase 1 能提供 Daunce/BIF 不提供的诊断信息（经验验证其正交性），且 Phase 2 能利用这些信息改善归因，这个全链条是差异化的核心。建议在探针阶段就收集 TRV 与 multi-seed variance（作为 Daunce 代理）的相关性数据。
