# Contribution Tracker: AURA

> 本文档跨阶段维护，记录项目贡献的演化过程。

---

## 贡献列表

### Crystallize 定义 (C v1.1)

| # | 贡献 | 类型 | 来源阶段 | 状态 | 依赖 |
|---|------|------|---------|------|------|
| C0 | TRV (TDA Robustness Value) 作为 per-test-point 归因可靠性诊断工具——操作化为 Jaccard@k 在 Hessian 近似层级链下的稳定性，量化"何时不该信任归因结果" | 问题定义 + 诊断工具 | C | **Probe: 核心 motivation 验证通过，cross-seed 稳定性未通过，需修正定义** | 无（Phase 1 独立成立） |
| C1 | TRV 变异性的经验特征化——首次系统量化 TDA 归因稳定性在 test point 之间的分布特征（均值、方差、条件依赖） | 经验发现 | C | 待后续验证 | C0 |
| C2 | SI-TRV 理论-经验桥接——验证 Self-Influence (Lipschitz bound, 2512.09103 Theorem 5.4) 与经验 TRV 的相关性，为 TRV 提供廉价计算代理 | 理论验证 | C | **Probe: H4 不成立，需放弃 SI 作为 TRV 代理的叙事** | C0 |
| C3 | RA-TDA 自适应融合框架——基于 TRV/SI 信号原则性融合 IF 与 RepSim（条件推进，Phase 2） | 方法创新 | C | **Probe: 暂缓——Phase 1 TRV 稳定性问题未解决** | C0 + C2 + TRV 变异度 Pass |
| C4 | "稳定 != 正确" 分析——定量展示 TRV-high 但归因错误的案例，为 RA-TDA 中 RepSim 补偿的必要性提供证据 | 分析贡献 | C | 初始 | C0 |

### Probe 探针验证详情 (2026-03-17)

| # | 变更 | 依据 |
|---|------|------|
| C0 | **核心 motivation 验证通过，但 cross-seed 稳定性未通过** | Jaccard@10 从 Full GGN 到 KFAC 骤降至 ~0.48，确认 Hessian 近似对归因影响巨大。但 TRV 跨 seed Spearman ρ ≈ 0，说明 TRV 是 model-instance 级而非 test-point 级属性。需修正 TRV 定义（ensemble TRV 或 continuous TRV） |
| C2 | **H4 不成立** | SI-TRV Spearman ρ ≈ 0（p > 0.05），SI 捕获的"分布偏移敏感性"与 TRV 的"Hessian 近似敏感性"正交。需放弃 SI 作为 TRV 代理的叙事 |
| C3 | **暂缓** | TRV 本身不可靠时，用 TRV 引导融合权重的增量贡献存疑 |

---

## 贡献评估

### 整体发表价值评估

| 评估维度 | C 阶段评级 | Probe 后评级 | 论据 |
|---------|-----------|-------------|------|
| Novelty | 中-高 | 中-高 | Meta-level 问题重定义（reliability diagnosis vs method improvement）；Daunce/BIF 已从不同角度（模型扰动/贝叶斯后验）探索归因不确定性，但 Hessian 近似选择敏感性维度无直接先例；跨领域迁移（sensitivity analysis -> TDA）无竞争。Core motivation validated：Hessian 近似确实大幅改变归因排序 |
| Significance | 中-高 | 中-低 | C 阶段：服务于所有 IF/TRAK/SOURCE/ASTRA 方法用户，回应 TDA 评估方法论危机 (2303.12922, G4)。Probe 后：Cross-seed 不稳定性削弱了 TRV 作为通用诊断工具的 significance，需修正定义后重新评估 |
| 与目标会议匹配度 | 高 | 中-高 | NeurIPS/ICML 接受 meta-methodology + evaluation + new metric 贡献；但 negative results（H4 不成立、cross-seed 不稳定）需要 honest reporting + constructive response |

### Phase 分层与贡献天花板

- **Phase 1 (最小论文单元)**：C0 + C1 + C2 + C4 = TRV 定义 + 经验分布 + SI 代理 + 稳定性 != 正确性分析
  - **C 阶段天花板**：Solid NeurIPS/ICML poster（经验分析 + 新诊断指标）
  - **Probe 后修正**：C2 (SI 代理) 不成立，C0 需修正定义。Phase 1 仍可行但需重构
- **Phase 1 + Phase 2**：C3 = RA-TDA diagnostic-guided adaptive ensemble
  - **当前状态**：暂缓，需 TRV 稳定性问题先解决
- **目标 venue 匹配度**：NeurIPS/ICML poster 可期，TMLR 也合适

---

## Metadata
- **目标会议/期刊**: NeurIPS 2026 / ICML 2026
- **上次更新**: Probe (2026-03-17)
- **当前状态**: Core motivation validated, TRV 定义需修正（ensemble TRV 或 continuous TRV）。Phase 1 有条件推进，Phase 2 暂缓。已补充 Daunce/BIF 差异化论证
