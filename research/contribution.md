# Contribution Tracker: AURA

> 本文档跨阶段维护，记录项目贡献的演化过程。

---

## 贡献列表

### Startup 初始化

| # | 贡献 | 类型 | 来源阶段 | 状态 |
|---|------|------|---------|------|
| C0 | TRV (TDA Robustness Value) 作为归因可信度诊断工具——量化 test point 级别的归因稳定性，填补 TDA 领域 meta-level 空白 | 问题定义 + 诊断工具 | Startup | 初始 |
| C1 | RA-TDA 自适应融合框架——基于 TRV 信号原则性融合 IF 与 RepSim，借鉴 DR 估计 | 方法创新（条件推进） | Startup | 初始 |
| C2 | SI-TRV 理论连接——Self-Influence (Lipschitz bound) 与经验 TRV 的关系验证 | 理论分析 | Startup | 初始 |

### Probe 探针更新 (2026-03-17)

| # | 贡献 | 变更 | 依据 |
|---|------|------|------|
| C0 | TRV 诊断工具 | **核心 motivation 验证通过，但 cross-seed 稳定性未通过** | Jaccard@10 从 Full GGN 到 KFAC 骤降至 ~0.48，确认 Hessian 近似对归因影响巨大。但 TRV 跨 seed Spearman ρ ≈ 0，说明 TRV 是 model-instance 级而非 test-point 级属性。需修正 TRV 定义（ensemble TRV 或 continuous TRV） |
| C1 | RA-TDA 融合 | **暂缓——Phase 1 稳定性问题未解决** | TRV 本身不可靠时，用 TRV 引导融合权重的增量贡献存疑 |
| C2 | SI-TRV 理论连接 | **H4 不成立** | SI-TRV Spearman ρ ≈ 0（p > 0.05），SI 捕获的"分布偏移敏感性"与 TRV 的"Hessian 近似敏感性"正交。需放弃 SI 作为 TRV 代理的叙事 |

---

## 贡献评估

### 整体发表价值评估

| 评估维度 | Startup 评级 | Probe 后评级 | 论据 |
|---------|-------------|-------------|------|
| Novelty | 中-高 | 中-高 | Core motivation validated：Hessian 近似确实大幅改变归因排序。Meta-level 重定义仍有独特价值 |
| Significance | 中 | 中-低 | Cross-seed 不稳定性削弱了 TRV 作为通用诊断工具的 significance。需要修正定义后重新评估 |
| 与目标会议匹配度 | 高 | 中-高 | 仍匹配，但 negative results（H4 不成立、cross-seed 不稳定）需要 honest reporting + constructive response |

---

## Metadata
- **目标会议/期刊**: NeurIPS 2026 / ICML 2026
- **上次更新**: Probe (2026-03-17)
- **当前状态**: Core motivation validated, TRV 定义需修正。Phase 1 有条件推进，Phase 2 暂缓
