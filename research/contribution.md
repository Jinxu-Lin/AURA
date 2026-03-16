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

---

## 贡献评估

### 整体发表价值评估

| 评估维度 | 评级 | 论据 |
|---------|------|------|
| Novelty | 中-高 | Meta-level 问题重定义（reliability diagnosis vs method improvement）；跨领域迁移（CI→TDA）无直接竞争 |
| Significance | 中 | 服务于所有 TDA 方法的可靠性评估；但取决于 TRV 的实际信息量 |
| 与目标会议匹配度 | 高 | NeurIPS/ICML 接受 meta-methodology 和 evaluation 贡献 |

---

## Metadata
- **目标会议/期刊**: NeurIPS 2026 / ICML 2026
- **上次更新**: Startup
- **当前状态**: 贡献不足（需 pilot 验证核心假设）
