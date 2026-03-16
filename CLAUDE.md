# AURA — Adaptive Unified Robust Attribution

> 项目入口文档。自动化运行器通过 Noesis 路径定位 Praxis orchestrator。

## 项目概览

- **一句话描述**: 量化 TDA 归因结果对 Hessian 近似的敏感性（TRV），并基于稳定性信号自适应融合 IF 与 RepSim（RA-TDA）
- **论文标题**: Beyond Point Estimates: Sensitivity-Aware Training Data Attribution
- **目标会议/期刊**: NeurIPS 2026 / ICML 2026
- **Noesis 路径**: `~/Research/Noesis`

## 当前状态

- **当前阶段**: C (Crystallize)
- **执行模式**: 首次
- **下一步**: 运行 `/praxis-research ~/Research/AURA` 继续自动化流程
- **阶段历史**: S ✓ (Go with focus) →

> 阶段状态由 `pipeline-status.json` 权威记录，本节仅供人类快速查阅。

## 关键文档

| 文档 | 状态 | 说明 |
|------|------|------|
| `project-startup.md` | ✓ | S — 项目知识基础 |
| `research/problem-statement.md` | | C — Gap + 攻击角度 + 探针方案 |
| `research/probe-results.md` | | P — 探针实验结果 |
| `research/method-design.md` | | D — 方法设计 |
| `research/experiment-design.md` | | D — 实验设计 |
| `research/contribution.md` | ✓ | 跨阶段 — 贡献跟踪 |
| `research/result.md` | | E — 实验结果与洞察 |
| `iteration-log.md` | | 版本变更历史 |

## 核心设计决策

1. **Phase 1-2 解耦**：Phase 1 (TRV 诊断) 是最小论文单元，可独立发表。Phase 2 (RA-TDA 融合) 条件推进——需 pilot 验证 TRV 变异度 + 显著超越 naive ensemble。
2. **必要 baseline**：W-TRAK + naive ensemble（固定权重 IF+RepSim 平均）是 Phase 2 核心对照。
3. **LLM 规模**：GPT-2 为保底方案，Llama-7B 为 stretch goal。

## 迭代记录

| 版本 | 触发 | 变更 | 日期 |
|------|------|------|------|
| | | | |

---

## Codes/ 子目录

> I（实现规划）时填写。

---

## Papers/ 子目录

> W（论文写作）时填写。
