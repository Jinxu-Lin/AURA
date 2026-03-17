# 项目: AURA

## 研究主题
Beyond Point Estimates: Sensitivity-Aware Training Data Attribution — 提出 TDA Robustness Value (TRV) 量化每个 test point 的归因稳定性，并基于稳定性信号自适应融合梯度方法 (IF) 与表示方法 (RepSim)。

## 背景与动机

### 核心问题
TDA 方法的归因结果对 Hessian 近似质量高度敏感，但目前：
1. **无诊断工具**: 没有方法量化"这个 test point 的归因是否可信"
2. **无原则性利用**: 即使知道不可信，也没有方法自动切换到更可靠的归因

### 评估可靠性危机
- Spearman miss-relation (2303.12922): 相同相关值可掩盖质量差异
- LOO 信噪比低 (2506.12965): 大数据集中单样本删除效应接近噪声
- Hessian 层级 (2509.23437): LDS 严格遵循 H ≥ GGN ≫ EK-FAC ≫ K-FAC

### 参数空间 vs 表示空间的张力
- IF/TRAK (参数空间): 有因果理论基础，但 Hessian 近似是致命瓶颈。LoRA+低 SNR 下完全失效 (0-7%)
- RepSim/RepT (表示空间): 无因果基础，但有害数据检测 96-100%，无需 Hessian
- 两类方法相关性仅 0.37-0.45 (2602.14869)，捕获不同信息

### AURA 两层解决方案
- **Phase 1 (TRV)**: 量化每个 test point 的归因在 Hessian 近似扰动下的稳定性（最小论文单元）
- **Phase 2 (RA-TDA)**: 基于 TRV 自适应融合 IF + RepSim（条件推进）

## 初始想法

### 已完成的前期工作（来自 ~/Research/AURA）

项目已独立完成 Startup → Crystallize → Strategic Review → Probe 四个阶段：

#### 6-Agent Idea Debate 结论
- **判定**: Go with focus（方向修正）
- 6 视角一致认为 Phase 1 (TRV) 有独立发表价值
- 4/6 视角指出 H2（IF-RepSim 误差独立性）是最弱假设
- Codex 外审 5/10，要求解决"诊断无 ground truth"和"substitution problem"

#### Probe 实验关键发现 (CIFAR-10/ResNet-18, 3 seeds)

**正面信号**:
- Hessian 近似对归因影响显著: Jaccard@10 从 1.0 (Full GGN) 骤降至 ~0.45 (KFAC) → 0.33 (Identity)
- TRV 呈三模态分布 (Level 1: 38-65%, Level 2: 11-40%, Level 5: 19-22%)，通过 ≥3 levels >10% 检验
- TRV 不是 OOD detector (|ρ| < 0.32)，捕获独立信号维度
- 条件数 κ ≈ 1.1-1.4 × 10⁶，确认谱放大显著

**致命问题**:
- **Cross-seed TRV 不稳定 (CRITICAL)**: 不同 seed 间 TRV Spearman ρ ≈ -0.006（基本为零）。TRV 是 (模型实例, test point) 联合属性，不是 test point 内在属性
- **SI-TRV 无相关性**: Self-Influence 无法作为 TRV 的廉价代理（H4 不成立）
- **Per-point 方差不足**: std 0.053-0.082 < 0.15 阈值，TRV 只能做粗分类
- **Hessian 层级底部坍缩**: Diagonal ≈ Damped Identity ≈ Identity，last-layer 设置区分度不足

#### 关键决策点
Probe 验证了核心动机（Hessian 近似影响归因），但暴露了 TRV 定义的根本性问题。方向选择：
1. 修正 TRV 定义（Ensemble TRV / 连续 TRV / full-model Hessian chain）
2. 重新定位叙事（从"per-test-point diagnostic"到"per-model-instance diagnostic"）
3. 放弃 Phase 2，聚焦 Phase 1 作为经验性表征 + 诊断工具

### 核心假设清单
- **H1**: 归因敏感性因 test point 而异且可预测 — 部分验证（有区分度但跨 seed 不稳定）
- **H2**: IF 和 RepSim 误差近似独立 — 未验证（最弱假设）
- **H3**: 低 TRV 点上 RepSim 仍有用 — 未验证
- **H4**: SI 是 TRV 有效代理 — **已否证**
- **H5**: Hessian 近似误差与分布偏移独立 — 未验证

### 研究问题
- **RQ1**: TRV 在 full-model Hessian 层级（非 last-layer）下是否有更高跨 seed 稳定性？
- **RQ2**: Ensemble TRV（多 seed 平均）能否产生稳定的 per-test-point 诊断？
- **RQ3**: 高 TRV 子集的 LDS 是否显著高于低 TRV 子集？
- **RQ4**: RA-TDA 自适应融合是否显著优于 naive ensemble 和单一方法？

### 预注册否证条件
- Full-model Hessian chain 下跨 seed TRV ρ < 0.3 → TRV 作为诊断工具不可行
- Ensemble TRV 的 LDS 分层效应 (high vs low) Cohen's d < 0.3 → TRV 无诊断价值
- RA-TDA vs naive ensemble LDS 提升 < 2% absolute → Phase 2 不成立

## 关键参考文献
- Hong et al. (2025), 2509.23437: Better Hessians Matter — Hessian 近似层级证据（Phase 1 基础设施）
- Li et al. (2024), 2512.09103: Natural W-TRAK — SI/κ 诊断 + 谱放大理论
- Li et al. (2025), 2409.19998: IF failure on LLM — LoRA 下 IF 完全失效 (0-7%)
- Kowal et al. (2026), 2602.14869: Concept Influence — IF-RepSim 低相关 (0.37-0.45)
- Mlodozeniec et al. (2025), 2506.12965: d-TDA — IF 的分布性框架
- Chernozhukov et al. (2018), 1608.00060: Doubly Robust Estimation — Phase 2 理论基础
- Cinelli & Hazlett (2020), 1912.07236: Sensitivity analysis framework
- Park et al. (2023), 2303.12922: Spearman miss-relation
- Li et al. (2025), 2410.17413: Attribution ≠ Influence

## 可用资源
- GPU: 4x RTX 4090
- 服务器: default (SSH MCP)
- 远程路径: /home/jinxulin/sibyl_system
- 已有代码: ~/Research/AURA/codes/probe_experiment/ (ResNet-18/CIFAR-10 probe 完整代码)

## 实验约束
- 实验类型: 轻量训练 (ResNet-18, GPT-2)
- 模型规模: 小-中 (ResNet-18, GPT-2, 可选 Llama-7B)
- 时间预算: Phase 1 pilot 实验 ~15min/task, Full 实验 ~1h/task
- 数据集: CIFAR-10 (Phase 1), DATE-LM (Phase 2)

## 目标产出
- NeurIPS 2026 / ICML 2026 论文
- Phase 1 + Phase 2（条件推进）

## 特殊需求
- 已有 probe 实验代码和结果需迁移到 workspace
- 需要 Sibyl 重新评估：鉴于 cross-seed TRV 不稳定的发现，项目方向是否需要 PIVOT
- full-model Hessian chain 实验需要 2509.23437 的代码（需 git clone + 适配）
