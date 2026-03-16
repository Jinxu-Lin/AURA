# Probe Experiment Results — Phase 1 Pilot

> **实验日期**: 2026-03-17
> **实验环境**: 4× RTX 4090, conda env `sibyl_CRA` (PyTorch 2.5.1, TRAK 0.3.2)
> **代码位置**: `codes/probe_experiment/`
> **结果位置**: `codes/probe_experiment/outputs/attributions/`

## 实验设计

### 目标

验证 AURA Phase 1 的核心假设：TDA 归因结果对 Hessian 近似质量的敏感性是否因 test point 而异，且有足够区分度支撑自适应框架。

### 配置

| 参数 | 值 |
|------|-----|
| 数据集 | CIFAR-10 |
| 模型 | ResNet-18 (CIFAR-adapted: 3×3 conv1, no maxpool) |
| 训练 seeds | 42, 123, 456 |
| 模型精度 | 95.12%, 94.93%, 95.06% (test) |
| ID test points | 100 (50 high-confidence + 50 low-confidence, correctly classified) |
| OOD test points | 20 (SVHN) |
| 训练集规模 | 50,000 (full CIFAR-10 train) |
| k (top-k) | 10 |
| Jaccard 阈值 | 0.5 |

### Hessian 近似层级

基于 last-layer (fc, 512→10, 5130 params) 的解析梯度计算，5 个近似层级：

| Level | 方法 | 描述 |
|-------|------|------|
| 1 | Full GGN | 完整 empirical Fisher：$(1/n) G^T G + \epsilon I$，直接求逆 |
| 2 | KFAC | Kronecker 分解近似：$A^{-1} \otimes B^{-1}$（W block）+ 独立 b block |
| 3 | Diagonal | GGN 对角元素 |
| 4 | Damped Identity | 标量缩放 Identity：$\alpha = \text{tr}(GGN)/d$ |
| 5 | Identity | 纯梯度点积（无 Hessian 修正） |

---

## 核心结果

### 1. 条件数 κ

| Seed | κ |
|------|---|
| 42 | 1.22 × 10⁶ |
| 123 | 1.13 × 10⁶ |
| 456 | 1.36 × 10⁶ |

**结论**: κ ≫ 10³，与 2512.09103 报告的 κ ≈ 2.71×10⁵ 量级一致（差异可能来自 CIFAR-adapted 架构细节）。确认谱放大显著，Phase 1 诊断是有必要的。

### 2. Jaccard@10 降级曲线 ✅ PASS

**否证条件**: 相邻等级间的平均 Jaccard@10 > 0.85 → Hessian 近似对归因影响不够大。

| Level | Seed 42 | Seed 123 | Seed 456 |
|-------|---------|----------|----------|
| full_ggn (self) | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| kfac | 0.456 ± 0.162 | 0.532 ± 0.147 | 0.447 ± 0.176 |
| diagonal | 0.337 ± 0.149 | 0.365 ± 0.158 | 0.372 ± 0.180 |
| damped_identity | 0.334 ± 0.151 | 0.351 ± 0.152 | 0.361 ± 0.175 |
| identity | 0.334 ± 0.151 | 0.351 ± 0.152 | 0.361 ± 0.175 |

**关键发现**:
- **Full GGN → KFAC 是最大跳变**：Jaccard 从 1.0 骤降至 ~0.45-0.53，远低于 0.85 否证线。Hessian 近似质量对归因排序影响巨大。
- **Diagonal 及以下层级高度相似**：Diagonal ≈ Damped Identity ≈ Identity（Jaccard 差异 < 0.02）。Hessian 层级的"梯度效应"集中在 full GGN ↔ KFAC 之间，后续层级近乎退化。
- **层级链不够细粒度**：5 层中实质上只有 2-3 个有效区分度（Full GGN / KFAC / 其余）。

### 3. TRV 分布 ✅ PASS（但有结构性问题）

**否证条件**: >80% test point 具有相同 TRV → H1 不成立。

| TRV Level | Seed 42 | Seed 123 | Seed 456 |
|-----------|---------|----------|----------|
| 0 (unstable) | 0% | 0% | 0% |
| 1 (full_ggn) | 59% | 38% | 65% |
| 2 (kfac) | 21% | 40% | 11% |
| 3 (diagonal) | 1% | 3% | 2% |
| 4 (damped_id) | 0% | 0% | 0% |
| 5 (identity) | 19% | 19% | 22% |

**≥3 个等级各含 >10% 的 test points**: 3/3 seeds 通过（Level 1, 2, 5 各超过 10%）。

**关键发现**:
- **三模态分布**: TRV 集中在 Level 1（~38-65%）、Level 2（~11-40%）、Level 5（~19-22%），中间层级（3-4）近乎空白。
- **~20% 的 test points "免疫"于所有 Hessian 近似**: 这些 TRV=5 的点，即使用最粗糙的梯度点积，top-10 排序仍与 Full GGN 有 ≥50% 重叠。
- **不同 seed 间 TRV 分布形状一致**（三模态），但各 level 占比波动大（Level 1 从 38% 到 65%）。
- **Level 3-4 的有效塌缩**验证了上述 Jaccard 降级曲线的发现——当前 Hessian 层级链在 last-layer 设置下有效区分度不足。

### 4. Per-point Jaccard 方差 ❌ FAIL

**目标**: Jaccard@10 的 per-point std > 0.15（TRV 作为路由信号有意义）。

| Seed | Mean per-point std |
|------|-------------------|
| 42 | 0.054 |
| 123 | 0.082 |
| 456 | 0.053 |

**结论**: 方差过低（0.05-0.08 vs 0.15 阈值）。虽然存在 per-point 差异（从 heatmap 可见），但差异幅度不足以让 TRV 成为精细的路由信号。TRV 更像是一个粗粒度的"类型标签"（稳定/不稳定/免疫）而非连续诊断值。

### 5. SI-TRV 相关性 ❌ H4 不支持

| Seed | Spearman(SI, TRV) | p-value | Spearman(1/SI, TRV) | p-value |
|------|-------------------|---------|---------------------|---------|
| 42 | 0.043 | 0.672 | -0.043 | 0.672 |
| 123 | -0.180 | 0.073 | 0.180 | 0.073 |
| 456 | -0.114 | 0.260 | 0.114 | 0.260 |

**结论**: SI 与 TRV 之间几乎没有相关性。H4（SI 是 TRV 的有效代理）不成立。这意味着：
- 2512.09103 的 leverage score（SI）捕获的信息与 Hessian 近似稳定性（TRV）正交
- SI 反映的是"测试分布偏移下的敏感性"，TRV 反映的是"Hessian 近似误差下的敏感性"——两者确实测量不同的东西
- Phase 0 不能用 SI 快速预测 TRV，TRV 必须直接计算

### 6. Cross-seed TRV 稳定性 ❌ CRITICAL FAIL

**目标**: 不同 training seed 间 TRV 的 Spearman ρ > 0.6。

| Seed Pair | Spearman ρ | p-value |
|-----------|-----------|---------|
| 42 vs 123 | -0.023 | 0.822 |
| 42 vs 456 | -0.073 | 0.472 |
| 123 vs 456 | 0.076 | 0.451 |
| **Mean** | **-0.006** | — |

**结论**: TRV 在不同 training seed 间的 rank correlation 基本为零。这是**最严重的问题**：

- TRV 不是 test point 的内在属性，而是高度依赖于特定模型实例的 Hessian landscape
- 同一个 test point 在不同 seed 训练的模型上可以有完全不同的 TRV
- 这意味着 TRV 作为"归因可信度诊断工具"的泛化性存疑——它诊断的不是"这个 test point 的归因是否可信"，而是"这个模型在这个 test point 上的归因是否对 Hessian 敏感"，后者是一个更弱、更不稳定的 claim

### 7. Confidence 分层

| Seed | High conf TRV (mean/median) | Low conf TRV (mean/median) | Mann-Whitney p |
|------|----------------------------|---------------------------|----------------|
| 42 | 2.12 / 1.5 | 1.86 / 1.0 | 0.127 |
| 123 | 2.36 / 2.0 | 2.08 / 1.5 | 0.078 |
| 456 | 1.98 / 1.0 | 2.08 / 1.0 | 0.441 |

**结论**: 高置信点有略高的 TRV 趋势（可能因为梯度更小，近似误差影响更小），但差异不显著（p > 0.05）。置信度不是 TRV 的有效预测器。

### 8. OOD Confound ✅ PASS

| Seed | ID TRV (mean) | OOD TRV (mean) | Spearman(OOD, TRV) | |rho| < 0.8? |
|------|--------------|----------------|---------------------|-------------|
| 42 | 1.99 | 1.00 | -0.315 | PASS |
| 123 | 2.22 | 1.45 | -0.246 | PASS |
| 456 | 2.03 | 1.45 | -0.157 | PASS |

**结论**: OOD 点倾向于有更低的 TRV（更不稳定），但相关性不高（|ρ| < 0.32）。TRV 不是简单的 OOD detector。

---

## 汇总判定

| 检查项 | 结果 | 严重程度 |
|--------|------|----------|
| H1: TRV 分布有区分度 (≥3 levels >10%) | ✅ PASS | — |
| Early signal: Hessian 影响归因 (adj Jaccard < 0.85) | ✅ PASS | — |
| Per-point 方差 (std > 0.15) | ❌ FAIL | Medium |
| H4: SI 是 TRV 代理 | ❌ FAIL | Medium |
| Cross-seed TRV 稳定性 (ρ > 0.6) | ❌ FAIL | **Critical** |
| OOD confound (|ρ| < 0.8) | ✅ PASS | — |
| Phase 1 viable? | **Conditional** | — |

---

## 深度分析与解读

### 正面信号

1. **Hessian 近似对归因影响是实质性的**: Jaccard@10 从 Full GGN 到 KFAC 骤降至 ~0.45-0.53，说明超过一半的 top-10 归因在 Hessian 近似变化后发生了改变。这验证了 AURA 的核心 motivation——归因结果对 Hessian 质量高度敏感。

2. **存在"免疫"亚群**: ~20% 的 test points 在所有近似层级下都保持相对稳定的归因排序（TRV=5）。这些点的归因结果更可信，TRV 作为可信度标签有诊断价值。

3. **TRV 不是 OOD detector**: TRV 与 OOD 的相关性低，说明它捕获了 OOD detection 以外的信息——归因对 Hessian 的敏感性确实是一个独立的信号维度。

### 核心问题

1. **Cross-seed 不稳定是根本性问题**: TRV 完全依赖于特定模型实例，不是 test point 的内在属性。这动摇了"TRV 作为 per-test-point 诊断工具"的叙事。可能的解释：
   - 不同 seed 的模型学习了不同的特征表示，导致同一 test point 在不同 Hessian landscape 中的稳定性不同
   - Last-layer 梯度高度依赖于最终层特征，而特征空间在不同 seed 间存在旋转/置换对称性
   - **可能的缓解**: 使用 ensemble average TRV（多个 seed 的 TRV 平均）作为更稳定的度量

2. **Hessian 层级链的有效退化**: 5 个层级中只有 Full GGN 和 KFAC 产生实质不同的归因，后三个层级（Diagonal / Damped Identity / Identity）高度相似。这可能是 last-layer-only 设置的特点：
   - Last layer 的 Hessian 结构相对简单（5130 维），Diagonal 可能已经捕获了大部分信息
   - 对于 full-model IF（如 LiSSA、EK-FAC 全层），层级间差异可能更大
   - **建议**: 后续实验应使用 full-model Hessian 近似（如 2509.23437 的代码），而非仅 last-layer

3. **Per-point 方差不足**: std ~0.05 说明虽然存在 TRV 的 per-point 差异，但差异幅度有限。TRV 更适合作为粗粒度的类型标签（3 类），而非连续的路由信号。这对 Phase 2 的 λ(TRV) 设计有影响——可能只需要 3 档权重而非连续函数。

### 与辩论预期的对比

| 辩论预测 | 实际结果 | 评估 |
|----------|----------|------|
| Contrarian: TRV 可能高度集中 | 三模态分布，有区分度 | **比预期好** |
| Empiricist: 需要 std > 0.15 | std ≈ 0.05 | **不及预期** |
| Empiricist: cross-seed ρ > 0.6 | ρ ≈ 0 | **远不及预期** |
| Pragmatist: SI 可作为快速代理 | SI-TRV 无相关 | **不及预期** |
| Empiricist: OOD confound | 不存在 | **符合预期** |

---

## 对后续工作的影响

### Phase 1 修正方向

1. **TRV 的定义需要修正**: 当前的 per-model TRV 不稳定。建议探索：
   - **Ensemble TRV**: 在多个 seed 上计算 TRV 后取平均/众数
   - **Continuous TRV**: 用 Jaccard 曲线下面积（AUC）替代离散阈值
   - **Full-model Hessian 层级**: 使用 2509.23437 的 full-model Hessian chain，增加层级间区分度

2. **层级链需要重新设计**: 当前 5 级链中 3 级退化为等价。需要：
   - 引入 Block-diagonal GGN（介于 Full GGN 和 Diagonal 之间）
   - 使用不同正则化强度的 damped GGN（λ = 0.01, 0.1, 1, 10）替代离散层级
   - 使用 TRAK 的不同 projection dimension 作为 continuous Hessian quality proxy

3. **叙事调整**: TRV 的价值可能不是"per-test-point 诊断"，而是"per-model-instance 诊断"——告诉用户"这个特定模型的归因在这些 test points 上不可信"。这是一个更弱但仍然有用的 claim。

### Phase 2 影响

- Cross-seed 不稳定性对 Phase 2 的 λ(TRV) 自适应融合是根本性挑战
- 如果 TRV 本身不可靠，用它引导融合权重的增量贡献可能有限
- 建议在解决 Phase 1 的稳定性问题之前暂缓 Phase 2

### 下一步行动

1. **[High Priority]** 在 full-model Hessian 设置下重复 pilot，使用 2509.23437 的 Hessian hierarchy
2. **[High Priority]** 测试 Ensemble TRV（3-seed 平均）的跨 seed 稳定性
3. **[Medium]** 扩展到 CIFAR-100 / ResNet-50 验证
4. **[Medium]** 探索连续 TRV 定义（Jaccard AUC、扰动版 TRV）
5. **[Low]** 理论分析：为什么 SI 和 TRV 不相关？两者测量的"稳定性"有什么本质区别？

---

## 附录：可视化

所有图表位于 `codes/probe_experiment/outputs/attributions/plots/`:

- `jaccard_degradation.png`: Jaccard@10 降级曲线（按 confidence 分层）
- `trv_distribution.png`: TRV 等级分布直方图
- `jaccard_heatmap.png`: Per-point Jaccard@10 热力图（按 TRV 排序）
- `si_vs_trv.png`: SI vs TRV 散点图
- `cross_seed_trv.png`: Cross-seed TRV 稳定性散点图
- `ood_comparison.png`: ID vs OOD TRV 分布箱线图
