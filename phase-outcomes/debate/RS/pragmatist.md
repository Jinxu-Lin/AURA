## [Pragmatist] 务实者视角 — 第二轮

### 修订响应性评估

v1.1 的修订主要在内容层面（补充 Daunce/BIF、修正 DR 叙事），探针方案的工程设计也有改进（增加了 Step 0 代码可行性验证、Step 7 偏相关检验、Step 8 TRV-LDS 对比、Step 9 OOD test points）。这些改进使探针更全面但也更耗时——需要重新评估工程可行性。

### 工程组件拆解（更新）

✓ **CIFAR-10 / ResNet-18 训练管线** — 无变化。PyTorch + torchvision，成熟。

✓ **EK-FAC / K-FAC Hessian 近似** — Hong et al. 代码已开源。v1.1 明确了 Step 0 优先验证代码可行性——这是正确的优先级安排。

△ **Per-test-point 归因排名提取** — 第一轮估计 3-5 天适配，这个估计仍然合理。v1.1 没有改变这个组件的技术要求。关键不确定性：Hong et al. 代码是否支持返回 per-test-point 的完整归因分数向量（而非聚合 LDS）。如果代码架构使用了 batch 矩阵运算 + 即时聚合（计算归因分数后立即 Spearman-average 到 LDS），需要修改数据流以保留中间结果。

△ **Self-Influence SI(z) 计算** — 无变化，2-7 天取决于是否有开源实现。

✗ **TRV 计算 + 统计分析管线（扩展版）** — v1.1 增加了：(a) 梯度 norm 偏相关 (Step 7)；(b) TRV-high vs TRV-low LDS 对比 (Step 8)；(c) OOD 20 points (Step 9)。从头开发估计从 3-4 天增加到 5-6 天。代码本身不复杂，但分析的维度增加了（TRV 分布 + SI 相关 + 偏相关 + LDS 分组对比 + OOD 行为）。

✗ **Ground truth LDS 计算（新增需求）** — Step 8 需要计算 TRV-high 和 TRV-low 两组的 LDS。LDS 需要 leave-one-out retraining 或 counterfactual ground truth。Hong et al. 代码可能已包含 ground truth LDS 计算，但如果需要自行重训练子集模型，这是一个隐藏的重大计算开销。建议确认 Hong et al. 是否提供了预计算的 ground truth 归因。

### 最小 Pilot 设计（更新）

v1.1 的探针方案比第一轮更完整，但也更重。拆分为两个验证阶段更务实：

**Pilot A（核心假设，1-2 天 + 0.5-1 GPU-hour）**：
- Step 0：验证 Hong et al. 代码接口
- 1 seed，50 test points，EK-FAC vs K-FAC
- 输出：Jaccard@10 分布直方图
- 判断：TRV 是否非退化

**Pilot B（完整探针，在 Pilot A Pass 后执行，2-3 天 + 6-8 GPU-hours）**：
- 3 seeds，200 test points，EK-FAC vs K-FAC + Block-GGN vs EK-FAC
- SI 计算 + 偏相关 + TRV-LDS 对比 + OOD
- 输出：完整 §3.3 pass 标准评估

**理由**：如果 Pilot A 就 Fail（TRV 分布退化），无需投入 Pilot B 的 6-8 GPU-hours 和额外工程。

### 工程陷阱（更新）

⚠️ **探针范围膨胀风险**：v1.1 的探针从第一轮的 6 步扩展到 9 步（增加了 Step 0, 7, 8, 9）。每一步的增加都有充分理由，但组合在一起使"3-5 天人工时间"的估计可能偏乐观。更现实的估计：
- Step 0 代码验证：1-2 天
- 模型训练 + 归因计算 + 基本 TRV 分析（Step 1-6）：3-5 天
- 扩展分析（Step 7-9）：2-3 天
- **总计：6-10 天人工时间**（原估计 3-5 天翻倍）

⚠️ **第一轮提出的 Hong et al. 代码适配复杂度**：仍然是最大工程不确定性。v1.1 增加了 Step 0 来提前验证，这是正确的。但需要明确 Step 0 的 kill 条件：如果 Hong et al. 代码不支持 per-test-point 输出且改造工作量 > 2 周，是否考虑放弃 Hong et al. 代码，改用 TRAK 代码库（TRAK 的 API 设计可能更支持 per-sample 归因分数提取）？

⚠️ **Ground truth LDS 的计算成本**：Step 8 的 TRV-high vs TRV-low LDS 对比需要 ground truth 归因（leave-one-out retraining）。在 CIFAR-10/ResNet-18 上，LOO 对 200 test points 需要 200 次完整训练（每次 ~30 min），总计 ~100 GPU-hours。这远超探针预算（6-8 GPU-hours）。**替代方案**：(a) 使用 Hong et al. 论文中 Full Hessian 的归因作为近似 ground truth（如果代码支持）；(b) 使用随机重训练子集（d-TDA 风格）作为近似 ground truth；(c) 降低到 20 个 test point 做 LOO。建议在 §3.2 中明确 ground truth 的获取策略。

### 综合预估（更新）

⏱️ **日历时间（到第一个有意义结果）**：
- Pilot A：1 周（含 Step 0 验证）
- Pilot B（如果 A 通过）：2-3 周
- **总计：3-4 周**

💻 **算力（到第一个有意义结果）**：
- Pilot A：1-2 GPU-hours
- Pilot B：8-15 GPU-hours（含 debug 迭代）
- Ground truth LDS（如果需要 LOO）：可能额外 50-100 GPU-hours

🔧 **主要工程风险**：(1) Hong et al. 代码适配——Step 0 应在 2 天内给出 go/no-go；(2) Ground truth LDS 的获取策略——如果需要 LOO retraining，预算严重超支，需要降级方案。

**对 Phase 2 的可行性评估（无变化）**：Phase 2 仍需 2-3 个月，前提是 Phase 1 代码可复用。v1.1 的 "diagnostic-guided adaptive ensemble" 叙事比 DR 更务实，但 lambda(TRV) 映射函数的设计仍是工程不确定性——是学习还是规则？需要 ablation。
