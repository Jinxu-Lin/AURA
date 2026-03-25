# 文献调研报告

**研究主题**: Beyond Point Estimates: Sensitivity-Aware Training Data Attribution — 提出 TDA Robustness Value (TRV) 量化每个 test point 的归因稳定性，并基于稳定性信号自适应融合梯度方法 (IF) 与表示方法 (RepSim)。

**调研时间**: 2026-03-17

**arXiv 搜索关键词**: training data attribution robustness, influence function fragility, TRAK data attribution, representation similarity attribution, EK-FAC influence scalable, approximate unrolling TDA, GraSS gradient sparsification, DataInf LoRA influence

**Web 搜索关键词**: training data attribution survey benchmark 2024 2025, influence function instability fragility, TRAK TracIn datamodels, AirRep representational optimization NeurIPS 2025, ensemble data attribution adaptive fusion, data attribution stability consistency across methods, LoGra scalable gradient projection

## 1. 领域现状摘要

Training Data Attribution (TDA) aims to quantify the contribution of individual training examples to model predictions. The field has matured significantly since Koh & Liang (2017) introduced influence functions (IF) for deep learning, branching into three major paradigms: (1) **retraining-based methods** (Leave-One-Out, Shapley values, Datamodels) that provide ground-truth attributions but are computationally prohibitive; (2) **gradient-based influence estimators** (IF, TracIn, TRAK, EK-FAC) that approximate counterfactual effects via gradient information; and (3) **representation-based methods** (cosine similarity in embedding space, Representer Points, AirRep) that are fast but lack theoretical grounding for counterfactual prediction.

The field is currently experiencing a tension between **scalability** and **fidelity**. Recent work (LoGra, GraSS, DataInf) has dramatically reduced computational costs, enabling TDA on billion-parameter LLMs. Simultaneously, foundational work has exposed that influence functions in deep learning are **fragile** (Basu et al., 2021) and actually answer a different question than leave-one-out retraining — specifically the Proximal Bregman Response Function (Bae et al., NeurIPS 2022). This means the reliability of attribution scores varies significantly across test points, methods, and hyperparameters, yet no existing work systematically quantifies this per-test-point attribution stability or leverages it to improve attribution quality.

The most recent advances (2024-2025) focus on two fronts: scaling TDA to LLMs (LoGra for Llama3-8B, AirRep for instruction-tuned LLMs) and improving theoretical understanding (Source via approximate unrolling, EK-FAC for multi-stage training). However, **no existing work proposes a per-test-point robustness metric for attribution scores or adaptively fuses different TDA paradigms based on stability signals** — this is the core research gap our proposed TRV approach targets.

## 2. 核心参考文献

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 1 | Understanding Black-box Predictions via Influence Functions (Koh & Liang) | ICML | 2017 | 将经典统计学 IF 引入深度学习，通过 Hessian-vector product 近似 LOO 效果 | 非凸性假设不成立；计算成本高；深层网络估计误差大 |
| 2 | Influence Functions in Deep Learning Are Fragile (Basu et al.) | NeurIPS | 2021 | 系统实验揭示 IF 在深度网络中的脆弱性：深度越大误差越大 | 仅诊断问题，未提出修复方案 |
| 3 | Revisiting the Fragility of Influence Functions (Epifano et al.) | Neural Networks | 2023 | 重新审视 IF 脆弱性，发现 Fisher/GGN 矩阵比 Hessian 提供更稳定近似 | 仍限于单一方法内的改进 |
| 4 | If Influence Functions are the Answer, Then What is the Question? (Bae et al.) | NeurIPS | 2022 | 揭示 IF 实际近似的是 PBRF 而非 LOO；分解 IF-LOO 差异为五个独立项 | 理论贡献为主，未提供实用改进方案 |
| 5 | TRAK: Attributing Model Behavior at Scale (Park et al.) | ICML | 2023 | 基于随机投影的 After Kernel 方法，比 IF 快 100x，效果优于 TracIn | 需要多个 checkpoint；非凸设定下理论保证有限 |
| 6 | Intriguing Properties of Data Attribution on Diffusion Models (Zheng et al.) | ICLR | 2024 | D-TRAK: 理论上不合理的设计选择反而提升归因性能；挑战理论指导的方法设计 | 仅针对扩散模型验证 |
| 7 | Training Data Attribution via Approximate Unrolling — Source (Bae et al.) | NeurIPS | 2024 | 桥接隐式微分（IF）与展开方法，处理 underspecification 和多阶段训练 | 计算成本仍高于 TRAK |
| 8 | Representer Point Selection for Explaining DNNs (Yeh et al.) | NeurIPS | 2018 | 将预测分解为训练点激活的线性组合，提供激励/抑制解释 | 仅利用最后一层表示，信息损失 |
| 9 | Enhancing TDA for LLMs with Fitting Error Consideration | EMNLP | 2024 | 考虑 LLM 训练中的拟合误差对 IF 精度的影响，AUC 达 91.64% | 需要额外拟合误差估计步骤 |
| 10 | DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs (Kwon et al.) | ICLR | 2024 | 利用 LoRA 低秩结构的闭式 IF 近似，极大降低计算成本 | 仅适用于 LoRA 微调场景 |
| 11 | What is Your Data Worth to GPT? — LoGra (Choe et al.) | NeurIPS | 2024 | 低秩梯度投影实现隐式降维，Llama3-8B 上 6500x 吞吐量提升 | 投影维度选择影响精度 |
| 12 | GraSS: Scalable Data Attribution with Gradient Sparsification (Hu et al.) | NeurIPS | 2025 | 利用梯度稀疏性实现亚线性复杂度压缩，FactGraSS 165% 吞吐加速 | 稀疏度假设可能不适用所有架构 |
| 13 | AirRep: Enhancing TDA with Representational Optimization (Wei et al.) | NeurIPS | 2025 Spotlight | 可训练编码器 + 注意力池化优化表示空间归因，推理时比梯度方法快 ~100x | 需要额外训练编码器；泛化到新任务需重训 |
| 14 | Influence Functions for Scalable Data Attribution in Diffusion Models | ICLR | 2025 | 系统开发 K-FAC/GGN 近似用于扩散模型，无需方法特定超参 | 聚焦扩散模型 |
| 15 | Scalable Multi-Stage Influence Function for LLMs via EK-FAC | IJCAI | 2025 | EK-FAC 参数化实现预训练数据到下游预测的多阶段 IF 归因 | 多阶段近似累积误差 |
| 16 | Exploring TDA under Limited Access Constraints | arXiv | 2025 | 研究黑盒/有限访问场景下的 TDA 方法 | 受限场景精度下降明显 |
| 17 | Influence-based Attributions can be Manipulated | arXiv | 2024 | 证明基于影响函数的归因可被对抗性操纵 | 安全性分析为主，非方法改进 |
| 18 | Towards Unified Attribution in XAI, Data-Centric AI, and MI | arXiv | 2025 | 统一特征归因、数据归因和机制可解释性的理论框架 | 框架性工作，缺乏具体实验 |
| 19 | A Survey of Data Attribution: Methods, Applications, and Evaluation | HAL | 2025 | 生成式 AI 时代最全面的 TDA 综述，组织三个基本问题 | 综述无新方法 |
| 20 | Learning to Weight Parameters for TDA | arXiv | 2025 | 学习参数权重以改进归因质量 | 方法细节待确认 |

## 3. SOTA 方法与基准

### 当前最先进方法

**梯度类（高精度）**:
- **TRAK** (Park et al., 2023): 随机投影 + After Kernel，当前梯度类方法的标准基线。代码: [MadryLab/trak](https://github.com/MadryLab/trak)
- **Source** (Bae et al., 2024): 近似展开方法，处理 IF 在非凸/多阶段训练下的失效。适用于需要高保真归因的场景
- **EK-FAC IF** (2025): Kronecker 因子化 + 特征值校正，适用于 LLM 预训练数据归因
- **LoGra** (Choe et al., 2024): 低秩梯度投影，Llama3-8B 规模的高效 IF。吞吐量提升 6500x
- **GraSS/FactGraSS** (Hu et al., 2025): 梯度稀疏压缩，十亿参数模型上 165% 吞吐加速。代码: [TRAIS-Lab/GraSS](https://github.com/TRAIS-Lab/GraSS)

**表示类（高效率）**:
- **AirRep** (Wei et al., 2025): 可训练归因编码器，推理时 ~100x 快于梯度方法，NeurIPS 2025 Spotlight。代码: [sunnweiwei/AirRep](https://github.com/sunnweiwei/AirRep)
- **Representer Points** (Yeh et al., 2018): 最后一层激活分解，实时反馈
- **DataInf** (Kwon et al., 2024): LoRA 场景闭式 IF。代码: [ykwon0407/DataInf](https://github.com/ykwon0407/DataInf)

**重训练类（Ground Truth）**:
- **Datamodels** (Ilyas et al., 2022): 回归建模训练集子集效果，精度最高但需训练数千模型
- **Leave-One-Out (LOO)**: 标准金标准，计算成本 O(n) 次重训练
- **Data Shapley / Data Banzhaf**: 博弈论方法，公平分配但计算量极大

### 主流评估指标
- **Linear Datamodeling Score (LDS)**: 衡量归因分数预测 LOO 效果的线性相关性（TRAK 论文提出）
- **Counterfactual Evaluation**: 移除 top-k 有影响训练点后模型行为变化（最直接但最昂贵）
- **Mislabel Detection AUC**: 利用归因分数检测错误标注的能力
- **Retraining Without Top Influences**: 移除归因最高的样本后重训练，观察性能下降

### 主流数据集/场景
- **图像分类**: CIFAR-10/100, ImageNet (ResNet, ViT)
- **NLP**: SST-2, MNLI, 指令微调数据集 (RoBERTa, Llama2/3, QWEN2, Mistral)
- **生成模型**: Stable Diffusion, DDPM (CelebA, LAION 子集)
- **多模态**: CLIP (image-text pairs)

## 4. 已识别的研究空白

- **空白 1: 缺乏 per-test-point 归因稳定性量化**。现有工作评估 TDA 方法整体精度（如 LDS），但不区分哪些 test point 的归因是稳定可靠的、哪些是噪声主导的。Basu et al. (2021) 和 Epifano et al. (2023) 诊断了 IF 的脆弱性，但未提出点级别（point-level）的稳定性指标。

- **空白 2: 梯度方法与表示方法的互补性未被利用**。TRAK/IF（梯度类）和 AirRep/RepSim（表示类）在不同 test point 上表现差异显著（D-TRAK 论文暗示了这一点），但没有工作系统研究这种差异的模式，更没有自适应融合策略。

- **空白 3: 归因一致性（cross-method agreement）从未被作为信号使用**。多项工作观察到不同 TDA 方法的排序相关性不稳定，但这种不一致性仅被视为问题，而非可利用的信息源。一个关键洞察：对于某些 test point，所有方法高度一致（高稳定性），对于另一些则严重分歧（低稳定性）——这种模式本身具有信息量。

- **空白 4: 自适应/条件性 TDA 方法选择**。现有方法都是 one-size-fits-all：对所有 test point 使用同一种 TDA 方法。没有工作根据 test point 特征（如到训练分布的距离、梯度信息量、表示空间几何）动态选择或融合最合适的归因策略。

- **空白 5: TDA 鲁棒性与下游应用质量的关联**。在数据选择、数据定价、模型调试等下游任务中，归因不稳定的 test point 可能导致错误决策。没有工作量化这种传播效应，也没有利用稳定性信号来改善下游应用。

## 5. 可用资源

### 开源代码
- **TRAK**: [MadryLab/trak](https://github.com/MadryLab/trak) — PyTorch 实现，支持 CUDA 加速 JL 投影。`pip install traker[fast]`
- **AirRep**: [sunnweiwei/AirRep](https://github.com/sunnweiwei/AirRep) — 表示优化 TDA，NeurIPS 2025 Spotlight
- **GraSS**: [TRAIS-Lab/GraSS](https://github.com/TRAIS-Lab/GraSS) — 梯度稀疏压缩，NeurIPS 2025
- **D-TRAK**: [sail-sg/D-TRAK](https://github.com/sail-sg/D-TRAK) — 扩散模型归因，ICLR 2024
- **DataInf**: [ykwon0407/DataInf](https://github.com/ykwon0407/DataInf) — LoRA 场景高效 IF，ICLR 2024
- **Representer Points**: [chihkuanyeh/Representer_Point_Selection](https://github.com/chihkuanyeh/Representer_Point_Selection) — NeurIPS 2018
- **OpenDataArena**: [OpenDataArena/OpenDataArena-Tool](https://github.com/OpenDataArena/OpenDataArena-Tool) — 数据价值评估平台 (2025)
- **pyDVL**: [aai-institute/pyDVL](https://github.com/aai-institute/pyDVL) — 数据估值库，集成多种 IF 方法

### 数据集
- **CIFAR-10/100**: 标准 TDA 评估数据集，支持完整 LOO 重训练作为 ground truth
- **ImageNet**: 大规模评估，通常用子集 (ImageNet-1K 部分类)
- **SST-2 / MNLI**: NLP 情感/推理任务，文本 TDA 标准基准
- **LAION 子集**: 文生图模型归因评估
- **指令微调数据集**: Alpaca, Dolly 等用于 LLM TDA 评估

### 预训练模型
- **HuggingFace**: ResNet, ViT, RoBERTa-large, Llama-2/3, QWEN2, Mistral 系列
- **Stable Diffusion v1.5**: 扩散模型 TDA 基准
- **CLIP**: 多模态归因基准

## 6. 对 Idea 生成的启示

### 值得探索的方向

1. **TRV (TDA Robustness Value) 作为元指标**。核心 idea 有坚实基础：IF 脆弱性已被广泛确认 (Basu 2021, Epifano 2023)，PBRF 理论 (Bae 2022) 揭示 IF 实际测量对象与预期不同，D-TRAK (Zheng 2024) 证明理论指导可能失效。将这些观察统一为 per-test-point 稳定性指标是一个自然但被忽视的步骤。建议：TRV 应通过多种扰动（随机种子、超参数、方法间一致性）综合计算，而非单一扰动源。

2. **自适应融合 IF + RepSim 的可行性高**。AirRep (2025) 证明表示方法可以达到梯度方法的精度，这意味着两类方法的互补空间确实存在。关键实验设计：先在标准基准上系统比较两类方法的 per-test-point 胜率分布，验证互补性假设是否成立。如果不同 test point 确实呈现不同方法偏好，则自适应融合的价值明确。

3. **TRV 的实用下游价值**。在数据选择/清洗场景中，仅对 TRV 高（归因稳定）的样本做决策，可以显著降低误判率。这给出了 TRV 的直接应用价值，而非仅作为诊断工具。

### 已被做烂或风险较高的方向

- **单纯提升 IF 精度**：LoGra、GraSS、EK-FAC 等已充分覆盖，竞争激烈且改进空间有限
- **新的梯度投影/压缩方法**：2025 年已有 GraSS + LoGra + TRAK 三强格局，增量贡献门槛高
- **纯表示方法改进**：AirRep (NeurIPS 2025 Spotlight) 已设定很高的 bar

### 跨域借鉴潜力

- **不确定性估计 (UQ)**：贝叶斯深度学习中的预测不确定性分解 (epistemic vs. aleatoric) 可以类比到归因不确定性分解。TRV 本质上是归因空间的不确定性估计。
- **模型集成理论**：多方法融合可借鉴 ensemble learning 中的 oracle 选择、stacking 等策略，用 TRV 作为 gating signal。
- **鲁棒统计学**：M-estimator 和 breakdown point 的概念可直接映射到归因鲁棒性度量。
- **主动学习**：TRV 低（归因不稳定）的 test point 可能是模型决策边界上的困难样本，与主动学习中的不确定性采样策略相呼应。
