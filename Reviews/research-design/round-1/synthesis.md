## 多维辩论综合（Design Review Synthesis）

### 分歧地图

**共识点（>=4/6 视角）**：
1. **DATE-LM probe is existential**: All reviewers agree that H4 (repr-space competitive on DATE-LM) must be validated before committing to the FM1/FM2 direction. Without LLM-scale evidence, the paper is extrapolation.
2. **Method-design/experiment-design are misaligned with problem-statement**: The BSS-focused designs don't match the FM1/FM2-focused problem statement. This needs resolution.
3. **AURA CIFAR-10 results are confirmed but at wrong scale**: The variance decomposition and disagreement analysis are solid but do not directly support FM1/FM2 claims at LLM scale.
4. **Compute budget is tight**: 155 GPU-hours for DATE-LM with ~14 remaining from AURA budget + A6000 access uncertain.

**主要分歧**：
- **Contrarian (Revise) vs others (Pass with conditions)**: Contrarian argues the misalignment between BSS-focused designs and FM1/FM2-focused problem statement is a fundamental issue requiring revision. Others acknowledge the misalignment but consider it resolvable through design updates.
- **Skeptic vs Theorist on FM1/FM2 independence**: Skeptic argues tau = -0.467 can be explained by geometric differences without FM1/FM2. Theorist accepts the RMT framework but notes theoretical gaps.

**独特洞察**：
- **[Interdisciplinary]** (from formalize review): MRC soft weighting may be more effective than hard routing for adaptive selection.
- **[Methodologist]**: Missing baselines (SI, confidence, oracle routing) should be added.
- **[Contrarian]**: The project should decide between BSS direction (aligned designs, smaller scope) and FM1/FM2 direction (misaligned designs, larger scope).

---

### 优先级排序

**必须处理**：
1. **Resolve BSS vs FM1/FM2 direction** — Contrarian. The method-design and experiment-design must be updated to match whichever direction is chosen. Current misalignment is a blocking issue.
2. **Execute DATE-LM probe** — All. If choosing FM1/FM2 direction, this is the gating experiment with zero current evidence.
3. **Update method-design for chosen direction** — All. Current method-design describes BSS components; if FM1/FM2 is the direction, it needs 2x2 ablation design, representation method evaluation protocol, etc.

**可选处理**：
- Add RIF as baseline (Comparativist)
- Add Daunce comparison (Comparativist)
- Layer selection ablation for BSS (Methodologist)
- ICC for cross-seed stability (Empiricist)

**暂时搁置**：
- Stieltjes transform analytics
- Bode sensitivity integral
- Full 5-method representation family evaluation (can start with RepSim + RepT)

---

### 判定

**Revise**

The design review identifies a fundamental misalignment: the problem-statement has pivoted to FM1/FM2 + DATE-LM, but the method-design and experiment-design still describe the BSS diagnostic approach. This is not a minor gap — it means the technical designs do not address the formalized research questions.

**Two viable paths**:

**Path A (FM1/FM2 + DATE-LM)**: Higher risk, higher reward. Requires rewriting method-design and experiment-design for the 2x2 ablation framework. Needs DATE-LM probe (existential). Needs ~155 GPU-hours. Timeline is very tight. If probe succeeds, this is a NeurIPS-competitive paper.

**Path B (BSS diagnostic + TECA negative result)**: Lower risk, lower reward. method-design and experiment-design are already aligned. Needs ~14 GPU-hours. BSS cross-seed stability is the gating experiment. If BSS validates, this is a solid but more modest contribution. TECA geometric incommensurability adds value at zero cost.

**Recommendation**: Execute DATE-LM probe (3-5 days) IMMEDIATELY to determine which path. If probe passes, update designs for Path A. If probe fails, execute Path B with existing aligned designs.

---

### 修订后的研究方向

The direction depends on the DATE-LM probe outcome. Designs need updating for whichever path is chosen.

---

### 下一阶段重点

1. **Execute DATE-LM probe** (RepSim vs TRAK on Pythia-1B toxicity filtering) — 3-5 days, ~6 GPU-hours
2. **Based on probe outcome, update method-design and experiment-design** for the chosen direction
3. **If FM1/FM2 path: add RIF as baseline** and address DDA fitting error as potential FM3

---

### 未解决的开放问题

- **BSS vs FM1/FM2 direction**: Not yet resolved. DATE-LM probe result will determine.
- **Compute budget**: 155 GPU-hours needed for FM1/FM2, ~14 remaining from AURA budget. A6000 availability uncertain.
- **Timeline**: 8 weeks to NeurIPS with zero margin. Any engineering delay risks deadline.
- **FM1/FM2 independence**: tau = -0.467 on CIFAR-10 may not generalize to LLM scale.

---

### Outcome

**outcome: revise**
**notes**: Design review requests revision. Method-design and experiment-design are misaligned with the FM1/FM2-focused problem-statement. Immediate action: execute DATE-LM probe to determine direction (FM1/FM2 vs BSS), then update designs accordingly. Return to design phase after probe execution and design alignment.
