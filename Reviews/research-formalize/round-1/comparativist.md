## [Comparativist] 文献对标者视角

### SOTA 定位

**绝对 SOTA**：
- **Parameter-space TDA for LLMs**: TrackStar (ICLR 2025) — scales to 8B parameters, 160B tokens, 80,000x gradient compression. Combines task-specific Hessian approximation + optimizer second moment correction. Currently the best-performing scalable gradient-based method.
- **Representation-space TDA**: AirRep (arXiv:2505.18513) — trainable encoder + attention pooling, matches gradient-based SOTA while being ~80x more efficient at inference. This is the strongest representation-space method as of mid-2025, and its learned representations are fundamentally different from RepSim's fixed representations.
- **Hessian-quality axis**: Hong et al. "Better Hessians Matter" (2509.23437) — establishes H >= GGN >> EK-FAC >> K-FAC hierarchy, demonstrating that Hessian quality is the single most impactful lever in parameter-space TDA.
- **Contrastive/debiased**: DDA (2410.01285) — debias + denoise, AUC 93.49% on hallucination tracing. Best contrastive approach.

**最相近 approach**：
- **Bae et al. (2409.19998)**: Diagnosed IF failure on LoRA-tuned LLMs (IF 0-7% vs RepSim 96-100%), attributed to low-rank Hessian degeneracy. This is the closest existing "diagnostic" work explaining *why* parameter-space methods fail, but their diagnosis is narrow (LoRA-specific iHVP degeneracy) and does not generalize to full fine-tuning or propose FM1/FM2 as independent failure modes.
- **Choe et al. "Concept Influence" (2602.14869)**: Quantified IF-RepSim correlation at 0.37-0.45, establishing that they measure different things. But did not explain *why* or propose a unifying framework.
- **Wang et al. "d-TDA" (2506.12965)**: Distributional framework unifying IF and kernel methods under a common estimand. Theoretical unification, but does not address representation-space methods or diagnose failure modes.

**最强简单 baseline**：
- RepSim (cosine similarity of last-layer representations) — zero Hessian computation, robust on LLM tasks (Bae et al.). Should be the floor baseline for any representation-space claim.
- TRAK with multi-checkpoint ensembling — the standard scalable parameter-space baseline.

**其他关键竞争方法**：
- Rescaled Influence Functions (RIF, arXiv:2506.06656) — addresses IF underestimation in high-dimensional/overparameterized regimes by incorporating higher-order information. Drop-in replacement for IF with negligible overhead. Directly attacks the "parameter-space methods fail in high dimensions" narrative from a different angle.
- Bayesian Influence Functions (BIF, arXiv:2509.26544) — replaces Hessian inversion with loss landscape statistics via SGMCMC, scales to billions of parameters.
- SOURCE (multi-checkpoint aggregation) — addresses instability through ensembling rather than diagnosis.

### 文献覆盖漏洞

**缺失关键工作**：

1. **Rescaled Influence Functions (RIF, 2506.06656)** — CRITICAL OMISSION. RIF directly addresses the "IF fails in high dimension" problem (what the problem-statement calls FM1: Signal Dilution) by incorporating higher-order information about parameter removal effects. If RIF substantially closes the gap between IF and representation methods, the FM1 diagnosis becomes less actionable — signal dilution would be a *solvable* problem rather than a *structural* limitation of parameter space. The problem-statement must engage with RIF as a direct counter-argument.

2. **Bayesian Influence Functions (BIF, 2509.26544)** — Replaces Hessian inversion entirely with SGMCMC sampling. Scales to billion-parameter models. Another parameter-space improvement that could undermine the "Hessian approximation is a dead end" narrative.

3. **AirRep (2505.18513)** — Listed in the problem-statement's method table but not engaged with deeply. AirRep's *learned* representations (trainable encoder) are fundamentally different from RepSim's *fixed* representations. AirRep achieves 80x efficiency over gradient methods — this is a stronger argument for representation space than the signal-processing framing. The problem-statement should position its framework relative to AirRep's empirical success and explain whether AirRep's advantage is because it addresses FM1, FM2, or something else entirely.

4. **Zhu & Cangelosi "Revisiting Data Attribution for IF" (2508.07297)** — Comprehensive review of IF data attribution capability. Should be cited for literature completeness and to ensure the problem-statement's IF characterization is up-to-date.

5. **DDA's fitting error framing (2410.01285)** — Already referenced indirectly, but the problem-statement does not engage with DDA's "fitting error" perspective: LLM training does not converge to ERM, causing IF assumptions to break. This is arguably a *third* failure mode distinct from FM1/FM2. The framework should either incorporate it explicitly or explain why it is subsumed by FM1/FM2.

6. **Scalable IF for Diffusion Models (ICLR 2025 Oral)** — Shows IF can work at scale in non-LLM domains with proper engineering. Relevant for the generality claim: is FM1/FM2 LLM-specific or general?

**覆盖充分方向**：
- The 5 representation-space methods (RepSim, RepT, In-the-Wild, Concept IF, AirRep) are well-catalogued with a clear bilinear structure table.
- The Hong et al. Hessian hierarchy is properly positioned as the key tension.
- DDA's contrastive scoring ablation results (debias removes 55.2pp, denoise removes 8.71pp) are correctly cited.
- Bae et al.'s IF failure diagnosis is correctly characterized as LoRA-specific.

### 贡献边际

**实际 delta**：

The problem-statement proposes three contributions:
1. **Unification**: Recognizing 5 representation-space methods as a coherent family sharing bilinear structure.
2. **Diagnosis**: FM1 (Signal Dilution) + FM2 (Common Influence Contamination) as independent failure modes, formalized via signal-processing lens.
3. **Evaluation**: First systematic comparison of representation-space vs parameter-space methods on DATE-LM across all three tasks.

Assessment of each:

(1) *Unification*: **Moderate novelty**. The bilinear structure phi(z_test)^T * psi(z_train) is a useful taxonomic observation, but it is somewhat obvious once stated — all similarity-based methods have this form. "Recognizing a family" is typically a survey contribution, not a research contribution. **Risk: A reviewer could dismiss this as taxonomic rather than scientific.**

(2) *Diagnosis*: **This is the core novelty claim, but with significant prior art overlap.**
- FM1 (signal dilution in R^B): The curse of dimensionality in gradient inner products is a well-known intuition. Johnson-Lindenstrauss orthogonality is standard. The observation that high-dimensional gradients have low SNR is implicit in *every* gradient compression method (LoGra, GraSS, TrackStar, TRAK's random projection). The delta is in the *formalization* as "FM1" with signal-processing vocabulary (matched filtering), not in the observation itself.
- FM2 (common influence contamination): More novel as a *named concept*. DDA already identified "knowledge bias" and proposed debias as a fix, but did not frame it as a general failure mode affecting all parameter-space methods. The delta is in the generalization: connecting DDA's debias to a structural defect in parameter-space scoring.
- **Independence of FM1 and FM2**: This is the most novel and testable claim. The 2x2 ablation {parameter/representation} x {standard/contrastive} is the key experimental design. If the interaction term is small, this validates the decomposition as scientifically meaningful.

(3) *Evaluation*: **High practical value, moderate novelty.** Putting RepSim/RepT on DATE-LM is engineering work the community needs, but "first benchmark comparison" papers are increasingly hard to publish at top venues without a methodological contribution.

**是否足够**：**Marginal for NeurIPS 2026, execution-dependent.** The paper sits at the boundary between "empirical study" and "analysis paper." The signal-processing framing elevates it above pure benchmarking, but the 2x2 ablation must produce clean, interpretable results. If the interaction term is large (>30% of main effects), the independence claim fails and the framework loses its theoretical appeal.

**Critical gap**: All AURA evidence is on **CIFAR-10/ResNet-18** — far from the LLM regime where FM1/FM2 are claimed to be dominant. The problem-statement explicitly acknowledges FM1 is "mild" and FM2 is "absent" in low-dimensional settings without pre-training. This means the current evidence *does not support* the core claims. The DATE-LM probe (Section 3.2, Pythia-1B) is existential: without LLM-scale evidence, the paper is extrapolation, not empirical science.

**创新类型**：Analysis/diagnostic paper with benchmark evaluation. Not a new method paper. Closest template: "Understanding X" or "When Does X Fail?" papers at NeurIPS.

**核心差异点**：
- vs. Bae et al.: Broader diagnosis (FM1+FM2 vs. LoRA-specific iHVP degeneracy), covers full fine-tuning, proposes independence framework
- vs. DDA: Explains *why* debias works (FM2) and connects it to a broader failure taxonomy; extends to contrastive scoring for representation methods
- vs. Hong et al.: Argues Hessian improvement is necessary but insufficient at LLM scale due to FM1/FM2
- vs. RIF/BIF: **This is the key unresolved tension.** RIF/BIF improve parameter-space methods from within. If they substantially close the gap, the "structural limitation" argument weakens. The paper must include RIF as a baseline or explain why rescaling/Bayesian approaches do not address FM1.

### 并发工作风险

**风险等级**：**中-高 (7/10)**

**依据**：

1. **Area activity**: TDA for LLMs is extremely active — my arXiv searches found 10+ directly relevant papers from 2025-2026. Multiple well-funded groups (Google DeepMind via TrackStar, Anthropic, academic labs) are investing heavily. The field produces ~50+ papers per year.

2. **Specific concurrent risks**:
   - **AirRep group (2505.18513)**: Has the infrastructure and representation-space perspective. A natural follow-up is a systematic comparison paper with gradient baselines — directly overlapping with contribution (3).
   - **Hong et al. follow-up**: The "Better Hessians Matter" group has the most complete Hessian comparison infrastructure. A natural extension to representation-space baselines would directly overlap with the DATE-LM evaluation.
   - **DDA follow-up**: The DDA authors have already identified "knowledge bias." Formalizing this as a general framework directly overlaps with FM2.
   - **RIF/BIF line**: If these methods substantially close the IF-RepSim gap at LLM scale, the "parameter space is fundamentally broken" narrative weakens.

3. **Scooping probability**: The DATE-LM evaluation gap (no representation-space methods evaluated) is visible to everyone in the community. The probability that someone runs RepSim/RepT on DATE-LM within 2 months is non-trivial. The FM1/FM2 diagnostic framework is harder to scoop because it requires the specific signal-processing framing, but "why representation methods work better" is an obvious question.

4. **Mitigating factors**: The specific combination of (a) signal-processing FM1/FM2 framing, (b) independence test via 2x2 ablation, and (c) AURA's per-point complementarity evidence (tau = -0.467, AUROC = 0.691) is unique. No single competing group has all three elements. But each individual element has low barrier to entry.

5. **Timeline pressure**: NeurIPS 2026 submission ~May 2026 is ~2 months away. The DATE-LM probe (2.5 GPU-days on Pythia-1B) must start immediately. Any delay compounds scooping risk.

### Top Recommendations

1. **Engage with RIF (2506.06656) explicitly** — it is the strongest counter-argument to FM1 being "structural." Include as baseline or explain why rescaling does not address signal dilution at LLM scale.
2. **Prioritize DATE-LM probe execution** — all current evidence is CIFAR-10/ResNet-18 where FM1 is mild and FM2 is absent. The paper's entire thesis depends on LLM-scale evidence. This is existential.
3. **Frame the 2x2 ablation as THE core contribution** — the unification and evaluation are necessary but not sufficient. The FM1/FM2 independence test is the scientific novelty.
4. **Address DDA's fitting error as potential FM3** — either incorporate it into the framework or explain why non-convergence to ERM is subsumed by FM1/FM2.
5. **Sharpen the AirRep positioning** — AirRep's learned representations likely bypass both FM1 and FM2. If so, it validates the framework; if not, it challenges it. Either way, it must be analyzed.
