# P5 Paper Review

**Title:** When Can You Trust Training Data Attribution? A Spectral Diagnostic Approach

---

## Dimension Scores

### 1. Novelty (weight: 0.25) -- Score: 6.5

BSS as a per-test-point spectral diagnostic for Hessian approximation sensitivity is genuinely new. The insight that eigenvalue magnitudes are seed-stable (via RMT) while eigenvector directions are not, and that this asymmetry can be exploited for TDA diagnostics, has not appeared in prior work. The ANOVA variance decomposition applied to TDA is a novel empirical finding. MRC combining is a straightforward application of known techniques once BSS exists. Compared to concurrent work: RIF/W-TRAK improve methods globally; BIF/Daunce address different uncertainty dimensions; BSS addresses a complementary per-point Hessian sensitivity dimension. The problem identification (per-point sensitivity matters) may be the most impactful contribution.

### 2. Soundness (weight: 0.20) -- Score: 6.0

Proposition 1 (spectral decomposition) is correct under the shared-eigenvector assumption, which is approximately valid for K-FAC/EK-FAC within Kronecker structure. Proposition 2 (bucket stability) relies on RMT assumptions that are violated by the actual experimental setup (batch norm, skip connections, data augmentation). Proposition 3 (MRC optimality) is standard inverse-variance weighting but the paper overstates the guarantee for the parameterized proxy version. The perturbation factor uniformity finding is a significant concern: when damping >> eigenvalues, BSS is driven by gradient projections rather than eigenvalue mismatch, potentially reducing to a rotated gradient norm. Corrections are proposed but unvalidated.

### 3. Significance (weight: 0.15) -- Score: 6.5

The paper addresses a real practitioner need: knowing whether to trust TDA results for a specific query. The ANOVA finding that 77.5% of sensitivity variance is per-test-point residual is an important community result regardless of BSS's success. If BSS proves seed-stable and MRC works, the framework could become a standard diagnostic step in TDA workflows. The negative results on TRV and SI are useful for the community. Impact is limited by CIFAR-10 scale; practitioners working with LLMs need evidence at that scale before adopting.

### 4. Experiments (weight: 0.25) -- Score: 4.5

The completed experiments (ANOVA on 500 points, BSS pilot on 100 points, disagreement analysis) are well-designed and clearly reported. Tables 1-5 present solid data. However, the paper is approximately 40% complete experimentally: the core BSS seed-stability claim (C2) and the entire MRC combining contribution (C3) are PENDING. Table 6 (main results) is empty. The Pareto frontier figure does not exist. The ablation section is entirely PENDING. Given the PENDING context, the completed portions demonstrate competent experimental methodology, and the pilot results (93.5% within-class BSS variance, 0.755 AUROC for disagreement) are encouraging. The score reflects current evidence plus reasonable confidence that PENDINGs will be filled.

### 5. Presentation (weight: 0.10) -- Score: 7.5

Strong writing throughout. The opening practitioner scenario is compelling. Contributions are crisply stated. The method section builds logically (phenomenon, diagnostic, action). Related work positioning is precise and thorough. Equations are well-motivated. Minor issues: introduction is slightly long, section numbering in the intro does not match actual structure, and the experiment section awkwardly mixes completed and PENDING results. Equation density (15 in 9 pages) is high but manageable.

### 6. Reproducibility (weight: 0.05) -- Score: 5.5

Good high-level specification (model, seeds, damping values, evaluation protocol). Missing: training hyperparameters (LR, momentum, batch size), K-FAC estimation details, specific RepSim layer, TRAK checkpoint strategy. No code release mentioned. BSS computation timing is helpfully reported (70.7s for 100 points).

---

## Weighted Composite Score

| Dimension | Weight | Score | Weighted |
|-----------|:------:|:-----:|:--------:|
| Novelty | 0.25 | 6.5 | 1.625 |
| Soundness | 0.20 | 6.0 | 1.200 |
| Significance | 0.15 | 6.5 | 0.975 |
| Experiments | 0.25 | 4.5 | 1.125 |
| Presentation | 0.10 | 7.5 | 0.750 |
| Reproducibility | 0.05 | 5.5 | 0.275 |
| **Composite** | **1.00** | | **5.95** |

---

## Decision: PASS (conditional)

**Composite score: 5.95 (threshold: 5.5)**

The paper passes the 5.5 threshold. The score is pulled down by the incomplete experimental section, but the conceptual framework is sound and the completed experiments demonstrate competent execution. The ANOVA variance decomposition and negative diagnostic results are solid contributions even without MRC.

**Condition:** The critical PENDINGs (cross-seed BSS stability, MRC main results, Pareto frontier) must be filled before submission. If cross-seed BSS stability fails (rho < 0.3), the paper's thesis collapses and the score drops below threshold.

**Key risks:**
1. BSS after gradient-norm correction may have negligible residual signal
2. MRC may not improve over naive ensemble given IF's dominance (100/100 pilot points)
3. Cross-seed stability may be lower than RMT predicts due to batch norm / skip connections

**Recommendation:** Fill critical PENDINGs, fix section numbering, soften MRC optimality claim, add training hyperparameters, commit to code release.
