# Testable Hypotheses

## Core Hypotheses (Phase 3)

### H1: TECS Signal Existence
**Statement:** TECS at the ROME editing layer l* is significantly higher than Null-A (unrelated fact's editing direction) with Cohen's d > 0.3 for N=200 facts on GPT-2-XL.

**Expected outcome:** Mean TECS ~ 0.02-0.05 (small but detectable above the ~0.001 noise floor imposed by d_k=1600 dimensional concentration).

**Falsification:** Cohen's d < 0.2 with bootstrap 95% CI crossing zero. This triggers the negative-result path.

**Measurement:** Cohen's d (TECS_real vs Null-A) with 10,000 bootstrap resamples. Bonferroni-corrected alpha = 0.01.

---

### H2: Rank-One Decomposition Validity
**Statement:** The theoretically predicted decomposition TECS ~ cos(C^{-1}k*, k_i) * cos(v* - Wk*, d_v_i) correlates with empirically measured full TECS at Spearman rho > 0.7.

**Expected outcome:** The decomposition holds approximately (rho ~ 0.6-0.8), with deviations attributable to GELU nonlinearity breaking the exact rank-one structure.

**Falsification:** Spearman rho < 0.3. This would mean the linear associative memory model is too idealized for GPT-2-XL MLPs, and the theoretical framework needs revision.

**Measurement:** Spearman rank correlation between full TECS and decomposed product, computed per-fact across 200 facts.

---

### H3: Dose-Response with Editing Success
**Statement:** TECS shows significant positive correlation with ROME editing efficacy after controlling for fact difficulty (pre-edit perplexity) and relation type.

**Expected outcome:** Partial Spearman rho ~ 0.15-0.30 after controlling for covariates. TECS adds AUROC improvement > 0.05 over covariates alone in logistic regression.

**Falsification:** Partial Spearman rho < 0.1 or AUROC improvement < 0.02 after covariate control. This would reproduce the Hase et al. (2023) finding in a new form: directional alignment, like positional localization, does not predict editing success.

**Measurement:** Partial Spearman correlation; LOOCV AUROC for logistic regression P(editing success) ~ TECS + covariates vs covariates alone; likelihood ratio test.

---

## Layer and Spectral Hypotheses (Positive Path)

### H4: Layer Specificity
**Statement:** TECS at the ROME editing layer l* is significantly higher than at l* +/- 5 (Null-B), confirming layer-specific geometric alignment.

**Expected outcome:** Cohen's d (TECS at l* vs TECS at l*+/-5) > 0.3.

**Falsification:** TECS is equally high at non-editing layers (d < 0.1). This would mean alignment is fact-specific but not layer-specific -- suggesting distributed rather than localized knowledge geometry.

**Note:** TECS >> Null-A but TECS ~ Null-B is itself an interesting finding: knowledge alignment exists but is not localized.

---

### H5: Spectral Band Selectivity
**Statement:** When TECS is decomposed by spectral bands of W (via SVD), alignment peaks in mid-range singular value bands (indices 10-200) rather than in dominant bands (top-10) or tail bands (200+).

**Expected outcome:** TECS_spectral(mid-range) > TECS_spectral(dominant) and TECS_spectral(mid-range) > TECS_spectral(tail), with the mid-range band showing Cohen's d > 0.3 vs other bands.

**Falsification:** (a) Alignment is uniform across all bands (knowledge lacks spectral structure), or (b) alignment peaks in dominant bands (knowledge is entangled with general linguistic competence, not separable).

**Connection:** Mid-range alignment would be consistent with REVIVE's finding (Zhang et al., 2026) that dominant bands encode general abilities while factual knowledge resides elsewhere.

---

## Incommensurability Hypotheses (Negative Path)

### H6: Whitening-Induced Misalignment
**Statement:** If TECS ~ 0, then TECS computed without ROME's C^{-1} whitening (TECS_unwhitened) is significantly higher than standard TECS_whitened, with Cohen's d > 0.3.

**Expected outcome:** TECS_unwhitened ~ 0.01-0.03, significantly above TECS_whitened ~ 0. This would prove that ROME deliberately rotates the editing direction away from the natural gradient geometry to achieve functional editing, at the cost of geometric consistency with TDA.

**Falsification:** TECS_unwhitened ~ TECS_whitened ~ 0. This would mean whitening is not the source of misalignment; the editing and attribution directions are fundamentally different regardless of whitening.

**Implication:** If confirmed, provides a concrete mechanistic explanation for STEAM's "isolated residual streams" at the parameter level: ROME creates isolated structures because its whitening operation pushes the edit into a subspace orthogonal to the natural training gradient flow.

---

### H7: Structured Incommensurability
**Statement:** If TECS ~ 0, the minimum principal angle between the editing subspace S_E = span(delta_W_1, ..., delta_W_N) and the attribution subspace S_A = span(g_1, ..., g_N) is significantly smaller than the expected angle for two random subspaces of the same dimension in R^{d_v * d_k}.

**Expected outcome:** theta_min(S_E, S_A) < theta_min(random, random) at p < 0.01, but theta_min >> 0 (partial but incomplete overlap).

**Falsification:** theta_min matches the random subspace baseline exactly. This would mean the misalignment is genuinely random and TECS has no geometric content -- the worst-case dead end.

**Measurement:** scipy.linalg.subspace_angles on top-r principal subspaces (r chosen by elbow criterion), compared against 100 random subspace samples.

---

## Secondary Hypotheses (Extensions)

### H8: Consolidation Correlation (from CLS theory)
**Statement:** Facts with higher estimated training frequency (BM25 retrieval count as proxy) show higher |TECS|, consistent with the Complementary Learning Systems prediction that well-consolidated knowledge shows better alignment between fast (editing) and slow (attribution) encoding systems.

**Expected outcome:** Spearman rho(|TECS|, BM25_count) > 0.15.

**Risk:** Both TECS and BM25 count may correlate with fact difficulty, creating a confound. Control by including pre-edit perplexity as covariate.

---

### H9: TECS Layer Profile vs Causal Tracing Profile
**Statement:** The TECS layer profile (TECS at all 48 layers) and the Causal Tracing indirect effect profile are positively correlated across layers (Spearman rho > 0.3 on the 48-layer profile vector).

**Expected outcome:** Moderate positive correlation, with TECS profile showing a broader peak than Causal Tracing (because TDA gradients are more distributed than activation-level causal effects).

**Falsification:** Near-zero or negative correlation. This would strengthen the Hase et al. conclusion: localization and editing geometry are fundamentally different phenomena.

---

## Summary Table

| ID | Statement (short) | Path | Primary Metric | Threshold | Priority |
|----|-------------------|------|---------------|-----------|----------|
| H1 | TECS > Null-A | Core | Cohen's d | d > 0.3 | P0 |
| H2 | Decomposition holds | Core | Spearman rho | rho > 0.7 | P0 |
| H3 | TECS predicts editing | Positive | Partial Spearman | rho > 0.2 | P1 |
| H4 | TECS is layer-specific | Positive | Cohen's d vs Null-B | d > 0.3 | P1 |
| H5 | Mid-range spectral peak | Positive | Band-wise Cohen's d | d > 0.3 | P2 |
| H6 | Whitening explains gap | Negative | d(unwhitened vs whitened) | d > 0.3 | P1 |
| H7 | Structured misalignment | Negative | theta_min < random | p < 0.01 | P1 |
| H8 | Consolidation-TECS | Secondary | Spearman rho | rho > 0.15 | P2 |
| H9 | Layer profile ~ CT | Secondary | Spearman rho | rho > 0.3 | P2 |
