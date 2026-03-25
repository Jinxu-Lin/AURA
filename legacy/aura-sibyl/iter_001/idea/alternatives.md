# Backup Ideas for Pivot

## Alternative A: Rigorous Negative Results Paper

**Title**: "Structural Limitations of Per-Test-Point TDA Diagnostics"

**Trigger**: H-G1 fails (residual per-sample variance < 20% after class and gradient-norm control).

**Core thesis**: Per-test-point attribution sensitivity is dominated by class membership and gradient magnitude. All proposed per-sample diagnostics (TRV, BSS, spectral fingerprints, cross-method disagreement) reduce to class-conditional effects. The community should invest in better global Hessian approximations rather than per-sample routing.

**Structure**:
1. ANOVA decomposition across 3 datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet), 3 architectures (ResNet-18, ResNet-50, ViT-Small), showing class+gradient norm explain > 80% of sensitivity variance
2. Pareto frontier showing uniform methods dominate adaptive methods at every compute budget
3. Stability-correctness partial correlation analysis showing near-zero association after class control
4. Constructive recommendation: report class-conditional LDS in all TDA papers; use class-conditional method selection

**Compute**: ~30-80 GPU-hours depending on scope.

**Success probability**: 55% (Probe data trends this direction; Papyan 2020 class-structure dominance supports it).

**Publication target**: TMLR (explicitly welcomes negative results) or NeurIPS Datasets & Benchmarks track. High citation potential -- cf. Basu et al. 2020 "Influence Functions Are Fragile" (600+ citations).

**What makes this a strong contribution**: The TDA field is in an expansion phase (ASTRA, W-TRAK, Daunce, BIF, AirRep published at every top venue). A rigorous constraint on what is achievable with per-sample diagnostics would save community effort and be highly citable.

---

## Alternative B: Conformal Attribution Sets -- Distribution-Free Reliability Guarantees

**Title**: "Conformal Prediction Sets for Training Data Attribution Rankings"

**Trigger**: H-G1 passes but H-D1 fails (per-sample variation exists but spectral diagnostics are unstable). Or as a standalone theory contribution if BSS is too compute-heavy.

**Core thesis**: Replace point-estimate attribution rankings with conformal prediction sets -- for each test point, produce a set of plausible top-k lists with coverage guarantee: "with probability >= 1-alpha, the true attribution ranking is contained in this set." The set *size* is the new reliability diagnostic.

**Key innovation**: No existing work combines conformal prediction with TDA reliability. The closest is trust-score calibrated conformal sets for classification (2501.10139), but classification != attribution ranking.

**Approach**:
1. Define nonconformity score as Kendall tau distance between IF ranking and TRAK/Datamodel "ground truth" ranking
2. Calibrate split conformal threshold on 200 calibration points
3. Evaluate marginal coverage on 100 held-out points
4. Analyze conformal set size as per-test-point reliability diagnostic

**Advantage**: Sidesteps cross-seed problem entirely (calibrated per model instance). Provides distribution-free, finite-sample guarantee from standard statistical theory.

**Compute**: ~25 GPU-hours (dominated by TRAK/Datamodel ground truth computation for calibration).

**Success probability**: 40%. The nonconformity score design for rankings is non-trivial; exchangeability assumptions may be violated. Risk of uniform set sizes (no diagnostic value).

**Publication target**: ICML or NeurIPS theory-leaning paper.

---

## Alternative C: Compute-Optimal Multi-Fidelity TDA

**Title**: "How Much Hessian Does This Test Point Need? Adaptive Fidelity Selection for Efficient TDA"

**Trigger**: H-G1 passes and H-D1 passes but H-F1 fails (spectral diagnostics work but adaptive routing doesn't beat uniform strategies on the Pareto frontier). This means the diagnostic has scientific value but not practical value for fusion -- so pivot the practical value to compute savings.

**Core thesis**: Reframe the Hessian approximation hierarchy as a multi-fidelity surrogate ladder. For each test point, adaptively select the minimum-fidelity approximation that is "good enough" (attribution ranking within epsilon of the next-higher fidelity). Some test points need only K-FAC (cheap); others require full GGN (expensive). The per-test-point fidelity requirement is the new diagnostic.

**Key innovation**: Multi-fidelity optimization (Peherstorfer et al. 2018) is mature in engineering but never applied to TDA. Adaptive compute allocation per test point for TDA is entirely new.

**Approach**:
1. Compute attributions at 4 fidelity levels (Identity, K-FAC, EK-FAC, Full GGN) for 200 test points x 3 seeds
2. For each test point, determine minimum fidelity where Jaccard@10 with next tier > 0.8
3. Train logistic regression: cheap features -> fidelity level
4. Evaluate: does adaptive fidelity achieve > 90% of full-GGN LDS at < 50% compute?

**Compute**: ~8 GPU-hours.

**Success probability**: 50%. The main risk is that the effective fidelity hierarchy collapses to 2 levels (as the Probe showed for last-layer).

**Publication target**: NeurIPS poster or workshop paper.

---

## Decision Matrix for Pivot Selection

| Scenario | Primary Pivot | Secondary Pivot |
|---|---|---|
| H-G1 fails (class dominates) | **Alternative A** (negative results) | None needed |
| H-G1 passes, H-D1 fails (spectral unstable) | **Alternative B** (conformal sets) | Alternative A (partial negative + conformal) |
| H-G1 + H-D1 pass, H-F1 fails (no practical gain) | **Alternative C** (multi-fidelity efficiency) | Report BSS as diagnostic paper (C0 + C1 only) |
| All pass | Main proposal proceeds | Keep alternatives as future work |
