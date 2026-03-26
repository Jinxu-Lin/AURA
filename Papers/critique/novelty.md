# Novelty Critique

## Is BSS genuinely new vs RIF/BIF/W-TRAK?

**Assessment: Moderate novelty with clear differentiation.**

BSS occupies a distinct niche in the TDA reliability space. RIF corrects spectral bias globally (reweighting eigenvalues for all test points uniformly). W-TRAK reweights TRAK projections by inverse eigenvalues, again as a global method improvement. BIF quantifies epistemic uncertainty about model parameters. Daunce measures sensitivity to training randomness. None of these provides a *per-test-point* diagnostic for *Hessian approximation sensitivity* specifically.

The core novelty is the insight that eigenvalue magnitudes are seed-stable while eigenvector directions are not, and that bucketing by magnitude exploits this asymmetry. This is a genuine contribution---the connection between RMT spectral stability and TDA diagnostics has not been made before.

**Concerns:** (1) The decomposition in Eq. 5-6 is relatively standard linear algebra; the novelty is in the bucketing and the RMT stability argument, not the decomposition itself. (2) MRC soft combining borrows heavily from communication theory; the adaptation to TDA is straightforward once the variance proxy (BSS) exists. (3) The ANOVA variance decomposition (C1) is methodologically standard---the novelty is in applying it to TDA, which is more of an empirical finding than a technical contribution.

**Verdict:** BSS is genuinely new as a per-point diagnostic for Hessian sensitivity. The novelty is more in the problem formulation and the RMT connection than in the mathematical machinery. This is acceptable for a venue like NeurIPS/ICML where identifying the right question can be as valuable as the answer.
