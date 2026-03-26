# Soundness Critique

## BSS formula correctness, RMT applicability, MRC justification

**Assessment: Mostly sound with notable caveats.**

**BSS formula (Eq. 5-7).** The spectral decomposition in Proposition 1 is correct under the shared-eigenvector assumption. However, K-FAC and EK-FAC do *not* share eigenvectors---EK-FAC explicitly rotates K-FAC eigenvectors. The paper acknowledges this holds "within the Kronecker factored structure" but the actual computation uses Kronecker products of per-layer factors, where the shared-eigenvector assumption is approximate. The gap between the theoretical decomposition and the practical computation needs more explicit discussion.

**RMT applicability (Proposition 2).** The RMT predictions (outlier count = C-1, MP bulk, O(1/sqrt(N)) convergence) assume i.i.d. data in the overparameterized regime. CIFAR-10/ResNet-18 with batch normalization, skip connections, and data augmentation violates the i.i.d. assumption. The paper correctly identifies this as an architecture limitation but understates the gap: the RMT convergence rate guarantee may not hold for the actual GGN spectrum. The adaptive percentile thresholds (0.2%/0.5%/99.3%) are a pragmatic workaround but undermine the theoretical grounding.

**MRC justification (Proposition 3).** The optimality result is standard (inverse-variance weighting). The gap is that BSS is a *proxy* for IF variance, not the variance itself. The paper correctly parameterizes the weight function (Eq. 13) rather than plugging BSS directly into Eq. 12, but the theoretical guarantee from Proposition 3 does not transfer to the parameterized version. The "provably optimal" claim in the introduction is overstated.

**Perturbation factor uniformity.** The finding that damping dominates eigenvalues (making perturbation factors uniform) is concerning: it means BSS reduces to gradient projection energy, which is essentially a rotated gradient norm. The paper proposes corrections (partial BSS, BSS ratio) but their effectiveness is PENDING.

**Verdict:** Core formulas are correct. RMT grounding is aspirational rather than rigorous for the actual experimental setup. MRC optimality claim needs qualification.
