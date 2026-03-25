# Phase 2a BSS Pilot Summary

## Configuration
- Mode: PILOT (100 test points, 10/class, 1 seed)
- Seed: 42 (200-epoch ResNet-18, 95.5% test acc)
- K-FAC factors estimated from 5000 training samples
- Damping: K-FAC=0.1, EK-FAC=0.01

## Eigenvalue Spectrum
- Total eigenvalues: 11,164,362 (full model K-FAC)
- Maximum EK-FAC eigenvalue: 4.98e-05
- All eigenvalues are extremely small (< 0.001)
- Original bucket thresholds (outlier>100, edge>10) yielded 0 outlier eigenvalues
- **Adapted thresholds**: outlier > 4.35e-06 (top 19), edge > 1e-06 (next 80)

## BSS Results
| Bucket | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Outlier | 60.18 | 299.39 | 8e-06 | 2244.5 |
| Edge | 4.78 | 23.42 | 5e-07 | 173.2 |
| Bulk | 1.81 | 8.92 | 4e-07 | 65.6 |
| Total | 66.78 | 331.73 | 1e-06 | 2483.4 |

## Pass Criteria
- [PASS] Outlier bucket >= 5 eigenvalues: 19 eigenvalues
- [PASS] BSS_outlier std > 0.01: std = 299.39
- [PASS] No OOM

## Key Findings

### 1. BSS is NOT a class detector (GOOD for H-D3)
Within-class variance fraction: 93.5%. Most BSS variation occurs WITHIN classes,
not between them. This means BSS captures genuine per-test-point structure beyond
class membership.

### 2. BSS strongly correlates with gradient norm (CONCERN)
- BSS_outlier vs gradient_norm: rho = 0.906
- BSS_outlier vs confidence: rho = -0.912
- BSS_outlier vs entropy: rho = 0.910

This near-perfect correlation suggests BSS may be a proxy for gradient magnitude
rather than providing independent spectral diagnostic information.

### 3. Perturbation factors are nearly uniform
The perturbation factors |1/lambda_ekfac - 1/lambda_kfac| are approximately 90
across all buckets. This is because both K-FAC and EK-FAC eigenvalues are much
smaller than the damping terms, so 1/(lambda + damping) ≈ 1/damping for both.
The BSS is therefore dominated by the squared projection (V_k^T g)^2 term,
not the perturbation factor.

### 4. Eigenvalue scale issue
The K-FAC eigenvalue products are extremely small (max ~5.8e-05). This is because
the Kronecker factorization produces A and B matrices whose eigenvalues multiply
together to give the GGN eigenvalues. For ResNet-18 with cross-entropy loss on
CIFAR-10, the B (output gradient) eigenvalues are tiny (most < 1e-05) since
softmax outputs are near 0 or 1 for well-trained models.

### 5. Extreme skewness
BSS_outlier distribution is heavily right-skewed (CV = 4.97). A few test points
(likely misclassified or low-confidence) have BSS values 2000x larger than the
median. This skewness tracks the gradient norm distribution.

## Implications for Full Experiment

1. **Bucket thresholds need recalibration**: The spec's fixed thresholds (100, 10)
   don't apply to this eigenvalue scale. Need adaptive/percentile-based thresholds.

2. **Need to disentangle BSS from gradient norm**: The strong correlation means
   we need to:
   - Compute partial BSS after regressing out gradient norm
   - Test whether BSS adds predictive power beyond gradient norm for LDS
   - Consider normalizing BSS by gradient norm squared

3. **Cross-seed stability (H-D1) remains untested**: This pilot used only 1 seed.
   Full experiment needs 5 seeds to test cross-seed BSS ranking stability.

4. **Perturbation factor uniformity**: With small eigenvalues + large damping,
   the perturbation weighting is ineffective. May need smaller damping or a
   different BSS formulation that's meaningful at this eigenvalue scale.

## GO/NO-GO: GO (conditional)
BSS produces valid, non-degenerate output with substantial within-class variation.
However, the strong gradient-norm correlation is a yellow flag that needs
investigation in the full experiment. Recommend proceeding with additional
normalization strategies.

## Timing
- K-FAC factors: 3.2s
- BSS per-point computation: 51.4s
- Total: 70.7s (1.2 min)
- Estimated full (300 points, 5 seeds): ~30 min (well within budget)
