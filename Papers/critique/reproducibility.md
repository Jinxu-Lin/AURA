# Reproducibility Critique

## Implementation details, code availability

**Assessment: Good detail level for pilot; gaps remain.**

**Well-specified:** (1) Model training: CIFAR-10, ResNet-18, 200 epochs, SGD, cosine annealing, seed 42, test accuracy 95.50%. (2) TDA methods: EK-FAC (damping 0.01), K-FAC (damping 0.1), RepSim (penultimate layer cosine similarity), TRAK-50 (50 checkpoints, JL dim 512). (3) Evaluation: 500 test points stratified 50/class, 5K training subset for pilots. (4) BSS computation: Kronecker-factored GGN, 11.2M eigenvalues, adaptive percentile thresholds (0.2%/0.5%/99.3%), timing (70.7s for 100 points). (5) Seeds for cross-seed experiments: {42, 123, 456, 789, 1024}.

**Missing details:** (1) No learning rate, momentum, weight decay, or batch size specified for ResNet-18 training. (2) K-FAC factor estimation: how many batches, which layers, covariance moving average coefficient? (3) EK-FAC eigenvector rotation: how many samples used for the rotation step? (4) RepSim: which specific layer (pre-avgpool? post-avgpool? pre-fc?). (5) TRAK: checkpoint selection strategy (evenly spaced? last N?). (6) MRC calibration: grid search ranges for (a, b, c) parameters, Nelder-Mead convergence criteria. (7) ANOVA: software package used (statsmodels? scipy?), handling of gradient-norm as continuous vs. discretized.

**Code availability.** No mention of code release. For a paper proposing a diagnostic tool that practitioners should adopt, code availability is important. At minimum, a reference implementation of BSS computation (given K-FAC factors and test-point gradients) would substantially improve reproducibility.

**Verdict:** Sufficient detail to understand and roughly reproduce the approach, but several hyperparameters are unspecified. A code release would significantly strengthen the paper.
