# Experiment Critique

## CIFAR-10 sufficient? 100-point pilot reliable? PENDING triage?

**Assessment: Pilot results are promising but the paper is substantially incomplete.**

**CIFAR-10 sufficiency.** CIFAR-10/ResNet-18 is the standard TDA benchmark (used by TRAK, Hong et al., Grosse et al.). For establishing the phenomenon (ANOVA decomposition) and validating BSS as a diagnostic, it is adequate. However, the practical value proposition---should practitioners use BSS?---requires at least one larger-scale experiment. The paper acknowledges this limitation honestly.

**100-point pilot reliability.** The pilot uses 100 test points (10 per class) and a 5K training subset. For ANOVA (Table 1), 500 points are used, which is adequate. For BSS (Table 3), 100 points with massive variance (std >> mean for all buckets) raises concerns about statistical power. The BSS outlier bucket has mean 60.18 but std 299.39---this is a heavily right-skewed distribution where 100 points may not capture the tail reliably. The 93.5% within-class variance finding is based on these 100 points and could shift with more data.

**Critical PENDINGs (must have for a complete paper):**
1. Cross-seed BSS stability (5 seeds) -- this is the core claim; without it, BSS's advantage over TRV is theoretical only
2. MRC main results table (Table 6) -- the combining contribution (C3) has zero empirical support currently
3. Pareto frontier figure -- needed to justify the compute overhead argument

**Important PENDINGs (strongly recommended):**
4. Partial BSS and BSS_ratio results -- needed to address the gradient-norm entanglement concern
5. Full-scale 500 test points for cross-method disagreement
6. At least one ablation (damping sensitivity, since it directly affects BSS discriminability)

**Optional PENDINGs (can defer):**
7. LOO validation of TRAK-50 ground truth
8. Full ablation table (Table 7)
9. BSS--LOO correctness partial correlation

**Verdict:** The paper currently supports C1 (ANOVA) and C4 (negative results) with data, but C2 (BSS seed stability) and C3 (MRC) are PENDING. This is roughly a 40% complete experimental section.
