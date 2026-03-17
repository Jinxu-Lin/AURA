# Planning Critique — AURA

**Critic Agent** | 2026-03-17 | Updated post-debate comprehensive review

---

## Critical Issues

### 1. No Hard Kill Switch Defined

The project pre-registered falsification conditions. The Probe FAILED the critical criterion. Instead of executing the pre-registered Fail path (Section 3.5: "different seeds produce different TRV distributions -> root cause analysis was wrong"), the project entered a rescue phase generating 9+ alternatives.

**Missing:** A meta-level decision framework: "If [next experiment X] also fails [criterion Y], we [specific action: pivot/abandon/negative-result]." Without this, the project risks infinite rescue attempts with diminishing probability of success.

**Required kill switch:** "If BSS cross-seed rho < 0.3 AND Disagreement Diagnostic attribution-quality prediction rho < 0.2 AND class-conditional ANOVA explains >80% of variance, abandon per-test-point diagnostics entirely and pursue the Contrarian's negative result paper."

### 2. Four Zero-Cost Experiments Remain Undone

Before ANY new GPU-intensive rescue experiments, these analyses on EXISTING probe data should have been completed:

| Experiment | Cost | Information Value | Status |
|---|---|---|---|
| Class-conditional TRV variance decomposition | 30 min, 0 GPU | Gates all rescue angles (H-Con1 test) | NOT DONE |
| Softmax confidence vs. TRV | 10 min, 0 GPU | Trivial baseline novelty check | NOT DONE |
| Cross-seed attribution variance vs. TRV | 20 min, 0 GPU | Daunce differentiation gate | NOT DONE |
| TRV-high vs. TRV-low LDS | 30 min, 0 GPU (using Full GGN as approx. GT) | Stability-correctness validation | NOT DONE |

Total: ~90 minutes of analysis. The fact that these were not prioritized before generating 9+ rescue angles indicates a process failure: theoretical elaboration was prioritized over basic empirical characterization.

### 3. Timeline Crisis for NeurIPS 2026 / ICML 2026

Assuming NeurIPS 2026 submission deadline is approximately May-June 2026, the project has ~2-3 months. Current status: idea_debate stage with multiple failures.

Remaining pipeline: resolve idea direction -> plan -> pilot experiments -> full experiments -> analysis -> writing -> review iterations.

**Critical path analysis:**
- Zero-cost diagnostics: 1 day
- Rescue angle validation (if pursued): 1-2 weeks
- Full experiments (if rescue succeeds): 2-3 weeks
- Paper writing + revision: 2-4 weeks
- Total minimum: 6-10 weeks, assuming no further failures

This is tight but feasible IF the project converges to a single direction within 1 week. Testing multiple rescue angles serially will blow the deadline.

### 4. Resource Allocation: Theory vs. Empirics Mismatch

The project has invested heavily in theoretical framework building (Theorist's 3 angles include semiparametric efficiency, information-geometric decomposition, operator perturbation theory) while basic empirical characterization remains incomplete. The four zero-cost experiments above would provide more decision-relevant information than all three theoretical angles combined.

**Diagnosis:** The debate process rewards intellectual depth over empirical pragmatism. The Theorist's BSS framework is mathematically impressive but builds on an empirically unvalidated foundation.

---

## Major Issues

### 5. Probe Design Violated Its Own Staging Recommendations

The strategic review recommended splitting into "Pilot A (core assumptions, 1 week, fail-fast) + Pilot B (full analysis, 2-3 weeks)." The probe attempted all 9 steps simultaneously without the staging gate, resulting in:
- Step 0 (code feasibility) was skipped or inadequately executed, leading to last-layer-only computation
- Steps 7-8 (gradient norm control, TRV-LDS comparison) were planned but not completed
- The "fail-fast" intent was lost

### 6. Phase 2 Requirements Remain Completely Unspecified

Phase 2 (RA-TDA) has been deferred but not killed. Its requirements include:
- A working diagnostic (currently broken)
- IF and RepSim attributions for the same test points
- Ground truth for fusion validation (LOO or Datamodels)
- Comparison with naive ensemble AND per-method baselines
- Concrete algorithm specification (weight computation, normalization, aggregation)

The Codex reviewer specifically flagged the algorithm underspecification. If Phase 2 is to remain a conditional option, its algorithm should be specified NOW so it can be immediately executed if Phase 1 rescue succeeds.

### 7. Missing Concrete Fallback Paper Strategy

The Contrarian's "negative result paper" is mentioned but not developed into a concrete plan:
- What would the paper's title and 3-sentence abstract be?
- What are the 3-4 key claims?
- Which venue (NeurIPS D&B track, TMLR, workshop)?
- What experiments beyond the Probe are needed?

A concrete fallback plan should be developed IN PARALLEL with rescue angle validation, not after rescue fails.

---

## Minor Issues

### 8. Iteration Budget Misalignment

The Sibyl system recommends ~1 hour per experiment task. Rescue angles range from 3.5h (Pragmatist Angle 1) to 48h (Contrarian's full negative result paper). These need to be decomposed into sub-tasks fitting the iteration budget.

### 9. Full-Model Hessian Feasibility Unknown

Multiple rescue angles assume access to full-model Hessian eigendecompositions. For ResNet-18 (~11M params), this requires Lanczos iteration or Kronecker factorization. Memory requirements, convergence guarantees, and compatibility with Hong et al.'s code are not assessed.

### 10. No Priority Ordering Among Rescue Angles

The debate produced angles with self-estimated success probabilities of 40-55%. But these estimates are not calibrated, and the angles are not priority-ordered by information value per compute hour. A formal expected-value-of-information analysis would help:
- Disagreement Diagnostic: ~3.5 GPU-hours, tests whether cross-method signal exists at all
- Class-conditional ANOVA: ~0 GPU-hours, tests whether per-test-point signal exists at all
- BSS: ~5 GPU-hours, tests a specific spectral decomposition
- Spectral Fingerprint: ~4 GPU-hours, tests a different spectral decomposition

The ANOVA should come first (zero cost, highest information value), followed by the Disagreement Diagnostic (lowest GPU cost among informative experiments).
