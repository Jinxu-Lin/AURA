# Ideation Critique — AURA

**Critic Agent** | 2026-03-17 | Updated post-debate comprehensive review

---

## Critical Issues

### 1. The Core Hypothesis Has Failed Its Own Pre-Registered Test — Rescue Angles Are Premature

The Probe pre-registered cross-seed TRV stability (rho > 0.6) as the CRITICAL pass criterion. The result (rho = -0.006) is not borderline — it is categorically indistinguishable from random. By the project's own framework, this should trigger the pre-registered "Fail" path (Section 3.5 of the problem statement: "TRV is not a test-point intrinsic property"). Instead, the project generated 9+ rescue angles across 4 perspectives without first executing the Fail path's prescribed diagnostics.

**The rescue angles are not independent.** Nearly all share a foundational assumption: that per-test-point spectral structure is meaningful and measurable. The Contrarian's H-Con1 (class + gradient norm explain >80% of TRV variance) is the null hypothesis that could invalidate the Innovator's spectral fingerprint, the Theorist's BSS, and the Pragmatist's disagreement diagnostic simultaneously.

**Four zero-cost diagnostic experiments should have been executed BEFORE generating rescue hypotheses:**
1. Class-conditional TRV variance decomposition (~30 min on existing data)
2. Softmax confidence vs. TRV correlation (~10 min, ~10 lines of code)
3. Cross-seed attribution variance vs. TRV correlation (~20 min on existing data)
4. TRV-high vs. TRV-low LDS comparison (Probe Step 8 — planned but never executed)

These four experiments, totaling ~1 hour of analysis with zero GPU cost, would determine whether the per-test-point diagnostic direction has any future. Their absence represents a major process failure: the debate stage generated hypotheses to test without first fully analyzing the data already in hand.

### 2. The Per-Test-Point Diagnostic Framing May Address a Non-Problem

The problem statement frames AURA as answering: "For this specific test point, should I trust the attribution result?" But the Probe evidence suggests:

- The answer is YES universally that Hessian choice matters (Jaccard ~ 0.45)
- The variation across test points is dominated by model-instance randomness (cross-seed rho ~ 0)
- The practical question practitioners face is "Which TDA method should I use for my task?" — a task-level decision, not a test-point-level one

The ~20% "immune" subset (TRV=5) is the one finding that DOES have per-test-point diagnostic value: these points can be reliably attributed regardless of Hessian approximation. Ironically, this finding is treated as peripheral rather than central.

### 3. The DR-to-"Adaptive Ensemble" Retreat Killed the Theoretical Backbone

The original AURA narrative derived power from the Doubly Robust estimation analogy — a principled theoretical framework from causal inference. The retreat to "diagnostic-guided adaptive ensemble" (forced by the Contrarian and Theorist challenging the estimand mismatch) collapses the theoretical contribution to:
- "Adaptive ensemble" — a well-known concept (model averaging, stacking, gating networks)
- "Use a diagnostic signal to weight methods" — oracle method selection, studied extensively

The difference between RA-TDA and "run multiple methods, ensemble with learned weights" is now unclear. The Theorist's semiparametric efficiency framework attempts to recover theoretical depth, but it requires a common estimand (distributional TDA) that is itself approximate.

---

## Major Issues

### 4. The Debate Process Generated Agreement, Not Genuine Adversarial Challenge

The pre-Sibyl 6-agent debate reached unanimous "Go with focus" consensus. No agent advocated for killing the project. The Contrarian raised H2 weakness and naive ensemble baseline challenges, but still recommended continuation.

The post-Probe Contrarian perspective is far more adversarial (advocating for a negative result paper), revealing that the initial debate was systematically biased toward continuation. The debate process should include a "hard stop" advocate from the outset.

### 5. Phase 1/Phase 2 Decoupling Creates a Contribution Trap

- Phase 1 alone (new diagnostic metric + empirical characterization) is acknowledged as "poster at best" even before the Probe failures degraded it further
- Phase 2 (adaptive fusion) cannot proceed until Phase 1 is fixed
- Together they form a coherent story, but Phase 2 depends on Phase 1 working

The project is stuck between a weak solo contribution and an unreachable combined contribution.

### 6. Intellectual Honesty Gap: Probe Results Are Being Softened

The Probe report classifies Phase 1 viability as "Conditional" when 3/7 criteria failed including the one designated CRITICAL. Under the project's own framework, the honest verdict is "NOT viable in current form, rescue required." The softened language delays the hard reckoning the project needs and creates false optimism for downstream planning.

---

## Minor Issues

### 7. The ~20% Immune Subset Deserves Elevation to Primary Finding

The most robust empirical finding — ~20% of test points maintain stable attributions across ALL Hessian approximations — is underexploited. Characterizing what makes these points immune (class centroid proximity, gradient norm, prediction confidence) could yield a practical, defensible contribution without the TRV instability baggage.

### 8. Missing Consideration of Simpler Alternatives

The "simpler alternative" challenge from the Codex reviewer remains unanswered: report uncertainty via direct intersection/union of top-k lists across multiple Hessian approximations. This requires no new theory, no proxy (SI), and is directly interpretable. AURA's value over this simpler approach depends SPECIFICALLY on the SI-TRV connection — which the Probe killed.
