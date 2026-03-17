# Writing Critique — AURA

**Critic Agent** | 2026-03-17 | Updated post-debate comprehensive review

Note: No paper draft exists (project is in idea_debate stage). This critique covers existing written artifacts: spec, problem statement, contribution tracker, probe results, debate synthesis, perspective documents, and pre-Sibyl reviews.

---

## Major Issues

### 1. Narrative Coherence Broken — No Revised Story Exists

The original AURA narrative was clean:
1. Hessian approximation affects TDA attributions (established)
2. This effect varies per test point (TRV — novel claim)
3. TRV can be cheaply approximated via SI (practical contribution)
4. TRV guides adaptive IF+RepSim fusion (methodological contribution)

After the Probe, elements 2-4 are in question. No revised narrative has been articulated. The problem statement (v1.1), contribution tracker, and spec still reference the original narrative, creating confusion between the project's aspirational story and its actual evidential basis.

**Required action:** A 1-page "revised elevator pitch" stating:
- What the project CAN claim (Hessian choice dramatically reshapes attribution rankings; ~20% of test points are immune)
- What it CANNOT claim (per-test-point TRV is a stable diagnostic; SI predicts TRV; adaptive fusion improves over naive ensemble)
- What the open question is (does per-test-point attribution sensitivity exist as a data property?)

All subsequent work must be anchored to this honest assessment.

### 2. Overclaiming in Problem Statement

Section 1.2's gap statement claims practitioners face "the most direct problem: 'Should I use EK-FAC or K-FAC? Will the choice change the answer?'" The Probe answered: YES, universally (Jaccard ~ 0.45), not in a per-test-point-varying way.

The claim that TRV, Daunce, and BIF measure "theoretically orthogonal" dimensions is asserted without evidence. Should be weakened to "conceptually distinct" until the zero-cost cross-seed variance correlation experiment is completed.

### 3. Probe Report Mixes Facts with Optimistic Interpretation

Examples of softened language:
- "TRV distribution: PASS (but with structural issues)" — the qualification is doing enormous work
- "Phase 1 viable? Conditional" — honest verdict is "FAILED critical criterion, rescue required"
- "Cross-seed TRV instability: CRITICAL FAIL" is stated but then followed immediately by proposed fixes rather than dwelling on the implication

**Recommendation:** Separate strictly: (a) raw results against pre-registered criteria, (b) interpretation, (c) proposed modifications. Currently these are interleaved.

### 4. Contribution Tracker Needs Hard Pruning

Current tracker lists 5 contributions (C0-C4) with status labels that understate the damage:

| Contribution | Tracker Status | Honest Status |
|---|---|---|
| C0: TRV diagnostic | "needs modification" | **FAILED critical criterion; requires fundamental redefinition, not modification** |
| C1: Empirical characterization | "pending" | **Blocked by C0; cannot characterize what is not defined** |
| C2: SI-TRV bridge | "needs revision" | **DEAD; H4 falsified at rho ~ 0** |
| C3: RA-TDA fusion | "on hold" | **Blocked by C0; TRV not functional** |
| C4: Stable != correct | "initial" | **Blocked by missing TRV-LDS comparison** |

Only C0 (if rescued) and C4 (if the missing experiment is run) remain potentially viable. The tracker should mark C2 as "Abandoned" and accurately represent the blocking dependencies.

### 5. Debate Perspectives Are Excessively Long and Redundant

Combined perspective output exceeds 1,300 lines across 4 documents. Substantial overlap exists:
- BSS (Theorist) approx Spectral Fingerprint (Innovator) with different weighting
- Semiparametric Fusion (Theorist) approx Confidence-Weighted Ensemble (Pragmatist) with different theory
- Information-Geometric Decomposition (Theorist) approx Daunce/BIF orthogonality validation

A concise synthesis mapping unique contributions and shared dependencies would be more actionable than the current format.

---

## Minor Issues

### 6. Literature Review Needs Post-Probe Update

The literature review (20 references) is comprehensive for the original TRV concept. Post-Probe, several references cited in perspectives but absent from the main review should be added:
- Ensemble uncertainty: Lakshminarayanan et al. (2017) — directly relevant to Disagreement Diagnostic
- Hessian spectral structure: Ghorbani (2019), Papyan (2020) — critical for BSS/spectral angle
- Model averaging / stacking — relevant since Phase 2 retreated from DR to "adaptive ensemble"

### 7. Missing Error Bars and Confidence Intervals

Throughout all quantitative artifacts, statistical claims lack uncertainty estimates:
- Correlation coefficients without CIs
- TRV distribution percentages without CIs
- Effect sizes without CIs
- No bootstrap intervals for non-parametric quantities

For a project whose central claim involves distinguishing signal from noise, this absence is particularly problematic.

### 8. Venue Ambition Misaligned with Evidence

The spec targets NeurIPS 2026 / ICML 2026. With C2 dead, C0 failed, and C3/C4 blocked, the achievable contribution ceiling is workshop-to-poster at best for Phase 1 alone. The venue target should be explicitly reassessed after the zero-cost diagnostic experiments resolve the direction question.
