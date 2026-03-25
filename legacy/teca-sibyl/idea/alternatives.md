# Backup Ideas for Pivot

## Alternative 1: Multi-Operation Knowledge Fingerprinting

**Source:** Innovator Angle 3, enhanced with Contrarian Direction 3.

**Core idea:** Instead of pairwise TECS between editing and attribution, compute a full "knowledge fingerprint" for each fact -- a set of direction vectors from multiple independent operations (ROME, MEMIT, TDA top-1, TDA aggregated, Knowledge Neurons). Perform clustering analysis on these fingerprints to discover knowledge archetypes (Type-A: all aligned; Type-B: editing-only alignment; Type-C: attribution-only; Type-D: no alignment).

**When to pivot:** If the core TECS proposal produces ambiguous results (d ~ 0.1-0.2) that are too weak for either a positive or negative story. Fingerprinting reframes the question from "are they aligned?" to "what is the typology of alignment/misalignment?" -- a richer analysis that can absorb ambiguous TECS results.

**Key hypotheses:**
- Knowledge archetypes correlate with linguistic properties (simple SRO triples = Type-A, complex knowledge = Type-D)
- Archetype distribution shifts across layers, with mid-layers showing more Type-A facts
- Sequential editing collapse (Zhang et al., 2026) preferentially affects Type-A facts

**Computational cost:** ~75 min on 1x RTX 4090 (200 facts x 5 directions x 3 layers).

**Risk:** Clustering in high-dimensional space may not yield clean archetypes. Mitigate by using continuous similarity distributions if discrete types are not supported.

**Success probability:** 50%.

---

## Alternative 2: TECS as a Confound Detector for TDA

**Source:** Contrarian Direction 2, with Empiricist's dose-response methodology.

**Core idea:** Flip the TECS narrative. Instead of "TECS validates TDA," propose "TECS detects TDA confounds." If TECS is high for a fact, it means the TDA gradient aligns with ROME's known artifact-driven editing direction -- a red flag for TDA reliability, not a validation. High TECS identifies facts where TDA is confounded by surface-level subject co-occurrence (the same shortcuts ROME exploits per Liu et al., 2025, arXiv 2506.04042).

**When to pivot:** If TECS is moderately positive (d ~ 0.3-0.5) but the dose-response analysis shows TECS does NOT predict editing success. This pattern -- TECS is nonzero but functionally meaningless -- is best explained by shared artifacts rather than shared knowledge geometry.

**Key hypotheses:**
- TECS correlates with BM25 retrieval score (surface overlap proxy), not with genuine causal influence
- High-TECS facts show inflated TDA scores relative to leave-one-out ground truth
- TECS-as-confound-detector has higher AUROC than TECS-as-quality-predictor

**Computational cost:** ~60 min on 1x RTX 4090 (reuses core TECS computation + selective LOO retraining on 10 facts).

**Risk:** Leave-one-out retraining on GPT-2-XL for 10 facts is expensive and may not provide sufficient statistical power. Mitigate by using counterfactual data augmentation instead of actual retraining.

**Success probability:** 45%.

---

## Alternative 3: Cross-Layer Directional Consistency as a Knowledge Structure Probe

**Source:** Contrarian Direction 3 ("Knowledge Has No Direction"), with Pragmatist's Layer-Sweep methodology.

**Core idea:** Before comparing editing and attribution directions, test a more fundamental question: does the "knowledge direction" (whether from editing or attribution) show any consistency across layers? If the direction of a fact's parameter footprint changes dramatically from layer to layer, then single-layer TECS is measuring an artifact of layer choice, not a property of knowledge.

**When to pivot:** If Phase 2 (TDA gradient validation) fails -- gradients lack fact-specificity. This means TECS cannot work at all, but we can still ask whether *editing* directions show cross-layer consistency (requiring only ROME, not TDA).

**Key hypotheses:**
- Cross-layer consistency of ROME delta_W (cosine between delta_W at adjacent layers) is near zero, indicating editing directions are layer-specific artifacts
- Cross-layer consistency of TDA gradients is also near zero, indicating attribution directions are equally layer-specific
- If both directions are layer-specific but TECS at the "right" layer is nonzero, it means alignment requires matching the layer, not that knowledge has a global direction

**Computational cost:** ~80 min on 1x RTX 4090 (100 facts x 5 layers x 2 methods + 4 retrieval-set variants).

**Risk:** If knowledge directions DO show high cross-layer consistency, this alternative provides no novel insight. But this would actually strengthen the main proposal by validating the premise that knowledge has a stable direction.

**Success probability:** 55%.

---

## Decision Matrix

| Trigger Condition | Pivot To | Rationale |
|-------------------|----------|-----------|
| TECS d ~ 0.1-0.2 (ambiguous) | Alternative 1 (Fingerprinting) | Richer typology absorbs ambiguous pairwise signal |
| TECS d > 0.2 but no dose-response | Alternative 2 (Confound Detector) | Nonzero but functionally meaningless TECS suggests shared artifacts |
| Phase 2 fails (gradients not fact-specific) | Alternative 3 (Cross-Layer Consistency) | Can still study editing directions without TDA |
| All TECS variants d < 0.05 AND random misalignment | Full pivot: "The Geometry of Knowledge Operations Is Incommensurable" negative-result paper with extended subspace analysis | No TECS-based story works; reframe as comprehensive negative characterization |
