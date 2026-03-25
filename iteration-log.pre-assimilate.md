# Iteration Log — AURA

<!-- Reverse chronological order: newest entries at top -->

---

## 2026-03-25 — Project Consolidation (Assimilation)

**Action**: Consolidated TECA (Sibyl), TECA_old (Noesis V1), and AURA (Sibyl) into unified project under Noesis v3.

**TECA thread outcome**: DEFINITIVE NEGATIVE. TECS ~ 0 (Cohen's d = 0.05). Editing and attribution directions are geometrically incommensurable. This IS a contribution.

**AURA thread status**: Pilot experiments complete. Variance decomposition PASS (residual 77.5%). BSS approach theoretically motivated but needs 5-seed empirical verification.

**Excluded directions**:
- TECS as TDA validation metric (TECS ~ 0 means editing is NOT a validation channel)
- Scalar TRV as per-test-point diagnostic (cross-seed rho ~ 0, fundamentally seed-unstable)
- SI as TRV proxy (rho ~ 0, measures orthogonal dimension)

**Key insight**: The geometric incommensurability finding (TECA) and the spectral sensitivity analysis (AURA) are complementary contributions that can form a unified paper about understanding TDA reliability.

---

## [Assimilation] 2026-03-25 — Sibyl → Noesis v3 Transition

**Context**: Project originally conducted under the Sibyl orchestration system (AURA_old) and then migrated to a new AURA directory under Sibyl. Now assimilated into Noesis v3 framework.

**Key decisions during transition**:
1. Init Module marked as COMPLETE — all probe experiments (Phase 0, Phase 1) are done and gates passed.
2. Research Module starts at formalize — problem-statement.md and method-design.md reconstructed from existing artifacts.
3. Legacy code remains in `iter_001/exp/code/`; new v3 code will go in `Codes/`.

**What was preserved**:
- All experimental results (Phase 0, Phase 1, Phase 2a pilot, Phase 2b)
- Review history from both AURA_old and AURA rounds
- Assumption status tracking (A1 confirmed, A3 confirmed, A5 falsified)

**What changed**:
- Directory structure reorganized to Noesis v3 conventions (Docs/, research/, Codes/_Results/, Reviews/)
- project.md rebuilt from template with full review history
- problem-statement.md, method-design.md, experiment-design.md created as standalone documents

**Excluded directions**:
- SI as TRV proxy (A5 falsified, ρ ≈ 0)
- Raw TRV as diagnostic (cross-seed ρ ≈ -0.006, completely unstable)
- Full Hessian computation (infeasible within GPU budget; Kronecker-factored eigendecomposition is sufficient)

**Key risks carried forward**:
- BSS-gradient norm correlation (ρ = 0.906 in pilot) — primary risk for C1
- Cross-seed BSS stability untested at 5-seed scale — gate for C1
- Adaptive fusion (C2) may not beat naive ensemble — determines paper scope
