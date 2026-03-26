# Iteration Log

> Record of all version changes. Newest first.
> Excluded directions recorded to prevent repeating failed approaches.

---

## v2.0 -- Noesis v3 Framework Restoration (2026-03-25)

**Trigger**: External agent contamination (CRA/TECA projects overwrote AURA framework files with unrelated content). All v3 files regenerated from original AURA sources in `iter_001/`.

**Changes**:
- Regenerated all research/ files (problem-statement, method-design, experiment-design, contribution)
- Regenerated CLAUDE.md, Codes/CLAUDE.md, pipeline status files
- Created paper pipeline (outline, sections, paper.md)
- Reset research-module-status.json to implement phase with correct AURA history

**Preserved**: All original experimental data in `iter_001/exp/results/` untouched.

---

## v1.1 -- Sibyl-to-Noesis Transition (2026-03-25)

**Trigger**: Migration from Sibyl research system to Noesis v3 framework.

**Changes**:
- Created v3 directory structure (research/, Codes/, Docs/, Papers/)
- Mapped Sibyl phases to Noesis state machine
- Preserved all Sibyl artifacts in iter_001/ as read-only reference

---

## v1.0 -- Post-Probe Pivot: TRV to BSS (2026-03-17)

**Trigger**: Phase 0 probe results falsified two core assumptions.

**Key findings that drove the pivot**:
1. TRV cross-seed Spearman rho ~ 0 (TRV is model-instance property, not test-point intrinsic)
2. SI-TRV Spearman rho ~ 0 (SI is orthogonal to Hessian sensitivity)
3. J@10 degrades 1.0 -> 0.48 (core motivation confirmed: Hessian approximation matters)

**Pivot**: From scalar TRV to Bucketed Spectral Sensitivity (BSS)
- BSS uses eigenvalue-magnitude buckets (stable via RMT) instead of eigenvector directions (unstable)
- SI abandoned as TRV proxy; becomes negative result
- RA-TDA renamed to MRC soft combining

**Excluded directions**:
- [EXCLUDED] Scalar TRV as seed-stable diagnostic -- falsified (rho ~ 0)
- [EXCLUDED] SI as TRV proxy -- falsified (orthogonal dimensions)
- [EXCLUDED] Last-layer-only Hessian analysis -- hierarchy collapses to 2-3 levels

---

## v0.1 -- Initial Proposal (2026-03-16)

**Initial direction**: TRV (TDA Robustness Value) + SI proxy + RA-TDA fusion
**Status**: Proceeded to probe experiments
