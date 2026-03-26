# Pipeline Evolution Log -- AURA

> Stage-level X-reflect outputs appended here. For /praxis-evolve processing.

---

## Entry 1 — Blueprint — 2026-03-26

**执行模式**: 首次
**时间分配**: ~60% on experiment-todo.md (translating design review binding conditions into executable task list), ~25% on CLAUDE.md (codebase documentation), ~15% on configs and directory structure.

### 观察

**输入质量**: The design review synthesis (round-1) was exceptionally high quality -- 6 debaters with concrete, actionable binding conditions. The damping-dominance consensus (BSS = gradient projection, not Hessian sensitivity) was the most important finding and fundamentally shapes how experiment code should be commented and results interpreted. However, the existing `Codes/experiment-todo.md` and `Codes/CLAUDE.md` were completely stale (from the previous TECA knowledge-editing direction), meaning the blueprint had to be written from scratch rather than incrementally updated.

**执行流程**: The main challenge was reconciling three layers of codebase: (1) Sibyl legacy in `iter_001/exp/code/` with hardcoded paths, (2) Noesis v3 experiments in `Codes/experiments/` (partially migrated from Sibyl), (3) new experiments needed for Phase 2a augmented controls. The Pragmatist's recommendation NOT to migrate code from iter_001/ was critical -- it prevented a costly refactoring that would have consumed implementation time for zero scientific value.

**阶段边界**: The existing experiment scripts in `Codes/experiments/` (phase1_*, phase2a_*, phase2b_*, phase3_*) are monolithic scripts with inline data loading, computation, and analysis. The blueprint specifies a `core/` module for shared utilities, but the implement phase must decide whether to (a) extract utilities first then write new scripts, or (b) write new scripts with inline code and refactor later. Option (b) is faster and aligns with the research-code-first philosophy, but risks duplicated logic across phase2a_bss_crossseed.py and phase2a_randomized_bucket.py.

**输出质量**: The experiment-todo includes explicit decision trees and kill-switches from both formalize and design reviews. The 0-cost kill-switch checks (perturbation factor ratio, eigenvalue rank stability from existing 3-seed data) are front-loaded before any GPU commitment -- this was not in the original design but follows naturally from the Pragmatist's resource-consciousness.
