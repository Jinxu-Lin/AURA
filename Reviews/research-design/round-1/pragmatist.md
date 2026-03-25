## [Pragmatist] 务实者视角

### 工程可行性

The experiment design has two distinct tracks:

**Track A (AURA/CIFAR-10, BSS)**: Mostly complete. ~14 GPU-hours remaining. Engineering risk is moderate (Kronecker factor accessibility).

**Track B (DATE-LM/Pythia-1B, FM1/FM2)**: Not started. ~155 GPU-hours estimated. Engineering risk is HIGH:
- DATE-LM integration: 1-2 weeks of engineering before first experiment
- Pythia-1B fine-tuning: untested infrastructure
- DDA contrastive scoring on data selection task: no natural reference point
- Full 2x2 ablation needs all methods to work on same benchmark

**Compute gap**: experiment-design.md estimates 155 GPU-hours for DATE-LM. Available budget is ~14 GPU-hours remaining from AURA + whatever can be allocated on the A6000 server. This is a significant gap.

### Time to meaningful result

- DATE-LM probe (minimal): 1-2 weeks (setup + RepSim vs TRAK on toxicity)
- BSS stability test: 1 week
- Full 2x2 ablation: 3 weeks after probe
- Paper writing: 2 weeks
- **Total**: 7-8 weeks. NeurIPS deadline is ~8 weeks away.

**Zero margin for error**. Any engineering setback (DATE-LM API issues, OOM errors, TRAK scaling problems) directly risks missing the deadline.

### Most likely engineering failure

1. DATE-LM code integration takes longer than expected (very common with benchmark codebases)
2. EK-FAC IF on Pythia-1B runs OOM or produces numerical instability
3. Contrastive scoring for data selection task requires non-trivial design decisions

### Recommendation

**Execute DATE-LM probe IMMEDIATELY**. Before any further design refinement, verify that RepSim can produce reasonable LDS on at least one DATE-LM task. If this fails, redirect ALL compute to the BSS diagnostic track (Track A) which has confirmed positive signals and requires only 14 GPU-hours to complete.

**Do not attempt both tracks simultaneously** unless the probe succeeds. The compute and engineering bandwidth for both is insufficient within the NeurIPS timeline.
