## [Contrarian] 反对者视角

### 逻辑闭合

The method-design.md describes BSS components (variance decomposition, BSS diagnostic, adaptive selection) for CIFAR-10/ResNet-18. The problem-statement.md has pivoted to FM1/FM2 framework on DATE-LM. These are fundamentally different research programs:
- BSS: per-test-point Hessian sensitivity diagnostic (CIFAR-10)
- FM1/FM2: structural failure mode analysis of parameter-space TDA (Pythia-1B/DATE-LM)

**The method and experiment designs are misaligned with the problem statement.** method-design.md describes BSS computation and adaptive selection, but the problem-statement asks for DATE-LM evaluation and 2x2 ablation. This creates a gap: either the method-design needs rewriting for the FM1/FM2 direction, or the problem-statement needs to revert to the BSS direction.

### 与探针一致性

The probe results are ALL from CIFAR-10/ResNet-18 where:
- FM1 is "mild" (d = 11M, not 10^9)
- FM2 is "absent" (no pre-training)
- RepSim LDS = 0.074 (much worse than IF)

These probe results CONTRADICT the FM1/FM2 hypothesis rather than supporting it. The problem-statement reinterprets them as "in low-dimensional settings without pre-training, we expect FM1 to be mild and FM2 absent" — but this is not evidence FOR the hypothesis, it's an absence of evidence AGAINST it.

### 最强反驳

**The entire paper depends on a single untested hypothesis (H4)** that has no empirical support and contradictory small-scale evidence. The method-design should include a clear fallback plan: if RepSim fails on DATE-LM, what exactly is the paper?

The most honest version of this project is: "We have a BSS diagnostic that works on CIFAR-10 (completed) + a TECS geometric analysis on GPT-2-XL (completed) + a theoretical framework about LLM-scale TDA failure (untested)." The BSS + TECS contributions are solid but modest. The FM1/FM2 contribution is potentially strong but completely unvalidated.

### Score: Revise

The method-design and experiment-design need updating to reflect the current research direction (FM1/FM2 + DATE-LM), OR the problem-statement needs to revert to the BSS direction where the designs are already aligned. Currently there is a fundamental misalignment between the formalization and the technical design.
