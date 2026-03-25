# Interdisciplinary Perspective: TECA Research Proposals

**Agent**: sibyl-interdisciplinary
**Date**: 2026-03-17
**Topic**: Using model editing (ROME/MEMIT) parameter update directions as independent validation signals for TDA attribution directions; proposing the TECS (TDA-Editing Consistency Score) metric to probe knowledge geometry in parameter space.

---

## Executive Summary

The TECS proposal -- comparing editing directions with attribution gradients in parameter space -- sits at an intersection that maps remarkably well onto deep structures from three fields outside ML: **statistical physics of disordered systems** (spin glasses and energy landscapes), **neuroscience of memory** (engrams and complementary learning systems), and **immunology** (clonal selection and somatic hypermutation). Each analogy is not merely metaphorical but identifies a precise structural correspondence that generates testable predictions for the TECA project. We develop three interdisciplinary research angles, each grounded in existing cross-disciplinary literature.

**Central interdisciplinary claim**: Knowledge facts stored in transformer MLPs are analogous to metastable states in a spin-glass energy landscape. ROME editing corresponds to an external field that tilts the landscape to create a new minimum, while TDA gradients trace the natural gradient flow that originally carved that minimum during training. TECS measures whether these two forces point in the same direction in parameter space -- a question that has direct analogs in spin-glass overlap functions, engram reactivation geometry, and antibody-antigen affinity maturation.

---

## Angle 1: Spin-Glass Energy Landscapes and Knowledge Attractors

### Source Field: Statistical Physics of Disordered Systems

### Core Analogy

In spin-glass theory, a system's state is described by a configuration of spins {s_i}, and the energy landscape E(s) has an exponentially large number of metastable minima separated by barriers. The **overlap function** q = (1/N) sum_i s_i^a s_i^b between two replicas (independent samples from the Gibbs distribution) is the central order parameter that characterizes the geometry of the landscape. Parisi's replica symmetry breaking (RSB) solution reveals that the overlap distribution P(q) has a hierarchical, ultrametric structure -- minima are not randomly scattered but organized into nested clusters.

**Structural correspondence to TECA**:

| Spin Glass Concept | TECA Analog |
|---|---|
| Spin configuration s | Vectorized MLP weight matrix vec(W) |
| Energy landscape E(s) | Loss landscape L(theta) restricted to the MLP weight subspace |
| Metastable minimum | Trained model's weight configuration encoding a specific fact |
| External field h_i | ROME's rank-one edit delta_W_E (an imposed perturbation to create a new minimum) |
| Gradient of E at a minimum | TDA gradient g_TDA (the direction training carved to reach this minimum) |
| Overlap q^{ab} = (1/N) s^a . s^b | **TECS = cos(delta_W_E, g_TDA)** -- overlap between the editing and attribution "replicas" |

The correspondence is deep: both the overlap function q and TECS measure the angular relationship between two independent probes of the same underlying structure. In spin glasses, high overlap between replicas indicates they are in the same "pure state" (valley); in TECA, high TECS indicates that editing and attribution access the same knowledge substructure.

### Grounding in Existing Cross-Disciplinary Work

1. **Barney, Winer & Galitski (2024, arXiv 2408.06421)** -- "Neural Networks as Spin Models": Maps neurons to Ising spins and weights to spin-spin couplings. They prove that an untrained network with random weights corresponds to a Sherrington-Kirkpatrick spin glass with RSB. Training progressively destroys the spin-glass phase and creates a phase with "hidden order" whose melting temperature T_c grows as a power law in training time. **Implication for TECA**: The hidden order phase after training is precisely where knowledge is encoded. ROME editing perturbs this ordered phase; TECS measures whether the perturbation aligns with the order parameter's gradient direction.

2. **Liao et al. (2024, arXiv 2407.20724)** -- "Exploring Loss Landscapes through the Lens of Spin Glass Theory": Uses random walks in parameter space, permutation-interpolation protocols, and hierarchical clustering to demonstrate RSB-like structure in DNN loss landscapes. They show that trained solutions form a hierarchy reminiscent of Parisi's ultrametric tree. **Implication for TECA**: If knowledge facts form an ultrametric hierarchy in parameter space, TECS should exhibit systematic structure -- facts in the same "cluster" (semantically related) should have correlated TECS values.

3. **Li (2025, arXiv 2508.07397)** -- "A Spin Glass Characterization of Neural Networks": Constructs a Hopfield-type spin glass from feedforward networks and uses replica overlaps as descriptors. Finds that the spin-glass description captures properties (capacity, robustness) not visible through conventional metrics. **Implication for TECA**: TECS can be reinterpreted as a special case of the replica overlap -- specifically, the overlap between the "editing replica" and the "attribution replica" of the same factual knowledge state.

4. **Koulischer et al. (2023, arXiv 2311.18434)** -- "Phase Transition in Modern Hopfield Networks": Demonstrates a critical temperature beta_c controlling the transition from a single global attractor to pattern-specific minima. The effective temperature beta_eff depends on both the hyperparameter and the distribution of stored patterns. **Implication for TECA**: The editing layer l* identified by causal tracing may correspond to the layer where beta_eff crosses the critical threshold -- the layer where pattern-specific attractors (individual facts) become well-separated, enabling both targeted editing and meaningful attribution.

### Concrete Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Compute the "replica overlap matrix" Q_{ij} = cos(delta_W_E^{(i)}, g_TDA^{(j)}) for 100 facts (a 100x100 matrix) | GPT-2-XL | 0 min (reuse cached data) |
| 2 | Apply hierarchical clustering to Q with Ward linkage; test for ultrametric structure using Rammal's cophenetic correlation | - | 5 min |
| 3 | Compare the dendrogram with semantic clustering of the 100 CounterFact facts (by relation type); test Adjusted Rand Index | - | 5 min |
| 4 | Compute the "overlap distribution" P(TECS) across all 100x100 pairs; compare shape with RSB predictions (bimodal vs unimodal) | - | 5 min |
| 5 | Repeat at non-editing layers l* +/- 5: the "spin glass" structure should be weaker (more uniform overlap) | GPT-2-XL | 20 min |

**Total**: ~35 min. **Success probability**: 45%.

### Testable Predictions

**P1** (Ultrametric structure): The TECS overlap matrix should exhibit hierarchical clustering that correlates with semantic fact categories (Adjusted Rand Index > 0.15). This is a direct prediction from RSB theory -- if knowledge is organized hierarchically in parameter space, the editing-attribution overlap should reflect that hierarchy.

**P2** (Layer-specific phase transition): At the editing layer l*, the TECS distribution P(TECS) should be broader (higher variance) than at non-editing layers, analogous to how the overlap distribution broadens below the spin-glass transition temperature. At non-editing layers, P(TECS) should concentrate near zero (paramagnet-like).

**P3** (Barrier height predicts editing difficulty): Facts with low TECS (editing and attribution directions misaligned) should correspond to facts where the energy barrier between the pre-edit and post-edit minima is high -- operationally, these facts should show higher editing loss and worse generalization scores.

### Why This is Novel

No prior work has interpreted TECS through the lens of spin-glass overlap functions. The theoretical perspective from Barney et al. (2024) provides the mapping (weights as couplings), and the empirical tools from Liao et al. (2024) provide the analysis protocols. Connecting these to the editing-attribution comparison is a genuinely new contribution that could transform TECS from a simple cosine metric into a probe of the parameter space's thermodynamic structure.

### Failure Modes

- **No ultrametric structure detected**: The 100x100 overlap matrix may appear essentially random, with cophenetic correlation < 0.2. This would suggest that knowledge encoding is not hierarchically organized at the scale of individual MLP layers, or that TECS is too noisy to detect it.
- **Spin-glass analogy breaks at the nonlinear boundary**: The spin-glass mapping assumes linear neurons (Barney et al. use Ising and binarized activations). GELU nonlinearity in GPT-2 may destroy the clean correspondence. Mitigation: analyze the skip-connection (linear) component separately from the GELU (nonlinear) component.

---

## Angle 2: Engrams, Complementary Learning Systems, and the Dual Geometry of Knowledge

### Source Field: Neuroscience of Memory

### Core Analogy

In neuroscience, an **engram** is the physical trace of a memory -- a sparse set of neurons whose synaptic connections were modified during learning and whose reactivation is necessary and sufficient for memory recall (Josselyn & Tonegawa, 2020, Nature Reviews Neuroscience). Recent work has revealed that engrams are **dynamic**: their neural composition changes during consolidation, with neurons being added to and removed from the engram (Nature Neuroscience, 2023), and computational models predict that memory selectivity emerges through inhibitory synaptic plasticity.

The **Complementary Learning Systems (CLS) theory** (McClelland, McNaughton & O'Reilly, 1995) posits that the brain requires two systems: the hippocampus for rapid, sparse encoding of individual episodes, and the neocortex for slow, distributed extraction of statistical regularities. Critically, these two systems encode the *same information* but in *geometrically different* ways -- sparse and pattern-separated vs dense and overlapping.

**Structural correspondence to TECA**:

| Neuroscience Concept | TECA Analog |
|---|---|
| Engram (sparse memory trace) | ROME's rank-one edit delta_W_E (a targeted, sparse modification encoding a specific fact) |
| Neocortical representation (distributed) | TDA gradient g_TDA (the distributed gradient signature accumulated over many training examples) |
| Engram reactivation pattern | The editing direction projected onto the attribution subspace: proj_{S_A}(delta_W_E) |
| CLS dual encoding | ROME (hippocampal-like: fast, targeted) vs TDA (neocortical-like: slow, distributed) |
| Memory consolidation | The process by which training data gradually shapes W into its final configuration |
| Engram selectivity (post-consolidation) | TECS increasing for well-consolidated facts vs remaining low for poorly consolidated ones |

The key insight is that **ROME editing operates like hippocampal encoding** (rapid, sparse, pattern-separated), while **TDA gradients reflect neocortical encoding** (gradual, distributed, overlapping). TECS measures the geometric alignment between these two fundamentally different encoding strategies applied to the same factual knowledge.

### Grounding in Existing Cross-Disciplinary Work

1. **Fontaine & Alexandre (2025, arXiv 2509.01987)** -- "Semantic and episodic memories in a predictive coding model of the neocortex": Demonstrates that a predictive coding model can recall individual examples (episodic-like memory) but only when trained on few examples. With many examples, the model transitions to semantic (distributed) memory. **Implication for TECA**: Well-trained LLMs should have neocortical-like (distributed) knowledge encoding. ROME imposes hippocampal-like (sparse) edits. High TECS would indicate that the distributed encoding happens to align with the sparse intervention direction -- like an engram reactivation that perfectly overlaps with the consolidated neocortical trace.

2. **Lee et al. (2024, arXiv 2406.02596)** -- "Hare and Tortoise Networks": Directly implements CLS theory in deep learning with a fast-adapting "Hare" (hippocampal analog) and slow-consolidating "Tortoise" (neocortical analog). **Implication for TECA**: The Hare's rapid adaptation is structurally analogous to ROME editing (fast parameter modification), while the Tortoise's gradual learning parallels TDA gradients (accumulated training signal). Their periodic resynchronization is analogous to asking "do these two systems agree on the knowledge geometry?" -- precisely what TECS measures.

3. **Szelogowski (2025, arXiv 2506.01659)** -- "Engram Memory Encoding and Retrieval: A Neurocomputational Perspective": Synthesizes how sparsity promotes efficient, interference-resistant memory representations. Engram neurons undergo lasting physical and biochemical changes. **Implication for TECA**: The rank-one structure of ROME edits (extreme sparsity in the matrix rank sense) mirrors the sparsity of engram populations. The TDA gradient, being a sum over many training examples, is the dense neocortical counterpart.

4. **Szelogowski (2025, arXiv 2507.21474)** -- "Hebbian Memory-Augmented Recurrent Networks: Engram Neural Networks": Introduces explicit, differentiable memory matrices with Hebbian plasticity. Memory formation follows Hebbian traces -- strengthening connections between co-active neurons. **Implication for TECA**: The MLP weight matrix W can be viewed as a Hebbian memory matrix where W = sum_i v_i k_i^T (as in the associative memory model underlying ROME). The TDA gradient for training example z_i is proportional to the Hebbian update that example would contribute. TECS thus measures how well the ROME edit aligns with the specific Hebbian contribution of the attributed training data.

5. **Dynamic engram composition** (Nature Neuroscience, 2023 -- Memory consolidation study): Engram composition in the dentate gyrus changes within hours, with neurons systematically added and removed. Inhibitory plasticity is necessary for selectivity. **Implication for TECA**: After training, some facts may be well-consolidated (selective engram, high TECS) while others are still in an intermediate state (non-selective engram, low TECS). This predicts a correlation between TECS and fact frequency in the training corpus.

### Concrete Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Estimate training data frequency for each CounterFact fact using BM25 retrieval counts as proxy | - | 5 min (cached) |
| 2 | Compute TECS for each fact; correlate with estimated frequency (Spearman rho) | GPT-2-XL | 0 min (cached) |
| 3 | Partition facts into "well-consolidated" (high BM25 count, high edit success) and "poorly-consolidated" (low count, low edit success); compare TECS distributions | - | 5 min |
| 4 | Compute the "engram sparsity" of each editing direction: effective rank of delta_W_E (nuclear norm / operator norm); correlate with TECS | - | 5 min |
| 5 | Compute the "neocortical distributedness" of each attribution gradient: effective rank of sum_i g_TDA^{(i)}; test whether more distributed gradients predict lower TECS | - | 5 min |

**Total**: ~20 min. **Success probability**: 50%.

### Testable Predictions

**P4** (Consolidation-TECS correlation): Facts with higher training frequency (better consolidated) will show higher |TECS|, because their knowledge trace has had more time to converge to a stable attractor aligned with the natural gradient geometry.

**P5** (CLS encoding asymmetry): The editing direction (hippocampal-like) will have lower effective rank (more sparse) than the attribution direction (neocortical-like) by a factor of at least 3x. This asymmetry is the parameter-space signature of complementary learning systems.

**P6** (Engram reactivation success predicts edit locality): Facts where TECS is high (editing aligns with natural attribution) should show better editing locality preservation, because the edit "reactivates" the existing memory trace rather than creating a new, incompatible one -- directly paralleling how engram reactivation leads to faithful memory recall only when the reactivation pattern matches the original encoding pattern.

### Why This is Novel

The CLS theory has been applied to continual learning and catastrophic forgetting in neural networks (Lee et al., 2024; Sorrenti et al., 2024, arXiv 2401.08623), but **never to the comparison between knowledge editing and training data attribution**. The specific insight that ROME is "hippocampal" (fast, sparse) while TDA is "neocortical" (slow, distributed) creates a framework where TECS becomes a measure of **memory system alignment** rather than a simple cosine similarity. This reframing has independent scientific value: it connects the knowledge editing literature to the rich theoretical apparatus of CLS, potentially explaining why some edits succeed (aligned with the consolidated trace) and others fail (misaligned, creating an isolated engram -- exactly what STEAM (arXiv 2510.10398) observed as "isolated residual streams").

### Failure Modes

- **Frequency proxy is too noisy**: BM25 retrieval count from a partial corpus may not reflect actual training data frequency for GPT-2. Mitigation: use model perplexity on the factual statement as an alternative consolidation proxy (lower perplexity = better consolidated).
- **Effective rank of rank-one matrices is trivially 1**: By construction, ROME delta_W_E is rank-one. The interesting question is whether the aggregated attribution gradient sum_i g_TDA^{(i)} has significantly higher effective rank. If it is also approximately rank-one (all training examples contribute in the same direction), the CLS asymmetry prediction fails.

---

## Angle 3: Immunological Clonal Selection and the Affinity Geometry of Knowledge Updates

### Source Field: Immunology -- Adaptive Immune Response

### Core Analogy

The adaptive immune system discovers and refines antibodies through a remarkable optimization process. When a pathogen enters the body, B cells with antibodies that weakly match the antigen undergo **clonal selection** -- they proliferate and enter germinal centers where **somatic hypermutation** introduces random point mutations into the antibody gene. Mutant B cells then compete for binding to the antigen: those with higher affinity survive (affinity maturation), while those with lower affinity undergo apoptosis. This process converges to high-affinity antibodies through directed evolution in sequence space.

The structural parallel to TECA is precise:

| Immunology Concept | TECA Analog |
|---|---|
| Antigen (foreign entity to be recognized) | Target fact to be stored/edited (k*, v*) |
| Antibody (recognition molecule) | Weight configuration W encoding that fact |
| Antibody-antigen binding affinity | Negative cross-entropy loss for the target fact |
| Somatic hypermutation (random perturbation) | Training updates (SGD steps) that gradually shape W |
| Affinity maturation direction | TDA gradient direction g_TDA (direction of improving affinity) |
| Passive immunization (injecting pre-made antibody) | ROME editing delta_W_E (directly injecting a solution) |
| Cross-reactivity (antibody binds similar antigens) | Editing generalization (edit transfers to paraphrases) |
| Autoimmunity (antibody attacks self) | Editing side effects (edit damages unrelated knowledge) |

**The key immunological insight for TECA**: In immunology, there is a well-studied distinction between **actively acquired immunity** (the body's own affinity maturation through somatic hypermutation -- analogous to learning from training data / TDA) and **passively acquired immunity** (injecting pre-formed antibodies -- analogous to ROME editing). A fundamental finding in immunology is that **passively acquired antibodies operate through different mechanisms than actively matured ones**, even when they target the same antigen. Passively transferred antibodies often have different epitope specificity (they bind different parts of the antigen) than the host's own matured antibodies.

**TECS, in this framework, measures the "epitope alignment" between passive (editing) and active (attribution) immunity.** Low TECS would mean that ROME edits and TDA gradients target different "epitopes" (parameter subspace regions) of the same factual "antigen" -- exactly as in passive vs active immunity.

### Grounding in Existing Cross-Disciplinary Work

1. **Artificial Immune Systems (AIS) literature**: The clonal selection algorithm (CLONALG, de Castro & Von Zuben, 2002) directly implements affinity maturation as an optimization algorithm. In AIS, the mutation rate is inversely proportional to affinity -- better-matching solutions are mutated less. **Implication for TECA**: Training data that contributes most to learning a fact (high TDA influence) should have gradients that are more "refined" (closer to the final parameter configuration), while low-influence data provides noisy, undirected gradients. TECS should be higher when computed with high-influence training examples.

2. **Immune memory and dynamic memory cells** (ScienceDirect, 2025 -- "An artificial immune classification method with deep feature enhancement and dynamic memory cells optimization"): Implements dynamic memory cells that store optimized representations through immune-inspired selection. **Implication for TECA**: The concept of "memory cells" in immunology (long-lived B cells that maintain high-affinity antibodies) maps directly to the MLP weight configurations that stably encode facts. The "quality" of these memory cells (affinity) should predict TECS -- well-optimized knowledge encodings should show higher alignment between the editing and attribution signals.

3. **Multiscale information processing in the immune system** (Frontiers in Immunology, 2025): Analyzes how information flows across spatial and temporal scales in the adaptive immune response -- from molecular (antibody-antigen binding) to cellular (B cell selection) to systemic (immune memory). **Implication for TECA**: Knowledge encoding in transformers similarly operates across scales -- from individual weight entries to neurons to layers to the full network. TECS at the single-layer level may miss multi-scale structure. The immunological perspective suggests analyzing TECS at multiple scales: per-neuron, per-layer, and cross-layer aggregated.

4. **Natural Machine Learning (NML) and the Adaptive Immune System** (Preprints.org, 2025): Proposes that the adaptive immune system performs a form of natural machine learning, with somatic hypermutation as the learning algorithm and antigen binding as the loss function. The convergence trajectory in antibody sequence space has a well-defined geometry governed by the affinity landscape. **Implication for TECA**: The TDA gradient traces the convergence trajectory of parameter optimization (the "somatic hypermutation path"), while the ROME edit is a direct jump to a target in parameter space (the "passive immunization injection"). TECS measures the angle between the natural maturation path and the direct injection vector.

### Concrete Experimental Plan

| Step | Description | Model | Time |
|------|-------------|-------|------|
| 1 | Rank training examples by TDA influence score; compute TECS separately for top-5, mid-5, and bottom-5 retrieved documents | GPT-2-XL | 10 min |
| 2 | Test "affinity maturation prediction": TECS should monotonically increase with TDA influence rank (higher influence = more "matured" = better aligned) | - | 5 min |
| 3 | Compute "cross-reactivity score": for each edited fact, measure TECS on semantically related but distinct facts (same relation, different subject); this is the "epitope overlap" measure | GPT-2-XL | 15 min |
| 4 | Test "autoimmunity prediction": facts where TECS cross-reactivity is high should show more editing side effects (lower locality score) | - | 5 min |
| 5 | Compute multi-scale TECS: aggregate TECS across layers l*-2 to l*+2 (weighted by causal tracing activation); compare signal with single-layer TECS | GPT-2-XL | 10 min |

**Total**: ~45 min. **Success probability**: 40%.

### Testable Predictions

**P7** (Affinity-TECS monotonicity): TECS computed with top-k highest-influence training examples will be significantly higher than TECS computed with bottom-k, because high-influence examples are the most "affinity-matured" contributors to the fact's encoding.

**P8** (Epitope divergence): For related facts sharing the same relation (e.g., "X is the capital of Y" for different X,Y), the cross-fact TECS (editing direction for fact A vs attribution gradient for fact B) will be positive but significantly lower than same-fact TECS. This parallels antibody cross-reactivity: antibodies matured against one antigen partially bind related antigens but with lower affinity.

**P9** (Multi-scale integration improves signal): TECS aggregated across the editing-relevant layer window (l*-2 to l*+2, weighted by causal tracing scores) will show higher Cohen's d than single-layer TECS, because knowledge encoding operates across scales -- just as immune responses integrate information from molecular to systemic levels.

### Why This is Novel

The immunological analogy has been applied to optimization (CLONALG, immune-inspired search), anomaly detection (negative selection), and clustering, but **never to the relationship between knowledge editing and data attribution in neural networks**. The specific insight about "active vs passive immunity" as a framework for understanding "training vs editing" is new and generates the non-obvious prediction P7 (TECS should correlate with influence rank). The "cross-reactivity" analysis (P8) is a genuinely new experimental design inspired directly by immunological epitope mapping.

### Failure Modes

- **Influence ranking is itself unreliable**: If TDA influence scores for LLMs are noisy (as suggested by arXiv 2409.19998), sorting by influence may not meaningfully stratify the training examples. Mitigation: use BM25 score as an independent proxy for "relevance" alongside TDA influence.
- **Cross-reactivity TECS is trivially low**: In a 10^7-dimensional parameter space, any two random directions have near-zero cosine. Cross-fact TECS may simply reflect this dimensionality rather than meaningful epitope overlap. Mitigation: compare against the null distribution (Null-A) and report effect size relative to null.

---

## Synthesis: How the Three Angles Interconnect

The three interdisciplinary angles are not independent proposals but form a coherent, multi-level framework:

```
Level 3: IMMUNOLOGY (Functional / Teleological)
  "Why does TECS take the value it does?"
  → Active vs passive immunity: training naturally matures knowledge encoding
    along specific directions; editing injects solutions from a different path.
  → TECS measures the convergence of these two paths.

Level 2: NEUROSCIENCE (Systems / Architectural)
  "What computational architecture produces this pattern?"
  → CLS theory: fast sparse encoding (editing/hippocampal) vs
    slow distributed encoding (attribution/neocortical).
  → TECS measures the alignment between two complementary memory systems.

Level 1: STATISTICAL PHYSICS (Mathematical / Structural)
  "What mathematical structure underlies these patterns?"
  → Spin-glass energy landscape: knowledge facts as metastable states,
    editing as external field, attribution as natural gradient.
  → TECS as replica overlap, organized by RSB ultrametricity.
```

Each level provides distinct predictions that can be tested independently, but they also make joint predictions:

- **Joint prediction 1**: Well-consolidated facts (high frequency, low perplexity) should simultaneously show (a) higher TECS (neuroscience: better engram consolidation), (b) deeper energy minima in the loss landscape (physics: more stable attractors), and (c) higher-affinity knowledge encoding (immunology: more matured antibodies). These three independently motivated predictions converge on the same measurable quantity.

- **Joint prediction 2**: The layer specificity of TECS (high at l*, low elsewhere) should be explained by (a) a phase transition at l* (physics: beta_eff crosses critical threshold), (b) the CLS encoding locus (neuroscience: l* is where episodic and semantic traces maximally align), and (c) the site of highest "antigen presentation" (immunology: l* is where factual information is most accessible for both operations).

---

## Computational Budget Summary

| Angle | GPU Time | CPU Time | Total | Shared with |
|-------|----------|----------|-------|-------------|
| 1. Spin-Glass Overlap | 20 min | 15 min | 35 min | Base TECS computation |
| 2. CLS Engram Geometry | 0 min | 20 min | 20 min | Angles 1, 3 |
| 3. Immunological Affinity | 35 min | 10 min | 45 min | Angle 1 |
| **Combined (with caching)** | **~40 min** | **~30 min** | **~70 min** | |

All experiments fit on a single RTX 4090. The interdisciplinary analyses are largely post-hoc computations on top of the base TECS data, so they add minimal GPU overhead.

---

## Recommended Priority

1. **Start with Angle 1** (Spin-Glass): This provides the deepest mathematical framework and makes predictions about the *structure* of the TECS overlap matrix that no other perspective generates. The ultrametric clustering test (P1) and layer-specific phase transition test (P2) are computationally cheap and high-information.

2. **Then Angle 2** (Neuroscience/CLS): The consolidation-TECS correlation (P4) is the most directly testable prediction with existing data. If confirmed, it provides a powerful narrative for the paper: "TECS measures how well the knowledge is consolidated, bridging the editing and attribution communities through the lens of complementary learning systems."

3. **Angle 3 last** (Immunology): The affinity maturation analogy generates the most novel experimental designs (cross-reactivity analysis, influence rank stratification) but depends on reliable TDA influence scores, which are known to be noisy for LLMs. Run this after validating that the base TECS signal is detectable.

---

## Key Cross-Disciplinary Literature

| Paper | Source | Cross-Disciplinary Relevance |
|-------|--------|------------------------------|
| Neural Networks as Spin Models (Barney et al., 2024) | arXiv 2408.06421 | Maps NN weights to spin couplings; proves untrained networks are SK spin glasses; training creates hidden order phase |
| Exploring Loss Landscapes through Spin Glass Theory (Liao et al., 2024) | arXiv 2407.20724 | Demonstrates RSB-like hierarchy in DNN loss landscape; provides analysis protocols for ultrametric structure detection |
| A Spin Glass Characterization of Neural Networks (Li, 2025) | arXiv 2508.07397 | Hopfield-type spin glass from feedforward NNs; replica overlaps as descriptors for individual network instances |
| Phase Transition in Modern Hopfield Networks (Koulischer et al., 2023) | arXiv 2311.18434 | Critical temperature for pattern-specific attractor formation; effective beta depends on stored pattern distribution |
| Beyond Scaling Laws: Transformer Performance with Associative Memory (Niu et al., 2024) | arXiv 2405.08707 | Energy function framework for transformer attention via Hopfield networks; memorization-size dependency |
| Dynamic Manifold Hopfield Networks (Li et al., 2025) | arXiv 2506.01303 | Context-dependent reshaping of attractor geometry in associative memory; 64% retrieval at 2N patterns |
| Semantic and Episodic Memories in Predictive Coding (Fontaine & Alexandre, 2025) | arXiv 2509.01987 | CLS theory in predictive coding; episodic recall from semantic learning only with few examples |
| Hare and Tortoise Networks (Lee et al., 2024) | arXiv 2406.02596 | CLS-inspired dual-rate learning; Hare = fast/hippocampal, Tortoise = slow/neocortical |
| Engram Memory Encoding and Retrieval (Szelogowski, 2025) | arXiv 2506.01659 | Comprehensive framework: sparsity, Hebbian plasticity, interference resistance in engram formation |
| Hebbian Memory-Augmented Recurrent Networks (Szelogowski, 2025) | arXiv 2507.21474 | Explicit Hebbian memory matrix with observable dynamics; bridges engram neuroscience and deep learning |
| Dynamic and Selective Engrams (Nature Neuroscience, 2023) | Nature Neuroscience | Engram composition changes during consolidation; inhibitory plasticity drives selectivity |
| Wake-Sleep Consolidated Learning (Sorrenti et al., 2024) | arXiv 2401.08623 | CLS + wake-sleep in continual learning; NREM consolidation, REM dreaming for forward transfer |
| TDA for Neural Network Analysis: Survey (Ballester et al., 2023) | arXiv 2312.05840 | Comprehensive survey of persistent homology applied to NN internal representations and parameter spaces |
| STEAM: Semantic-Level Knowledge Editing (Jeong et al., 2025) | arXiv 2510.10398 | Edited knowledge encoded as isolated residual streams -- the "failed engram" in our CLS framework |
| Natural Machine Learning Shapes the Adaptive Immune System (2025) | Preprints.org | Adaptive immune system as natural ML; somatic hypermutation as learning algorithm |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Spin-glass analogy is too loose for quantitative predictions | 30% | High | Use the framework for hypothesis generation only; validate all predictions empirically; report deviations as evidence of transformer-specific structure |
| CLS predictions are trivially confirmed (all methods show frequency dependence) | 25% | Medium | Design controls: if TECS-frequency correlation is no stronger than simple loss-frequency correlation, the CLS interpretation adds nothing |
| Immunological analogy generates predictions that are not distinguishable from simpler baselines | 35% | Medium | The cross-reactivity experiment (P8) is genuinely novel and cannot be derived from simpler frameworks; prioritize this over P7 |
| Reviewers dismiss interdisciplinary angles as "just metaphors" | 40% | High | Ground every analogy in specific, falsifiable predictions with effect-size estimates; present the spin-glass overlap matrix as a concrete mathematical object, not a vague analogy |
| The 100-fact dataset is too small for meaningful clustering/ultrametric analysis | 25% | Medium | Use bootstrap resampling; compare against random permutation null; report minimum sample size for detectable structure |

**Overall probability of at least one interdisciplinary angle yielding a publishable insight**: ~60%.

The greatest value of the interdisciplinary perspective may not be in the predictions themselves but in the **framing**: recasting TECS as a replica overlap (physics), a CLS alignment metric (neuroscience), or an epitope correspondence measure (immunology) makes the paper accessible and interesting to a much broader audience than a purely ML-internal analysis.
