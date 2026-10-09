# Synaptic research desk: comparative research and design decisions

Research date: October 9, 2026. This notebook connects primary sources to concrete interface and training choices. It is a design review, not a claim that the project reproduces the cited scientific models.

## The problem with the previous screen

The previous viewer fitted every neuron anchor, but anchors alone are not recognizable cell morphology. Pale transparent surfaces and a single optional skeleton left the brain looking like a cloud. The display also prioritized a short list of engineered readout cells; biological voltages, directed connectivity, synaptic parameter identity and historical comparisons were difficult to find. A technically correct point cloud did not meet the user's need to understand an anatomical brain executing a decision.

The old learning pipeline also optimized the decoder while keeping every fly connection fixed. That was a usable engineering baseline, not the requested synaptic-learning stage. The new screen must show the distinction as measurable state: the number of existing biological edges eligible for training, the number actually changed, their original and current strengths, and the identity of the graph used in the live run.

## Comparative review of scientific tools

| Primary source | Relevant capability | Design consequence here |
|---|---|---|
| [FlyWire Codex FAQ](https://codex.flywire.ai/faq) | Cell search, annotations, selected-cell details, morphology, input/output partners, bounded connectivity graphs and pathways | Add exact root-ID/type/class search and a persistent cell inspector. Show partner direction and numerical weights. Avoid presenting a dense global edge tangle as a useful full-brain map. |
| [Neuroglancer repository](https://github.com/google/neuroglancer) and [layer documentation](https://neuroglancer-docs.web.app/concepts/layers.html) | WebGL visualization separates image/segmentation layers, object surface meshes and object skeletons | Keep anatomical surfaces, reference anchors, real neuron skeletons and simulated spike activity independently visible. Their meanings must not be merged into a single color scale. |
| [neuPrint: An open access tool for EM connectomics](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.896292/full) and [official explorer](https://neuprint.janelia.org/help/videos?dataset=hemibrain%3Av1.2.1&q=1&qt=findneurons) | Query-oriented connectivity exploration, shortest-path queries and embedded 3D neuron inspection | Couple a selected neuron to a directed connection table and its geometry. Keep detailed data accessible alongside visualization. The current implementation does not claim to include shortest-path analysis. |
| [Virtual Fly Brain](https://natverse.virtualflybrain.org/) | Integrates neuroanatomical information and image datasets in shared template space for comparison and search | State the coordinate space and release explicitly. Join data by exact IDs and preserve anatomical context when examining a cell. Do not silently combine different fly datasets or templates. |
| [Janelia SharkViewer](https://github.com/JaneliaSciComp/SharkViewer) | A browser-based neuron viewer with multiple neuron representations and separate synapse annotations | Load multiple real skeletons, retain their cell identity and bound the scene size. Skeleton branches are not synapse locations, and graph arrows are not reconstructed axon paths. |

Codex's current public FAFB summary uses different connection filtering from this project's prepared minimum-one-contact matrix. Our 15,091,983 directed pairs therefore must come from the local manifest, not a number copied from a website. Individual contacts, directed neuron pairs and trainable parameters are three different counts.

The official neuPrint interface currently requests authentication for queries, and Codex likewise documents interactive sign-in. Consequently, embedding those sites would not reliably provide a local, reproducible, synchronized Doom instrument. Their interaction patterns inform this design; their applications are not bundled or impersonated. The existing local WebGL renderer is extended, keeping the project functional with its cached public anatomy and without adding a large frontend dependency solely for cosmetic changes.

## Visual and interaction specification

1. **A research desk, not a landing page.** Compact typography, an instrument identifier, measured counts and a restrained green/paper palette replace the large promotional header. A dark anatomical viewport separates real geometry from surrounding tables. No invented activity animations are used.
2. **Keep observation and anatomy together.** The wide desktop layout puts the anatomical workspace beside a large Doom image. The image switches explicitly between the policy's input and an available action outcome. Neural values always describe the input that generated the selected decision. An unavailable terminal outcome is labeled rather than fabricated.
3. **Recognizable whole-brain anatomy first.** Load all 75 real region surfaces with directional shading and depth testing. Camera fitting considers every region, all reference positions and loaded branches. Source-plane presets are XY/XZ/YZ; they are not relabeled as anatomical anterior/dorsal axes without a verified transform.
4. **Independent scene layers.** Surface opacity, all anchors, live spikes, reconstructed branches and selected directed connections have separate controls. Anchors start hidden so they do not obscure the surface shape; all remain available. Spikes are amber measurements from the retained simulation window, not an interpretation of region color.
5. **Several real neurons at once.** A bounded context set is loaded from the v783 skeleton service. Users can load a searched cell and three connected partners. At most 16 cells occupy the browser skeleton layer; discarded buffers are released. This is a local neighborhood/context view, not all 139,255 complete neuronal arbors.
6. **Restore biological detail.** The inspector exposes exact root ID, annotated type/class/transmitter, voltage, synaptic current, refractory state, spike count and incoming/outgoing pair counts. The local connection table shows source/target direction, original strength, current strength, multiplicative gain and plasticity group.
7. **Separate biological and engineered populations.** The activity tab lists spiking fly neurons with voltages and exposes all 64 engineered cells. Clicking an engineered cell shows all six signed readout products. A positive contribution is an arithmetic logit term, not proof of causal control by that cell alone.
8. **Make training inspectable.** Plot the measured KL of every candidate, including rejected or worse candidates where available. Show train/validation before-and-after metrics, parameter bounds, group sizes and a sample of actual changed edges. Do not replace a failed search with a success narrative.
9. **Inspect one time point consistently.** The decision strip references retained sequence IDs. Selecting a past decision pauses further execution and retrieves the matching image, neural activity and inspector data. Expired snapshots produce an explicit message; they are not silently substituted with the latest state.
10. **Preserve provenance without hiding it in logs.** Graph, patch, student and training-plan hashes appear in the protocol tab. Exports and the output directory are reachable from the same workspace. The anatomical coverage and biological assumptions are written next to the scientific context.
11. **Control memory deliberately.** Retain eight full neural snapshots, one graph/controller and bounded skeleton buffers. The live desk does not instantiate the Laya worker. Clearing the branch layer releases its WebGL buffers. Cache cleanup must not delete training evidence or checkpoints.
12. **Responsive, but scientifically useful.** Desktop retains side-by-side observation and brain. Narrow screens stack panels and scroll wide numeric tables within their own containers. The expanded anatomical view and Escape return are explicit controls. No data table is simply removed on a small display.

## Why the first synaptic method is constrained

[Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3) demonstrates task optimization of a connectome-constrained model of the fly visual system. Their model uses connectivity and synapse-count information with shared parameters, including cell-type connection-strength factors; it is not this project's whole-brain LIF/Doom model. [Flyvis](https://github.com/TuragaLab/flyvis) is the authors' public implementation and provides a useful reference for future differentiable mechanistic modeling. These sources motivate explicit constraints and parameter sharing rather than arbitrary dense rewiring.

Our immediate implementation keeps the exact existing LIF dynamics. A positive gain multiplies selected existing directed-edge strengths:

```text
W_after[target, source] = W_base[target, source] * gain[group(source)]
0.75 <= gain <= 1.25
```

The first mask includes nonzero connections into the 1,303 descending output neurons, grouped by the four largest presynaptic super-classes by absolute initial weight. This ties 293,249 eligible directed-edge entries to four parameters; it does not independently train 293,249 free parameters or all 54 million anatomical contacts. The full 139,255-cell network still executes for each image. The other graph entries, transmitter signs, neuron dynamics, input mapping, visual teacher and engineered readout stay fixed.

The objective is KL divergence from the student's new probabilities to the already recorded six-option Laya targets, plus a small gain-change penalty. The frozen readout remains part of that computation: graph changes must propagate through actual neural activity and the existing decoder to affect the objective. Reusing cached descending features would break that dependence, so each candidate replays the complete graph from an episode reset and applies the recorded causal action history in order.

This initial optimizer is gradient-free. It does not differentiate through millions of discontinuous spikes, and it does not claim to use surrogate-gradient backpropagation, STDP, reinforcement learning, LoRA or RAG. The advantage is immediate parity with the actual inference simulator; the cost is repeated simulation and a very small parameter budget. A later differentiable model requires explicit forward-parity and gradient checks, temporal-state treatment, memory profiling and a separately specified training protocol.

## Experiment separation and negative results

The first coarse pilot evaluates coordinate changes of 0.15 and 0.075. None improved its penalized training objective, so it correctly retained the initial graph and reported zero changed edges. The completed report is preserved at `runs/synaptic-pilot-v1`. A second, separately locked pilot uses smaller paired joint perturbations of 0.01 and 0.0025 with four seeded directions. It responds to the observed training sensitivity, not validation or gameplay outcomes. Both searches retain the unchanged graph as a candidate.

Each pilot uses the first two chronological frames of eight existing training episodes (16 examples) and four existing validation episodes (8 examples). This is deliberately a small implementation experiment; the student has already seen these scene families. Validation is measured before and after and is not used to choose gains. Two familiar 12-decision game starts are only a development smoke comparison. They cannot establish generalization, biological validity, reliable gameplay improvement or a topology advantage.

The next research protocol should increase complete trajectories, independently reserve unseen test starts, compare base/modified graphs with the same decoder, test rehearsal and failure states, and evaluate multiple training seeds. Fine-grained cell-type or edge-level parameterization should be introduced only with tests for sign/mask preservation and rollout stability. A gain learned for an engineered Doom task is a model parameter, not a measured change to a living fly's synaptic efficacy.

## Implemented follow-up: individual-edge eligibility proposals

Both shared-gain searches retained the original graph. The final working checkpoint therefore uses a third, separately locked experiment: individual gains on the same existing-edge mask. The four annotation groups remain in the interface as summaries; their means are not four free parameters in this checkpoint.

For each selected edge, a local eligibility trace follows the derivative of synaptic current and postsynaptic voltage with respect to that edge's strength. It uses the simulator's exact exponential membrane/synapse factors, actual delayed presynaptic arrivals and reset/refractory events. The frozen decoder supplies a surrogate gradient of the teacher KL with respect to descending voltage features. Their product proposes a direction for each existing synapse. Presynaptic spike times and reset decisions are treated as fixed in this derivative; indirect recurrent feedback and spike-rate-feature derivatives are omitted. This is a direct local approximation, not full recurrent backpropagation, a validated plasticity mechanism, or differentiation of the entire hard-spike graph.

Before using the proposal, a two-cell test checks the direct voltage derivative against central finite differences in a regime with unchanged spike timing. An observation-only eligibility trace also reproduces the ordinary forward voltage exactly. Eight signed step lengths then test the proposal against the actual whole-network simulator, including its discontinuous spikes. Only penalized training KL accepts a candidate; validation is still reported without selecting parameters.

This run found 14,484 edges with a nonzero local learning signal and accepted a step of 0.0001. Float32 application changed 13,867 actual CSR entries. The largest gain changes are about ±0.01%; smaller changes may round away. Training KL changed from 0.1429181 to 0.1425697; validation KL changed from 0.1743336 to 0.1733302. These are modest improvements on 16/8 already-seen scene-prefix examples. The two familiar development starts produced no hit in either condition within 12 decisions. The result establishes executable synaptic optimization and checkpoint-to-live integration, not better general Doom performance.

The live desk defaults to `runs/synaptic-eligibility-v1`. Original decoder weights, all unselected graph entries, topology and transmitter signs remain unchanged. All three completed searches are retained. The interface shows per-edge gains at sufficient precision to reveal these small updates, and displays annotation-group means with their minimum/maximum range.

## Acceptance checks

- Numerical: a plastic-edge change affects LIF voltage, while unselected entries and CSR topology are bitwise unchanged; nonfinite/out-of-bound gains are rejected before mutation.
- Identity: checkpoint loading verifies base matrix, source/target IDs, selected baselines, reconstructed strengths, trained graph hash and frozen decoder hash.
- Interface: measured surfaces render with depth; every loaded morphology remains identifiable; search, local connections, six output contributions and training history work.
- Temporal: pause/review/manual/automatic states remain distinct; a selected past decision cannot show new activity under its old image.
- Resources: bounded snapshots and skeleton buffers, no idle GPU teacher, closed test browsers and their disposable profiles removed.

Measured outcomes and browser evidence are recorded in `docs/VALIDATION.md` after execution. Features not implemented here include complete EM-volume rendering, every-cell morphology streaming, individual synapse coordinates, and shortest-path query analysis.
