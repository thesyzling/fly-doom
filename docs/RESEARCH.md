# Fly Doom research

Research date: September 26, 2026. This document distinguishes source findings from project decisions.

## Feasibility

The goal is to teach Doom control to a neural network constrained by the real fruit fly connectome. Connectivity alone does not determine neuronal electrical states, all receptors, synaptic strengths, or learning rules. The result will therefore be a computational model grounded in biological data; we will not claim it is a complete copy of a living brain.

The FlyWire adult female brain study includes 139,255 neurons and approximately 50 million chemical synapses. Directed neuron-pair counts and synapse counts are different: one pair can have many synaptic contacts. [Dorkenwald et al., Nature 2024](https://www.nature.com/articles/s41586-024-07558-y).

Shiu et al. built a brain-wide leaky integrate-and-fire (LIF) model using connectivity and neurotransmitter information, and studied feeding and grooming circuits. This provides a strong starting point for a dynamic model, but does not demonstrate Doom learning. [Paper](https://www.nature.com/articles/s41586-024-07763-9), [authors' code](https://github.com/philshiu/Drosophila_brain_model).

The 2026 FlyGM preprint trains a connectome-derived graph controller for simulated fly locomotion using reinforcement learning. It provides an example of a trainable controller constrained by connectivity. Its reported results concern a different environment and do not guarantee our Doom performance or biological validity. [FlyGM, arXiv v3](https://arxiv.org/abs/2602.17997v3).

## Dataset selection and access

We selected the **FAFB v783 publication archive** for the initial version: it offers a starting point close to the Shiu model and fixed source file versions. This does not mean it is the newest nervous system reconstruction. At the research date, Codex also lists datasets including BANC v888 and MCNS v1.0. We will not arbitrarily mix IDs and annotations across versions. Current Codex exports can differ from publication archives, and downloads can require an account or API token. [Codex FAQ](https://codex.flywire.ai/faq).

Selected inputs from [Zenodo 10676866](https://zenodo.org/records/10676866):

| File | Approximate size | Purpose |
|---|---:|---|
| `proofread_root_ids_783.npy` | 1.1 MB | Neuron IDs, including isolated neurons |
| `proofread_connections_783.feather` | 852 MB | Synapse counts per neuron pair and region |

The archive connection table includes `pre_pt_root_id`, `post_pt_root_id`, `syn_count`, regions, and neurotransmitter probabilities. The 9.5 GB individual-synapse file is not required for the initial graph. The current preparation code extracts directed connectivity counts without assigning neurotransmitter signs. Source MD5 values check download integrity; SHA-256 values are recorded in the local manifest. The source archive's existing quality filters still apply.

Preparation applies no additional synapse threshold, retains any self-connections, and sums regional rows for each directed pair. Matrix rows are targets and columns are sources. Regional and neurotransmitter information will be processed separately from the raw file when dynamics are added. Annotations and visual input and output populations must be matched to the same dataset version before simulation.

## Lessons from related Doom projects

[nftechie/doomfly](https://github.com/nftechie/doomfly) publishes a Doom experiment running on a real connectome. Its README reports that the current experiment failed visual, conditioning, and survival validation gates. Changing weights alone is not evidence of learning. We reviewed this report but did not independently run the upstream code.

[shreyash-sharma/doomFly](https://github.com/shreyash-sharma/doomFly/blob/main/docs/how-it-works.md) freezes connectivity and trains a small readout. It documents additional neural inputs derived from game state, including enemy direction. This differs from learning to play through pixels alone. Our primary experiment will not give the policy enemy coordinates or object labels.

## Proposed architecture

```mermaid
flowchart LR
    A[Doom image] --> B[Visual encoder]
    B --> C[Input neurons selected by annotation]
    C --> D[Real connectivity graph and neuron dynamics]
    D --> E[Selected neural activity]
    E --> F[Trainable action readout]
    F --> G[Game buttons]
    G --> A
    H[Training reward] -. parameter update .-> F
```

This diagram is the original training design. The experimental LIF kernel now runs inside a bounded pixel-to-game bridge, and a trainable engineered readout has been implemented, including offline Laya Vision distillation. The reward-driven parameter update shown here remains unimplemented; current teaching uses recorded target probabilities. See [the bridge assumptions and controls](BRIDGE.md) and [the Vision training procedure](VISION_WORKBENCH.md#offline-teaching-first-vision-student).

1. Start with a fixed graph and a small trainable readout. Inputs come only from images, and the output policy has no direct access to those images. This is a learned controller on top of a fixed connectivity model.
2. Then train the magnitudes of existing edges. The graph mask remains fixed, with no new connections. Record signs, weight bounds, and deviations from biological initialization. Choose between PPO/BPTT and local plasticity after initial performance measurements.
3. Use an LIF reference for the biological modeling path. A sparse rate model is a possible engineering comparison. If used, it will not be presented as LIF or as an exact reproduction of the paper.

The visual encoder is a central research challenge. Randomly assigning screen coordinates to neurons does not reconstruct natural fly vision. Visual-field mapping, annotation matching, and signal transmission to outputs will first be tested with small stimulation experiments. Inferring an effect's sign from neurotransmitter class is also a model assumption that depends on receptor context; unknown classes will not silently become excitatory.

## Game and experiment design

[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) provides pixel observations, scenario control, and a Python interface. Windows is supported; maintainers recommend considering Linux/WSL for long experiments. The package can run using Freedoom assets. Initial setup is tested on Windows with the window hidden and sound disabled.

Curriculum: `basic` target shooting, then visual orientation and navigation, survival, and more complex maps. Success on `basic` does not mean completing a Doom level. Rewards are training signals; privileged state such as health or position must not leak into the policy. Neural state will reset at episode boundaries; persistent memory will be enabled only as a separate experiment.

Controls include a random policy, an untrained readout, a standard network with the same training budget, a retrained degree-preserving rewired graph, and a graph silenced during evaluation. Silencing demonstrates circuit dependence only; retrained comparisons are needed to establish an advantage from biological topology.

Use at least three independent training seeds and at least 20 evaluation seeds unseen during tuning. Keep training, validation, and test seeds separate. Measure success, return, accuracy, survival, and level completion as appropriate to the scenario. Report distributions and uncertainty; the best single video will not serve as the metric. Stronger learning claims require examining evaluation uncertainty alongside differences from controls.

## Resource budget and milestones

Local GPU: NVIDIA RTX 3060 Laptop with 6,144 MiB VRAM. System RAM information could not be read in the initial session. A dense 139,255-by-139,255 float32 matrix would occupy approximately 77.6 GB (72.2 GiB), making sparse storage necessary. That estimate covers weights only. A sparse graph fitting in memory does not guarantee that gradients across time and optimizer states will also fit.

- M0: literature review, data preparation tools, and ViZDoom integration check.
- M1: real source downloads, checksum verification, graph report, and matched annotations.
- M2: finite and stable model activity, stimulation responses, visual input-to-output transmission, step speed, and memory measurements.
- M3: training with a fixed graph and control experiments.
- M4: training existing synapses and comparing under the same evaluation protocol.
- M5: recordings, video, and more difficult Doom tasks.

September 26 follow-up: M1 is complete. The real graph was prepared and the publication-matched `flywire_annotations` v2.1.0 rows were aligned with all 139,255 neuron IDs. Measurements and limitations are in the [validation record](VALIDATION.md).

September 27 follow-up: an experimental LIF kernel and four bounded whole-graph stimulation conditions now run. M2 remains incomplete: photoreceptor stimulation affects downstream voltage but does not elicit descending spikes, and large voltage excursions require calibration. The positive control bypasses the retina. See [model assumptions, timing, and negative results](SIMULATION.md).

Long training runs will not begin before passing M2. If a subcircuit is needed instead of the whole brain, its scope will be recorded explicitly and will not be described as whole-brain simulation. Training time estimates will follow the first benchmark.

September 27 integration follow-up: a bounded engineering bridge now feeds coarse screen brightness directly into positive visual projection cells and maps descending spike rates to game buttons. This deliberately bypasses the unresolved retinal pathway, uses artificial assignments, and does not constitute passage of M2 or evidence of learning. Disconnected and zero-input controls are included; physiological calibration remains outstanding.

## Scope review after Vision teaching and recovery evaluation (October 6, 2026)

The engineering prototype now meets the immediate goal: real connectivity drives an added trainable student, Laya Vision supplies offline teaching targets, the trained student plays the basic shooting scenario without executing Vision, and recordings plus decision/weight inspection are available. The larger research roadmap remains incomplete.

| Milestone | Current status | Remaining work |
|---|---|---|
| M0–M1 | Completed for the selected dataset and initial setup | Revisit sources only when changing the scope or dataset |
| M2 | Calibrated engineering bridge runs; biological validation remains incomplete | Resolve the retinal pathway and physiological assumptions; the brightness-bin bridge is an artificial bypass |
| M3 | Fixed-graph training, Vision distillation, three recovery continuation seeds and their locked 20-opening gameplay comparison completed | Retrained matched network/topology controls, independent initializations and candidate-state recovery teaching. Continuations share one warm-start parent; paired intervals include zero, and older parent-training independence is unresolved |
| M4 | Not started | Train magnitudes of existing biological edges under explicit sign/mask/bound constraints if pursuing this later research stage |
| M5 | Interactive game recordings and numerical inspection available | Orientation, navigation, survival and harder maps; current success is limited to `basic` |

The first candidate scored 6/6 target kills against the parent's 5/6 on a small development comparison. Two evaluation openings repeat training images. The subsequent locked evaluation measured 19/20 candidate hits, 15/20 parent hits and 0/20 for the disconnected candidate, but 13 starts repeat a single training opening. The seven remaining openings give 6/7 versus 2/7 hits as a descriptive subgroup. See the [full protocol, uncertainty and scene-overlap audit](VALIDATION.md#locked-20-start-vision-student-validation-october-3-2026). These findings support fixed-policy circuit dependence and better performance in this scenario, not a biological-topology advantage or broad generalization.

The recovery teaching stage used 12 distinct training openings and 6 validation openings, with 20 further distinct openings reserved before teaching. All three continuations completed those 20 starts against their common parent: 15/20, 14/20 and 15/20 hits versus 13/20, with better measured mean returns. Every paired uncertainty interval includes zero. Each candidate recovered some failures and lost four parent successes; frequent LEFT/RIGHT changes remain a problem. All actual openings matched the locked scan, and complete RGB/input trajectory overlap was audited. No gameplay-based seed selection or checkpoint promotion occurred. See [the protocol, regressions and independent artifact checks](VALIDATION.md#reserved-recovery-gameplay-comparison-october-6-2026).

The next M3 work is retrained matched network/topology controls and independent initialization studies. Before further recovery tuning, reserve a fresh test set, collect observations reached by the new candidates, and evaluate rehearsal of earlier useful behavior. The previous parent-controlled collection does not fully cover the candidates' new closed-loop states. Both inspected evaluation sets are now development evidence. M4 is a separate extension rather than a prerequisite for the fixed-graph prototype. Automatic online learning is not implemented; current training remains offline.

## Live handoff and data expansion (October 9, 2026)

`flydoom.experiment` now connects completed training and benchmark identity to explicit live model selection. The browser presents the common parent and all three continuations without promoting a gameplay winner. It reloads the selected model's inspector and starts a fresh paused game; all weights remain fixed during play. The map starts with the entire anchor cloud enlarged and higher contrast. Full neuron morphology still requires additional anatomical data and is not supplied by this rendering change.

The next recovery reservation is `runs/vision-recovery-reservation-v2/plan.json`: 24 training starts, 8 validation starts and 20 future evaluation starts. An outcome-free scan examined 151 openings, excluding 529 known RGB identities from the discovered local records and retained teaching data. Evaluation openings were allocated first. No policy actions, Vision targets or rewards were collected in this reservation. Exact opening separation does not establish semantic independence or audit every historical trajectory PNG. The next collection/training protocol must still fix policy rotation, rehearsal proportions, training budget and validation selection before execution. The live application protects these future evaluation seeds from demonstrations.

External data review:

| Source | Observed format | Fit to this project |
|---|---|---|
| [Hugging Face DoomFrameDataset](https://huggingface.co/datasets/brahmandam/DoomFrameDataset) | Policy-generated RGB/action rollouts with episode/step metadata; approximately 2.4 million samples and 68 GB. Preview includes turning and combined actions. | A candidate for later visual/action studies, but not a drop-in four-action teacher dataset. Its card describes policy rollouts, not expert human demonstrations. |
| [GitHub Doom gameplay dataset](https://github.com/thavlik/doom-gameplay-dataset) | Preprocessed gameplay videos and frame/video indices | Useful for studying visual representations; the documented format does not provide our causal fly features or aligned four-action teacher targets. |
| [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) | Controllable game environment | Preferred immediate expansion: generate ordered observations with our exact scenario, action timing, fly dynamics and causal memory, then ask the pinned Vision teacher for labels. |

No external training dataset was downloaded or merged in this step. External ingestion needs explicit episode boundaries, action semantics, source revision/licensing, teacher quality checks and train/validation/test separation. The stateful fly inputs must be regenerated from chronological observations under the project's simulator, rather than inferred from isolated screenshots. Anatomical morphology is a separate data need; [FlyWire's annotation repository](https://github.com/flyconnectome/flywire_annotations) links downloadable skeletons for its dataset.

## Anatomical surfaces and six-action pilot (October 9, 2026)

The map now adds 75 real FlyWire neuropil surfaces to the 139,255 reference positions and can download the selected neuron's actual v783 skeleton. Camera fitting includes the surfaces and selected branches. These additions supply anatomical shape without claiming that all full cell morphologies or synapses are rendered simultaneously. Surface color is not an activity measurement.

The separate six-action student adds forward and backward motor outputs, extends causal history, and transfers matching parent weights and normalization. A bounded offline Vision teaching pilot collected 165 training and 96 validation examples; validation teacher agreement increased from 10.42% to 47.92%. Biological connections and the Vision teacher remain fixed. The live mode executes the trained student without a Vision worker, and explicit manual motor checks demonstrate both directions in the engine without updating weights.

Backward is an available motor action, but the teacher supplied no backward top-choice labels in either split. This pilot therefore does not establish learned retreat. The next movement work needs a scenario and verified teaching examples in which retreat is useful, rehearsal of earlier behavior, and a new held-out evaluation protocol. The current basic shooting scenario, one training seed and small paired benchmark do not validate navigation. M2, matched topology controls under M3, and biological synaptic training under M4 remain unresolved. See the [validation record](VALIDATION.md) for the measured six-action comparison and its uncertainty.

## M4 started: constrained biological-edge learning (October 9, 2026)

M4 is now started at the engineering-pilot level. Two shared-gain searches made no accepted change; their reports are retained. A third experiment used direct local eligibility traces and a frozen decoder gradient to propose independent gains on selected existing synapses. Actual whole-graph replay, including hard spikes, accepted 13,867 changed CSR entries. Topology, signs, unselected edges and the decoder are unchanged. Training KL fell from 0.1429181 to 0.1425697; validation KL fell from 0.1743336 to 0.1733302. The source scenes were already seen by the decoder, and the two short development games show no gameplay gain.

This is an executable link from Laya teaching targets through the fixed decoder into biological-edge magnitude optimization. It is not full recurrent backpropagation or evidence that the brain's entire connectivity has been independently trained. The local derivative ignores indirect feedback and changes in spike timing; only exact forward rollout determines acceptance. M4 completion still requires longer trajectories, multiple seeds, stability analysis, separately reserved evaluation scenes and matched controls. M2 physiological validation remains unresolved. The new [research desk design and methods notebook](LAB_DESIGN_RESEARCH.md) documents source comparisons, exact scope, failed searches and next experiments.

### Automatic synaptic learning prototype (October 9, 2026)

The eight engineering follow-ups now have an implemented path: longer chronological training, reserved paired benchmarks, measured forward/backward curriculum feedback, an encoder ablation, full recurrent perturbation measurements, persistent population-state replay, named anatomy/path exploration, and an automatic candidate/promotion loop. See [Automatic learning](AUTOMATIC_LEARNING.md).

The first complete cycle trained real graph strengths but failed its promotion gate: training KL decreased, validation KL increased, and one task's native return regressed. The approved checkpoint is therefore unchanged. This completes the mechanism's integration, not the research objective of a reliably better Doom policy. Useful retreat, broader task generalization, statistically credible gains, biologically justified visual mapping and exact recurrent gradient training remain open research questions. Those limits are separated from implemented controls and preserved negative results in the [validation record](VALIDATION.md).
