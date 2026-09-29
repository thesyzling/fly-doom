# Initial infrastructure validation — September 26, 2026

- Environment: Windows, Python 3.13.11, ViZDoom 1.3.1; exact package versions are in `requirements.lock.txt`.
- `python -m pytest -q`: 9 tests passed initially. Checks covered connection direction, regional row aggregation, isolated neurons, self-connections, large integer IDs, invalid inputs, and corrupted checksums.
- `python -m flydoom.doom_smoke --episodes 3`: three random-policy episodes completed, with 240 × 320 RGB observations. The JSON report is in `runs/doom-smoke.json`.
- Turkish characters in the absolute directory path caused an error in ViZDoom's native layer. Relative paths to files in the project environment resolved the issue on this machine.

The real data had not been downloaded during the initial infrastructure check. The follow-up below completed full graph import and annotation matching.

## Real data validation — September 26, 2026

The `download`, `prepare`, and `annotate` commands completed successfully.

| Measurement | Result |
|---|---:|
| Unique neurons | 139,255 |
| Source connection table rows | 16,847,997 |
| Directed neuron pairs after regional aggregation | 15,091,983 |
| Total synaptic contacts | 54,492,922 |
| Neurons matched to annotations | 139,255 |
| Annotation rows outside the graph | 0 |
| Sparse matrix data and index arrays in RAM | 173.25 MiB |
| Nonzero matrix entries | 0.077826% |
| Neurons with self-connections in the source | 0 |

Both Zenodo files matched their published MD5 values. Source and output SHA-256 values are in `data/processed/fafb783/manifest.json`. The prepared matrix was reloaded from disk, and its shape, positive counts, and total synapse count were checked against the raw file.

In the real data, the root ID list uses `uint64`, while the connection table uses `int64`. NumPy `searchsorted` with these mixed types caused precision loss for large IDs. IDs are now converted losslessly to `uint64` before matching. A regression test with adjacent large IDs was added.

Annotation source: [flywire_annotations v2.1.0](https://github.com/flyconnectome/flywire_annotations/tree/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0), commit `ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0`. The downloaded source SHA-256 is `30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6`. Annotations were joined by exact neuron identity. The output was read back from disk to verify that its row order matches the graph's neuron IDs exactly.

Complete matching does not mean every field is known: 28,165 neurons have an empty `cell_type` field and 601 have an empty `top_nt` field. Source columns are preserved, and missing values were not filled with guesses. `top_nt` labels are predictions. No excitatory or inhibitory signs have been assigned to connections at this stage.

`python -m pytest -q`: **13 tests passed** after adding ID-type, negative-ID, and annotation-matching checks. These results concern data preparation correctness. Neuron dynamics, visual signal transmission, training, and biological validity have not yet been validated.

## Experimental simulation validation — September 27, 2026

`python -m pytest -q`: **29 tests passed** after adding the LIF kernel and probe. New checks cover analytical constant-drive and synaptic-decay solutions, time-step refinement below threshold, presynaptic sign orientation, excitation and inhibition, transmission delay, complete refractory intervals, state reset, invalid parameters, and connected/disconnected diagnostic comparisons.

`python -m flydoom.brain_probe` completed four 200 ms simulations on the verified 139,255-neuron graph. No-input and disconnected controls behaved as expected. All runs remained numerically finite. The photoreceptor condition changed downstream voltages without generating descending spikes; a separate direct excitatory control produced 1,290 descending spikes. Excessive negative voltage excursions mean that physiological calibration has not passed. These are diagnostic results, not evidence of vision or learning.

Full results, parameter assumptions, and limitations are in [SIMULATION.md](SIMULATION.md). Machine-readable output is `runs/brain-probe/report.json`, with source hashes and separate per-condition spike-count artifacts. Persistent network storage was approximately 119.9 MiB and connected CPU steps averaged approximately 12.5–13.0 ms for each 0.5 ms of simulated time. These measurements exclude load peaks and reporting overhead.

M2 remains incomplete. No Doom visual encoder, neural action controller, or training was added in this stage.

## Recorded experiment viewer — September 27, 2026

Added a local browser viewer with a WebGL point map, four selectable conditions, cell filters, neuron inspection, superclass totals, and a cumulative spike trace. It reads actual saved spike counts and source anchor coordinates, with voxel anisotropy corrected before normalization. It does not animate unrecorded neuron activity or infer mental functions.

`python -m pytest -q -p no:cacheprovider --basetemp <fresh temporary directory>`: **33 tests passed**. Viewer checks cover anisotropic coordinates, exact large neuron IDs, spike totals and binary buffers, modified artifact rejection, incomplete runs, and allowed HTTP routes.

An isolated headless Chrome check loaded all 139,255 points with WebGL error code 0 and no JavaScript exceptions. Switching conditions showed the recorded values: 42,260 photoreceptor-condition spikes with zero descending spikes, and 40,033 excitatory-control spikes with 1,290 descending spikes. Filtering and a 390-pixel mobile layout were exercised; no horizontal overflow was detected. These UI checks do not add evidence of biological validity.

## Temporal playback — September 27, 2026

New probes now capture per-neuron interval spike counts and voltage snapshots. A fresh full-graph run in `runs/replay-check` preserved the previous four conditions' total spike counts. Each condition contains 21 recorded samples, including time zero, at the default 0.5 ms integration step and 10 ms recording interval.

**36 tests passed.** Added checks confirm that recording does not change dynamics, temporal bins sum to neuron totals, initial and final voltages match the model state, a partial final interval is retained, binary replay ordering is correct, inconsistent counts are rejected, and legacy recordings remain readable.

Headless Chrome checks exercised automatic playback, pause, slider seeking, spike/voltage modes, condition switching, and mobile layout. At the 50 ms photoreceptor frame, the viewer read 8,452 interval spikes and 14,349 neurons with a voltage deviation greater than 0.01 mV. Seeking to time zero restored resting voltages; the no-input replay remained at rest. WebGL reported no errors and no JavaScript exceptions were observed. No activity is interpolated or synthesized for display. The physiological and visual-pathway limitations are unchanged.

## Visible Doom demo — September 27, 2026

Added optional game-window rendering, per-tic viewing pace, and action logging to the existing random baseline. No new dependency was required. Headless seeds 42, 43, and 44 retained returns of -340, -355, and 95, with 75, 75, and 2 decisions respectively. A native visible run with seed 44 completed successfully with a 480 × 640 RGB observation, return 95, and 2 decisions; its log showed ATTACK followed by WAIT. Reports are in `runs/doom-visible-change-headless.json` and `runs/doom-visible-check.json`.

**36 tests passed** after this change. The visible demo remains independent of the neural simulation and does not demonstrate learned behavior or brain control of Doom.

## Experimental neural game bridge — September 27, 2026

`flydoom.brain_doom` now reads Doom pixels, stimulates positive visual projection cells, computes the existing full-graph LIF dynamics, and selects buttons from mean descending spike rates. Input and output assignments are deterministic engineering conventions, and the input bypasses the retina. See [BRIDGE.md](BRIDGE.md) for the exact rules. No training or random-action fallback was added.

Real-data and native-engine checks used seed 42, a 50 ms neural window, 40 mV-equivalent maximum image drive, and 4 game tics per decision:

| Run directory under `runs/` | Decisions | Total neural spikes | Descending spikes | Game return | Outcome |
|---|---:|---:|---:|---:|---|
| `bridge-connected-check` | 6 | 59,471 | 3,133 | -29 | LEFT and ATTACK selected |
| `bridge-disconnected-check` | 6 | 53,956 | 0 | -24 | WAIT throughout |
| `bridge-zero-input-check` | 6 | 0 | 0 | -24 | WAIT throughout |
| `bridge-visible-check` | 24 | 243,202 | 12,765 | -121 | LEFT, RIGHT, and ATTACK selected |

All four trials reached their explicit decision limits; none constitutes a completed-game success evaluation. The first six visible decisions exactly matched the headless connected trace, including image features, neural counts, action vectors, rewards, and voltage extrema, excluding wall-clock compute time. Connected neural computation took approximately 1.3–1.4 seconds per decision on this CPU. The game holds its frame during computation.

The connected model remained finite in these bounded runs but reached a minimum post-step voltage of approximately **-453.7 mV**, which is not physiologically plausible. This was recorded without clipping. These checks establish software integration and dependence on synaptic transmission, not biological validity, target recognition, learning, or an advantage from the fly graph. M2 remains incomplete.

**44 tests passed** after adding the bridge. New checks cover spatial pixel averaging, normalization by output population size, silence and tie handling, input/transmission ablations on a known circuit, persistent neural state and episode reset, exact ID-based mappings, disjoint inputs and outputs, button-order handling, decision limits, artifact checksums, overwrite protection, and interruption cleanup with partial reports.

## Local teacher and additional neuron training setup - September 29, 2026

Implemented the staged pipeline described in [LEARNING.md](LEARNING.md): keyboard demonstrations, episode-separated datasets, full-graph feature extraction, supervised adaptation of the local Laya decision head, a teacher validation gate, and distillation into 64 additional LIF-like cells. Original connectome edges are frozen during learning. This adds training infrastructure; no human demonstration dataset or trained Doom policy has been produced yet.

The global synaptic gain sweep in `runs/calibration-training-v1/report.json` tested 0.03, 0.01, and 0.003 mV/contact. All passed the specified bounded engineering checks; the first passing candidate, **0.03 mV/contact**, was selected. Its minimum post-step voltage was approximately **-83.26 mV**, and its maximum deviation from rest after 400 ms without input was approximately **0.0000763 mV**. Controls included no input, three distinct spatial patterns, 1,000 ms of sustained white input, recovery, and disconnected transmission. This is an engineering gain selection, not a physiological fit or validation of natural vision. The original bridge defaults remain available for reproducing earlier runs. M2 remains biologically incomplete.

Installed Laya 0.3.21, PyTorch 2.14.0+cu130, Transformers 5.17.0, and their dependencies. The English checkpoint is pinned locally to Hugging Face revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, with per-file SHA-256 hashes. The exact environment is recorded in `requirements-training.lock.txt`. A lock-file sync dry run in offline mode proposed no changes; package compatibility checks passed for all 46 installed packages.

`runs/learning-setup-check/report.json` records actual offline Laya inference and a decision-head gradient update on the NVIDIA GeForce RTX 3060 Laptop GPU. Peak PyTorch allocation was **2,024.8 MiB** in this small check. A synthetic student update also changed the added group's weights. Synthetic labels were used only to verify computation; these weights were not saved as a gameplay policy.

The native Doom recording smoke test saved five episodes of three decisions each in `runs/learning-record-smoke`. Full-graph replay completed in `runs/learning-prepared-smoke`, producing feature arrays with shapes **9 by 2,610**, **3 by 2,610**, and **3 by 2,610** for train, validation, and test. Both artifacts are explicitly marked `random_smoke`. The training loader rejected them as intended. A pygame dummy-display render of a recorded Doom frame confirmed the recording window layout and instruction fit; `runs/learning-setup-check/recording-preview.png` contains that preview. This rendering check did not exercise physical keyboard input.

**54 tests passed** in the full suite. Added checks cover split assignment, observation fields, rejection of random labels, altered artifacts and duplicate episode seeds, teacher shortcut gates, encoder freezing, teacher-head restoration, spiking surrogate gradients, student learning and checkpoint reload, recording cancellation cleanup, and bounded student play with a small neural circuit and game double. The student playback check verifies button order and that Laya is never loaded. Neither the synthetic tests nor the short native recording establish learned game skill. Actual human recording, teacher acceptance, student gameplay evaluation, and comparisons with learned baselines remain pending.

## First human run and recording corrections - September 29, 2026

The human run `runs/learning-run-20260929-023438-852190` completed ten episodes and saved 205 decisions: 130 training, 38 validation, and 37 test. The original teacher completed five epochs. Its selected validation accuracy was 26/38 (68.4%), below the 32/38 (84.2%) constant-WAIT baseline; image shuffling left accuracy unchanged. Balanced validation accuracy was 20.3%. Validation labels contained 32 WAIT, three LEFT, one RIGHT, and two ATTACK examples. The gate correctly rejected the teacher and no student was trained. The original model files and recordings remain available as experiment evidence.

Code inspection identified a missed-input path: polling only the current held-key state once per four game tics can miss a key pressed and released between polls. The old recording did not log keyboard events, so the number of missed taps in that run is unknown. The revised recorder pumps events approximately 120 times per second, latches brief taps for one decision, retains held keys, pauses on focus loss, and clears stale input before ENTER resumes. The displayed frame now precedes the interval in which its action is collected. The screen includes an unrecorded keyboard check, the captured action, action counts for the current split, and missing coverage requirements. Saved episodes include press counts and recovered-tap flags.

The pipeline now collects at least 30 episodes and can extend to 100 based only on predeclared training/validation class coverage. Its fresh default seeds begin at 30000. Insufficient coverage stops the pipeline before full-graph replay or loading Laya. The separate teacher command also checks coverage before loading the model. The teacher gate itself was not weakened. Training now uses inverse-frequency class weights computed from training labels only, and teacher checkpoint selection uses mean per-class validation negative log likelihood. Confusion matrices, prediction counts, and explicit rejection reasons were added.

Visual observations now retain 256 brightness levels and include a fixed, pixel-derived blue-region description; see [LEARNING.md](LEARNING.md) for the exact heuristic and its limitations. `runs/learning-input-fix-audit/quality.json` reports 29 old versus 94 revised distinct training observations, and six versus 28 distinct validation observations on the original frames. This is a representation-diversity check, not evidence of improved prediction. The image-shuffle control now replaces every visual field together while keeping the previous action fixed. On the first recorded training episode, actual local Laya tokenization produced 216-220 tokens per revised observation, with all four choices present and no truncation at the 512-token limit.

A native-engine random smoke test again saved five episodes of three decisions in `runs/learning-input-fix-smoke`. A dummy-display render checked the new recording panel. **62 tests passed** after the corrections, including short press/release recovery through the recorder into both applied game buttons and saved arrays, held-key aliases, focus pause and resume, clearing the pre-recording keyboard check, class-weight contributions, test-label independence of coverage, complete visual shuffling, and early rejection before expensive work. Automated input events do not substitute for a physical keyboard check by the player. A fresh human recording and validation run are still needed to measure learning improvement; no new successful teacher or student is claimed.

## Resume interrupted recording - September 29, 2026

The revised human recording `runs/learning-run-20260929-025831-927335` stopped with 49 complete episodes; its unfinished 50th episode was discarded as documented. Training action counts were [249, 47, 40, 30] and validation counts were [85, 29, 2, 10] in WAIT/LEFT/RIGHT/ATTACK order. Validation RIGHT coverage remained below five, so neural replay and training had not started.

Added `learning resume --run-dir <existing run>` to append to this stopped recording and continue the pipeline after coverage passes. Resume verifies original file hashes, contiguous numbering, seeds, split assignments, input format, and timing; unlisted episode files are not overwritten. The original minimum/maximum episode limits are inherited. Completed episode data is preserved, while the manifest records resume history. Pipelines that have already started preparation or training cannot append data through this command.

**64 tests passed**, including append/resume preservation of episode bytes and split assignments, rejection of changed seeds and conflicting files before modifying the manifest, and inheritance of options from an older recording without an explicit saved seed field. The real human recording was inspected read-only; it was not automatically restarted or used to train a model during this check.

## Balanced memorization and full teacher retraining - September 29, 2026

The resumed human recording completed at 69 episodes with 872 decisions. Coverage passed with training counts [349, 66, 54, 42] and validation counts [116, 41, 10, 14]. Thirty brief taps were retained across all recorded splits. The original revised five-epoch teacher predicted WAIT for every validation example: 64.1% accuracy and 25% balanced accuracy. Its rejected checkpoint remains in the original run directory.

The training-only probe in `runs/laya-balanced-diagnosis-v1/report.json` selected four distinct observations per action. It progressed from 7/16 correct predictions to **16/16**, reaching negative log likelihood **0.01036** after 70 balanced updates. No validation/test examples were used by the probe and no gameplay checkpoint was saved. The head changed; frozen encoder parameters had no gradients. Cached and original logits matched with measured maximum absolute error **0.0** on the probe. Fourteen exact training observations had conflicting human labels; a deterministic predictor receiving only the current recorded observation has an empirical training accuracy ceiling of approximately 93.35% on this dataset. This ceiling is a data diagnostic, not an expected generalization score.

A single-example gradient audit illustrates how the former per-update clipping could suppress the intended class weighting: weighted gradient norms before clipping were approximately 0.237 for a sampled WAIT, 12.50 for LEFT, 4.36 for RIGHT, and 190.23 for ATTACK. These are illustrative selected examples, not a replay of every old two-example batch or proof of a sole causal explanation. The revised optimizer accumulates gradients from 16 examples, four per class, before clipping once. The learning rate changed from 0.00002 to 0.0001, and the default full-training budget increased to 15 epochs. The memorization probe disables dropout; full teacher training retains head dropout. These changes were not individually isolated in a controlled ablation.

The full 15-epoch retraining completed in `runs/laya-balanced-teacher-v2/`. Its selected epoch was 15; the saved natural-distribution validation metrics are:

| Measure | Old constant-WAIT teacher | Balanced-update teacher |
|---|---:|---:|
| Overall accuracy | 64.09% | 59.67% |
| Balanced accuracy | 25.00% | 63.95% |
| Accuracy with shuffled images | 63.54% | 44.75% |
| Predicted WAIT / LEFT / RIGHT / ATTACK counts | 181 / 0 / 0 / 0 | 67 / 53 / 21 / 40 |
| Correct LEFT / RIGHT / ATTACK | 0 / 0 / 0 | 34 / 3 / 13 |

The teacher now distinguishes active actions and depends on visual information in this control, but its overall accuracy still falls below the constant-WAIT baseline plus the unchanged five-percentage-point margin. The gate rejected distillation and no student was trained. A separate validation-only prior adjustment probe in `runs/laya-balanced-diagnosis-v1/prior_probe.json` examined powers 0, 0.5, and 1 of training action frequencies. The half correction produced 66.30% ordinary and 57.10% balanced accuracy; the full correction produced 69.61% and 36.85%, respectively. Neither passes all gates. This post-processing was not adopted in production. Its independently seeded shuffle differs from the main teacher report and should not be compared as a paired control to that report.

**67 tests passed**, including balanced class counts per optimizer update, distinct balanced probe selection, cached-logit agreement with the actual upstream Laya model class on a small encoder, and accumulated-gradient agreement with a full batch when dropout is disabled. Full retraining reused the existing prepared features and did not rerun the fly simulation or request more human recordings. Reported test metrics are exploratory because this dataset has been evaluated in earlier experiments; fresh episodes will be needed for a final gameplay claim. The current limitation is useful action timing/generalization, not failure of the gradient path. Temporal context is a candidate next experiment, not an established solution.

## Temporal context and autonomous teacher evaluation - September 30, 2026

Created `runs/temporal-prepared-v1/` from the existing 69-episode prepared dataset. Each teacher observation includes three strictly past visual summaries and applied actions, completed-decision count, and age of the last active action and shot. Current labels and future frames are excluded, and history resets per episode. Neural feature files were copied byte-for-byte with hashes checked; no full-graph replay or new demonstration recording was performed. All 511 training and 181 validation observations fit the input budget without truncation: 292-387 training tokens and 293-385 validation tokens. The loader now rejects overflow rather than silently truncating input.

A training-derived lookup using only decision index and previous action scored **71.82%** on validation. For temporal teachers, this is included in the strongest-baseline comparison, raising the required ordinary accuracy to at least **76.82%** while retaining the original balanced-accuracy and image-shuffle criteria. Visual shuffling replaces the current and past visual fields together and preserves action history and timing.

The first temporal run (`runs/laya-temporal-teacher-v1/`) completed 15 epochs and selected epoch 13 by balanced validation loss. A second run (`runs/laya-temporal-teacher-v2/`) started from that selected head, reset AdamW state, and completed 15 additional epochs, selecting additional epoch 13. Both used identical data/splits and the verified Laya base; neither evaluated the dataset's test split.

| Validation measure | Single-frame, 15 epochs | Temporal, 15 epochs | Temporal, 15 additional epochs |
|---|---:|---:|---:|
| Ordinary accuracy | 59.67% | 69.61% | 71.82% |
| Balanced accuracy | 63.95% | 71.23% | 74.69% |
| Image-shuffled accuracy | 44.75% | 46.96% | 44.20% |
| Correct LEFT / RIGHT / ATTACK | 34 / 3 / 13 | 39 / 5 / 11 | 38 / 7 / 10 |

Both temporal teachers failed the strengthened ordinary-accuracy condition, so no student was trained. The validation-only prior probe at `runs/temporal-prior-probe-v1/report.json` checked the original temporal head with powers 0, 0.5, and 1 of training action frequencies. Half adjustment reached 74.03% ordinary and 61.31% balanced accuracy; full adjustment reached 77.35% ordinary and 46.33% balanced accuracy. Neither passed all gates, and these adjustments were not adopted.

Added a separate experimental teacher-play command to measure actual autonomous behavior without changing or bypassing the distillation gate. It loads Laya directly, reconstructs history from its own actions, and does not load the fly graph. Only image-derived observations and past actions enter the policy. Kill count is queried after play solely as an outcome metric. The timing control uses training labels only; the random control chooses uniformly among the four actions.

Paired native-engine runs used seeds **50000-50005**, disjoint from the demonstration seeds, with four tics per decision and a 75-decision cap in the basic scenario. Every episode reached a game termination rather than the external decision limit; a game termination can be a timeout, so kills are reported separately.

| Policy | Target kills / 6 episodes | Mean return | Report directory under `runs/` |
|---|---:|---:|---|
| Single-frame Laya, 15 epochs | 0/6 | -305.00 | `single-frame-teacher-game-eval-v1` |
| Temporal Laya, 15 epochs | 1/6 | -254.83 | `temporal-teacher-matched-budget-game-eval-v1` |
| Temporal Laya, additional training | 6/6 | 29.00 | `temporal-teacher-game-eval-v1` |
| Timing-only control | 5/6 | -29.17 | `temporal-teacher-game-eval-v1` |
| Random control | 6/6 | 20.83 | `temporal-teacher-game-eval-v1` |

The continued temporal model's returns were [39, 39, 39, 39, 39, -21]. A negative return can still include a target kill because waiting and actions incur costs. Random also killed all six targets, so a 6/6 kill count alone is not evidence of a learned advantage. These small pilot results suggest an improvement over the earlier learned controllers on these starts, but do not establish broad superiority or isolate history from additional optimization. The six seeds are now development evaluation data and should not be reused as a pristine final benchmark.

**77 tests passed at this stage.** New checks cover strict exclusion of current/future labels, episode history reset, agreement between offline and online history construction, preserving timing/action hints during visual shuffling, token-overflow rejection, byte-preserving feature augmentation, warm-start provenance rejection and test-skipping behavior, autonomous use of the model's own past buttons, terminal action limits, and exact original observations for the single-frame comparison. The experimental visible-play command is documented in the README. At this stage the 64 added neurons were still untrained; the teacher gameplay results alone do not demonstrate fly-brain learning.

## Explicit experimental student transfer (September 30, 2026)

The user requested connecting the demonstrated teacher behavior to the fly network. Added an explicit `--experimental-teacher` student-training option for a completed, verified teacher whose offline gate failed. The default automatic pipeline still refuses this transfer. Training and gameplay reports retain `teacher_accepted: false`; the original teacher report and head are unchanged. `--skip-test` excludes development test metrics.

`runs/fly-student-experimental-v2/` completed 100 epochs using the existing temporal dataset and teacher v2. The selected checkpoint is epoch 12, chosen by ordinary validation NLL (0.73272). It trains the additional 64-cell group's input, recurrent, and action-readout weights; original fly weights remain frozen. The student receives the 2,606 descending neural features and four previous-action indicators, without the teacher's direct visual descriptions or explicit temporal summaries. Training used 511 human-recorded states and Laya probability targets; the held-out validation set contains 181 decisions. Test metrics were not evaluated.

| Validation condition | Ordinary accuracy | Balanced accuracy |
|---|---:|---:|
| Trained student with neural features | 71.27% | 52.76% |
| Same student, neural features zeroed | 64.09% | 25.00% |

The zero-feature condition predicts WAIT for all validation samples. This is evidence that the trained policy uses neural features, not that biological topology is uniquely useful. No random-graph or direct-pixel learned baseline was evaluated here.

Native-engine gameplay used **51000–51002**, disjoint from the training recordings and the earlier teacher pilot. There was no training during play and no Laya model loaded. The calibrated full graph contained 139,255 neurons; only its added readout was learned. Both conditions used the same saved student weights, seeds, four game tics per decision, and 75-decision bound.

| Condition | Kills / 3 episodes | Returns | Mean return | Report directory under `runs/` |
|---|---:|---|---:|---|
| Connected frozen graph + trained student | 2/3 | -315, 39, 31 | -81.67 | `fly-student-connected-eval-v1` |
| Synaptic transmission disabled + same student | 0/3 | -300, -300, -300 | -300.00 | `fly-student-disconnected-eval-v1` |

The connected trials took 75, 16, and 18 decisions. The disconnected trials produced 225 WAIT decisions in total and exactly zero descending-feature norms. Input cells still spike when directly stimulated under the disconnected control. All episodes ended through the game, including the failed timeout episodes. Connected neural computation averaged 1.302 seconds per decision; the minimum observed voltage was -66.065 mV, within the engineering bound. Kill count is read only after an episode for evaluation and never enters the policy. Traces record brain spike counts, descending spikes, feature norms, neural computation time, actions, and probabilities.

This three-start pilot demonstrates an operational brain-to-student-to-buttons route and some target kills without Laya inference. It does not establish robust game skill, biological realism, a topology-specific advantage, or improvement over the direct teacher; the latter was evaluated on different seeds. These starts are now development evaluation data.

The first attempt, `runs/fly-student-experimental-v1/`, failed at epoch 54 during atomic report replacement with Windows `PermissionError`. Its failed status and partial artifacts were retained. Report replacement now retries brief permission failures for a bounded interval without truncating the previous report. The exact process that held the file was not identified. The second training attempt completed and its model, implementation, and teacher-report checksums match.

**80 tests passed.** New or extended checks verify that experimental transfer preserves the teacher's rejection and original report, default rejection remains enforced, test skipping works, saved student play never loads Laya, kill/action metrics are recorded, and transient/permanent replacement failures preserve complete reports. Native gameplay is recorded separately from these automated tests.
