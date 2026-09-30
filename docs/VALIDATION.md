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

## Live synchronized neural observer (September 30, 2026)

Added `flydoom.live_brain` and a local browser screen, leaving `student.py`, the trained checkpoint, the simulator, and the recorded-experiment viewer unchanged. The observer verifies the saved student's implementation/model hashes and the calibration/graph mapping, then runs the same image → graph → student → buttons computation. Read-only forward hooks capture actual added-cell drive and mean spike activity. Exact root IDs remain strings in JSON and JavaScript. CSR target rows expose incoming edges; CSC source columns expose outgoing edges. A bounded strongest-edge selection keeps the diagram readable while any fly cell can be searched by exact ID.

The browser pairs the pre-action game observation with that decision's neural response and probabilities; reward is measured after the action. A sequence identifier keeps asynchronous neuron inspection tied to the displayed snapshot. Full-neuron observations are retained in a 24-snapshot memory buffer, and history resets visually at episode boundaries. Pause and single-step controls apply between decisions. Final reports explicitly retain the absence of Laya inference and the teacher's rejected status.

`runs/live-observer-validation-v1/` ran native Doom on development seed 51001 while driven from the browser. It produced **16 decisions, one target kill, and return 39**. Comparison with episode two of `runs/fly-student-connected-eval-v1/` found identical actions and returns and a **maximum absolute probability difference of 0.0**. `parity.json` records this result and confirms the student implementation hash is unchanged. This is an observer-equivalence check, not an independent gameplay-success estimate.

`runs/live-observer-ui-v2/` is a separate five-decision bounded UI check on the same seed. Its zero kills and return -20 reflect the intentional decision limit, not a completed gameplay trial. The first browser automation wait used a 30-second deadline that was too short for the full episode under browser-test load; the game subsequently completed normally. The bounded UI check then exercised three single steps followed by Run. The page remained paused between steps, disabled run controls at completion, rendered all 64 cells, and exposed learned action and real fly-neuron connections.

Headless Edge checks are saved in `runs/live-ui-check-report.json` and `runs/live-ui-final-check-report.json`, with desktop/mobile screenshots under `runs/live-dashboard-*.png`. The mobile page measured 390 pixels for both viewport and document width. Direct pointer tests followed an exact graph node ID and verified wheel zoom, drag pan, and reset. An initial pointer probe used off-viewport coordinates after browser emulation reset; rerunning with an explicit desktop viewport passed. The final page check reported no JavaScript exceptions or resource errors. A prior missing favicon warning was removed by supplying a local data-URL icon.

**86 tests passed.** The six new tests cover exact IDs and signed connection direction, unmodified forward outputs/weights under telemetry hooks, arithmetic action-logit contributions, pause/step and observation timing in the game loop, HTTP control origin/route bounds, and rejection of incomplete checkpoints before graph loading. The browser diagram is schematic, and the live activity is simulated; no claim of anatomical morphology, biological calibration, or causal action explanation is added.

## Balanced readout refinement and new starts (September 30, 2026)

Added `flydoom.student_refine` as a separate training entry point, preserving `student.py`, the original checkpoint, and the live observer. The experiment warm-starts the existing 64-cell group, freezes its normalization, resets AdamW at learning rate 0.0003, and gives each update eight human training examples from each of the four actions. The distillation loss remains teacher-probability KL plus 0.25 human cross-entropy. Validation selection now minimizes class-balanced NLL, with the untouched parent eligible as epoch zero. This compares complete training configurations; sampling is not the only changed optimization setting.

`runs/fly-student-balanced-v1/` completed 100 epochs, selecting epoch 5. Laya probabilities were computed on the 511 training observations only and saved with a checksum. The teacher's failed gate remains recorded; no game rewards, evaluation seeds, or test metrics selected the checkpoint. The original 181-example validation split was reused as development data.

| Validation metric | Original student | Balanced refinement |
|---|---:|---:|
| Ordinary accuracy | 71.27% | 69.06% |
| Balanced accuracy | 52.76% | 57.01% |
| Ordinary NLL | 0.73272 | 0.78725 |
| Balanced NLL | 1.08832 | 1.03528 |
| Correct ATTACK examples | 4/14 | 7/14 |

The new checkpoint predicts WAIT on all validation examples when neural features are zeroed (64.09% ordinary / 25.00% balanced accuracy). Shuffling the neural-feature block while retaining previous-action indicators gives 44.75% ordinary / 29.50% balanced accuracy. These controls measure dependence on the neural representation, not a topology-specific advantage. `parameter_changes.json` records changes in the input, recurrent, and action-readout parameters and verifies unchanged normalization and student implementation.

**90 tests passed.** Four added tests check reproducible balanced batches with rare classes, actual weight changes while parent files and normalization remain unchanged, retention of epoch-zero weights when validation does not improve, training-only teacher queries with test metrics skipped, and early rejection of a modified teacher.

Paired native Doom evaluation used six previously untested starts, **54000–54005**, disjoint from all human recording seeds and the earlier gameplay pilots. The student checkpoints were selected using offline validation before candidate gameplay. All policies used four game tics per decision and a 75-decision cap; every evaluated episode ended in the game, not at the external cap. Failed episodes timed out. The original student evaluation began before the candidate was trained; its observed results make this a development comparison rather than a pristine final benchmark.

| Policy | Target kills / 6 | Mean return | Report directory under `runs/` |
|---|---:|---:|---|
| Original student | 2/6 | -213.83 | `student-generalization-baseline-v1` |
| Balanced student | 2/6 | -216.50 | `student-generalization-balanced-v1` |
| Balanced student, transmission disabled | 0/6 | -300.00 | `student-generalization-disconnected-v1` |
| Direct temporal Laya | 5/6 | -77.67 | `student-generalization-controls-v1` |
| Training-derived timing lookup | 3/6 | -141.50 | `student-generalization-controls-v1` |
| Uniform random actions | 5/6 | -55.33 | `student-generalization-controls-v1` |

Original student returns were [-315, -340, -340, 51, -370, 31]. Candidate returns were [-340, -320, -320, 10, -360, 31]. Both students succeeded only on seeds 54003 and 54005. The longest WAIT streak fell from 42 decisions to 20; aggregate WAIT usage fell from 258/331 decisions (77.95%) to 250/340 (73.53%). ATTACK counts were 46 and 45, respectively. These decision-weighted percentages include longer failed episodes; they should not be treated as independent samples or a game-level success measure.

`runs/student-refinement-comparison-v1/report.json` verifies episode seeds, trace lengths, action counts, rewards, checkpoint hashes, and separation from demonstration seeds. It compares target kills first, then mean return for tied kill counts. The candidate does not surpass the parent under that rule, so **the original student remains the default**. Both the new weights and the negative result are retained. The disconnected candidate produced 450 WAIT decisions and no kills; this supports dependence on neural signals but does not establish an advantage of the biological topology over other feature generators.

This experiment reduced prolonged waiting without improving task success. It does not support a learned advantage over random actions on this basic scenario. Further work should collect additional policy-relevant states and evaluate recovery behavior; merely increasing training epochs on the same human examples is not established as a solution. No correction-data collection or reward-based training is claimed in this iteration.

## Teacher suggestions on student trajectories (September 30, 2026)

Implemented `correction_data.py` and `correction_training.py` without changing the hash-locked student runtime or simulator. The original student controls every recorded button; the frozen Laya teacher is queried afterward on causal observations reconstructed from those frames and actual past actions. Policy recordings have a separate schema from human demonstrations. Reward and kill count are stored only as outcomes, never as teacher input or training targets. The teacher's failed acceptance gate remains unchanged.

`student-correction-observations-v1` contains four training episodes (56000-56003) and two validation episodes (56004-56005). Since all four training episodes succeeded, a second collection deliberately included previously failed development starts 54000-54002 as training and 54003 as validation. Those earlier starts are no longer independent evaluation data for this candidate. Byte-preserving merge output `student-correction-merged-v1` contains **316 training and 106 validation decisions across ten episodes**. Reserved evaluation seeds 57000-57005 were declared before collection and are disjoint from both human and correction episode seeds.

Different seeds produced identical images and trajectories: 56000, 56001, 56003, and 56004 repeat the same 18-decision trajectory. Before checkpoint selection, exact feature-vector plus causal teacher-state matches against either training source removed 18 correction validation rows, leaving **88**. Near duplicates, shared scenarios, and the reused human validation split remain limitations. The post-selection initial-frame audit found that reserved starts 57003-57005 also exactly repeat correction-training opening frames. The other three opening frames did not match correction training; this does not certify their full trajectories or independence from human examples.

`student-correction-labels-v1` saves frozen teacher probabilities and checksums. The 511 existing human training probabilities were reused from the verified balanced-training cache; only probabilities were reused, not that candidate's weights. On correction training rows, Laya disagrees with **193/316** recorded actions, including **168 WAIT** actions. Teacher argmax counts are [95 WAIT, 35 LEFT, 158 RIGHT, 28 ATTACK], versus the student's [257, 27, 8, 24]. These disagreements are not verified errors.

Each update uses 32 human and 32 correction observations. The loss gives half its weight to human distillation (KL plus 0.25 human cross-entropy) and half to correction KL, with a three-to-one weight for disagreement versus agreement. Parent mean and scale stay fixed; AdamW resets. Checkpoint selection minimizes correction-validation KL plus 0.25 human-validation NLL, subject to both human accuracies remaining within five percentage points of the parent. Epoch zero remains eligible. No human test metrics or reserved gameplay outcomes select the model.

The first 100-epoch run at learning rate 0.0003, `fly-student-corrected-v1`, retained the unchanged parent at epoch zero because no updated checkpoint met both constraints. The second run at 0.00001, `fly-student-corrected-v2`, selected **epoch 11 of 100** using the same development validation rule. These are two optimization trials, not a single predeclared learning rate.

| Validation metric | Parent | Selected correction candidate |
|---|---:|---:|
| Human ordinary accuracy | 71.27% | 66.85% |
| Human balanced accuracy | 52.76% | 51.43% |
| Human NLL | 0.73272 | 0.79419 |
| Correction KL (88 retained rows) | 0.84144 | 0.64391 |
| Correction teacher argmax agreement | 34.09% | 35.23% |

`parameter_changes.json` verifies changes in input, recurrent, and readout weights, unchanged mean/scale, and unchanged parent, teacher report, and student implementation hashes. Only the added 64-cell group trains. The selected model's SHA-256 is `7aed99a3f699b282bb611ecb26404a79e950103bf8fb75bc2c2261d055f70e20`.

**100 tests passed.** Ten correction tests cover causal actual-action history, exclusion of injected reward/future fields even when checksums are updated, early seed-overlap rejection, immutable teacher queries, actual parameter changes with parent/normalization preserved, disagreement weighting, modified-target rejection, byte-preserving merges and duplicate-seed rejection, parent retention under human regression, and validation deduplication against both training sources.

The selected candidate and parent were compared in native Doom on **57000-57005**, with four tics per decision and a 75-decision cap. Every episode ended through the game. The parent's evaluation had completed before candidate selection; its aggregate outcomes were inspected only after the candidate was selected. Baseline, candidate, and disconnected reports are under `student-correction-{baseline,candidate,disconnected}-eval-v1`; direct-policy controls are under `student-correction-controls-v1`.

| Policy | Target kills / 6 | Mean return |
|---|---:|---:|
| Original student | 5/6 | -21.33 |
| Correction candidate | 5/6 | -11.83 |
| Candidate, graph transmission disabled | 0/6 | -300.00 |
| Direct temporal Laya | 6/6 | 22.33 |
| Training-derived timing lookup | 3/6 | -141.50 |
| Uniform random actions | 3/6 | -171.50 |

Parent returns were [63, 71, -355, 31, 31, 31]; candidate returns were [63, 71, -310, 35, 35, 35]. Both fail on 57002. ATTACK counts fell from 22 to 7, while WAIT usage rose from 93/147 (63.27%) to 102/144 (70.83%) and the longest WAIT streak grew from 10 to 22. Thus the candidate gains 9.5 mean reward points without increasing kills or demonstrating recovery. The disconnected candidate made 450 WAIT decisions and had zero descending-feature norms. This supports neural-signal dependence, not a special advantage of biological topology.

`student-correction-comparison-v1/report.json` verifies matching seeds, checkpoint hashes, trace counts, rewards, and action totals. The candidate improves under the stated kills-first, mean-return-second rule, but this six-start result, repeated scenes, reduced human accuracy, and increased waiting do not justify a broad upgrade claim. **The original remains the default**; the corrected model is separately playable in the interactive observer. No game-reward learning or original fly-edge training is claimed. Future recovery work needs more varied observations and a stronger teacher/student representation, rather than assuming more epochs solve the remaining failure.

## Action memory and recovery diagnosis (September 30, 2026)

Recorded correction-v2 on the known failed start 57002 and successful start 57003 in `recovery-diagnosis-observations-v1`, then queried the frozen teacher into `recovery-diagnosis-labels-v1`. These new observations and labels were used for diagnosis only, not memory training. The failed run reproduced return -310 with 75 decisions: WAIT 63, LEFT zero, RIGHT ten, ATTACK two. Laya suggested LEFT 50 times, disagreed with 56 actions overall, and suggested an active button for 49 WAIT decisions. The longest WAIT streak was 22, starting at decision 42. The successful comparison took 17 decisions and returned 35. These are reused development starts, not a fresh success estimate.

`recovery-review-v1/index.html` is a standalone interactive replay with the actual pre-action frame, recorded button/probabilities, teacher advice, causal history, and 64-cell activity. Activity is recomputed from the verified checkpoint and saved neural features, with each probability vector matched to the original recording within absolute tolerance 1e-6. It is not live training. On decision 41 of the failed run, the target is visibly left, the student selects RIGHT, and teacher advice favors LEFT. This observation alone does not establish which internal representation causes the error.

Added `action_memory.py` and `correction_training train --memory`. Seventeen causal inputs encode three past buttons, elapsed decisions, ages since active/shooting actions, and availability flags. The original neural channels and previous-action channels retain their normalization; new memory channels use zero mean and unit scale. New weights start at zero to preserve initial policy behavior. The same 64 engineered cells train; the original fly edges remain fixed, and the cells' internal state still resets every decision. This tests explicit action/timing information, not biological memory or the teacher's richer visual history.

`fly-student-memory-v1` used the same parent, data, cached teacher probabilities, loss, seed 29, learning rate 0.00001, and 100 epochs as correction-v2. It selected epoch 11 with the same human-accuracy constraints. Human ordinary/balanced accuracies remain 66.85%/51.43%. Correction-validation KL changed from 0.64391 to 0.63918. The 181-example human development validation and 88 deduplicated correction-validation rows select the checkpoint; test metrics and game outcomes do not. The teacher's rejection remains recorded.

`memory_effect_audit.json` verifies unchanged parent weights and original normalization channels. Learned memory weights reach maximum absolute value 0.001562. On the failed episode's fixed 75 saved inputs, removing memory changes probabilities by at most 0.01877 but changes zero argmax actions. Both replay conditions yield [63 WAIT, 0 LEFT, 10 RIGHT, 2 ATTACK]. This is an off-policy replay diagnostic, not a new autonomous trajectory or evidence of recovery.

The live observer now supports the separate `action_memory_student_v1` schema and verifies its implementation/mapping hash before graph loading. Its memory panel exposes actual normalized input values and learned edges. Previous-action edges still use the final four columns. Explicit `--seeds` lists allow noncontiguous comparisons; `--disconnected` supplies the graph-transmission control. Old student checkpoints remain supported without modifying `student.py` or `simulation.py`.

**109 tests passed.** Nine new tests cover online/offline causal-history equality, episode reset and real applied buttons in the native loop fixture, unchanged initial parent logits and feature ordering, training new memory weights without changing the parent, separate memory/previous-action edge columns, early implementation-hash rejection, rejection of memory checkpoints by the standard collector before graph loading, and review streak/disagreement summaries. The full suite passed without warnings. JavaScript syntax checks passed.

Headless Edge checks in `memory-review-ui-check.json` verified both recorded episodes, frame loading, 64 cells, slider seeking, replay advancement, episode reset, and a 390-pixel page without horizontal overflow. The live memory candidate displayed all 17 inputs and 64 cells, and clicking `since_active` showed its value and twelve learned edges. No JavaScript exceptions occurred. Screenshots are `recovery-review-desktop.png` and `memory-live-desktop.png`. The first restricted-browser attempts failed because Edge's GPU subprocess could not start; the separate-profile browser test then ran successfully outside that restriction. These UI checks did not send gameplay controls to the evaluation runs.

For the gameplay comparison, 48 starts (58000-58047) were scanned without taking actions, yielding 25 distinct opening images. `memory-evaluation-plan-v1/report.json` selected the first six whose exact opening pixels were absent from 715 distinct known human/student train/validation frames: **58000, 58002, 58004, 58006, 58007, 58010**. Human test frames were not inspected. The selection was made without gameplay outcomes and favors right-side and central positions; it is not a balanced location sample. Different opening pixels do not guarantee different trajectories or broad generalization in this same basic map.

All six native episodes completed under each policy using four game tics per decision and a 75-decision limit. Reports are `memory-{baseline,candidate,disconnected}-eval-v1` and `memory-controls-v1`; `memory-comparison-v1/report.json` checks plan and checkpoint hashes, episode seeds, trace lengths, returns, and action totals.

| Policy | Target kills / 6 | Mean return |
|---|---:|---:|
| Correction-v2 student | 2/6 | -200.17 |
| Student with action memory | 3/6 | -153.17 |
| Memory student, graph transmission disabled | 0/6 | -300.00 |
| Direct temporal Laya | 5/6 | -86.83 |
| Training-derived timing lookup | 1/6 | -253.83 |
| Uniform random actions | 2/6 | -203.17 |

The first five episode returns match: [55, 39, -315, -330, -325]. On 58010, correction-v2 times out after 75 decisions with return -325; the memory candidate kills the target in 34 decisions with return -43. Overall WAIT usage changes from 249/328 (75.91%) to 216/287 (75.26%), while the longest WAIT streak remains 33 decisions. The disconnected candidate makes 450 WAIT decisions with zero neural-feature norms. This is one additional success on six development starts, not a resolved waiting problem or established robust advantage. Direct Laya remains stronger on this pilot. The original checkpoint remains the default, and the memory candidate is separately playable.

A **post-hoc single-start ablation** tested the newly successful start 58010. `fly-student-memory-zeroed-control-v1` copies the selected memory checkpoint and zeros only its 17 memory-input columns; all other tensors and original input columns are verified identical. It does not optimize any weights. The ablated run, `memory-zeroed-eval-v1`, times out after 75 decisions with return -325 and zero kills, versus the active-memory model's one kill and return -43. The first action difference is decision 9: active memory chooses ATTACK, while zeroed memory chooses RIGHT. Before and including that decision, graph spike totals and neural-feature norms match exactly. `memory-ablation-comparison-v1/report.json` verifies model and report hashes, tensor equality, outcomes, and this divergence. This supports a causal contribution of the learned memory inputs in that one selected example; it is not a six-start ablation or evidence that the other failures are solved.
