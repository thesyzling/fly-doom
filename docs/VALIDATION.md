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


## Integrated anatomical map (October 1, 2026)

Replaced the live dashboard layout with a local map and inspector. `live_map.py` exposes all 139,255 neuron annotation anchors in source order, preserves root IDs as strings, converts 4 x 4 x 40 nm voxel coordinates to micrometers, and sends compact per-observation spike counts. These are reference positions, not neuron morphologies or synapse locations. The 64 trained cells, 17 optional memory inputs, four previous-action indicators, and four action outputs have schematic positions. Real model weights supply a bounded overview (161 nonzero links for this memory checkpoint); selecting a node exposes the existing directed neighborhood. No new dependency, training, or checkpoint change was introduced.

**113 tests passed.** The four added tests cover coordinate reconstruction and units, exact root IDs and actual learned weights, memory-column mapping, exact-sequence activity and rejection of expired observations, spike-encoding bounds, and rejection of non-finite coordinates. JavaScript syntax checks and `git diff --check` passed.

The actual native Doom session uses `fly-student-memory-v1`, seed 58010, and output `runs/integrated-map-ui-v1`. Headless Edge loaded all 139,255 points and 89 engineered/input/action nodes. Browser checks exercised actual pointer selection of a fly anchor and added/memory cells, exact-ID search, action inspection, rotation, Shift-pan, zoom/reset, filters, layer toggles, neighborhood view, and three paused single steps. Each displayed full-neuron spike sum matched the decision's recorded total, and the inspector/activity sequences matched the displayed game observation. The three actions, probability vectors, returns, fly spike totals, 64-cell spike counts, and memory inputs were exactly equal to episode six of the earlier `memory-candidate-eval-v1` trace. This is a three-decision observer check, not a new gameplay benchmark.

The 1600-pixel desktop and 390-pixel mobile layouts were inspected. There were no uncaught browser exceptions, no WebGL errors, and no page-level horizontal overflow; the connection table has its own horizontal scroll. Evidence is saved locally as `runs/atlas-browser-check.json`, `runs/atlas-final-visual-check.json`, `runs/atlas-desktop.png`, and `runs/atlas-mobile.png`. The run was left paused after decision three for interactive exploration. Model weights and the hash-locked training/simulation implementations were unchanged.


## Larger adjustable game view (October 1, 2026)

Expanded the default game panel to 480 CSS pixels where space permits, added a 300-640 pixel width slider with browser-local persistence, and added a reversible game-focus view. Run, Pause, and Step once remain available in focus mode. Escape or the return button restores the existing map. Narrow layouts show the game at full available width above the map. Model input images, weights, and inference code were unchanged.

Actual Edge checks at desktop widths 1600, 1280, 1100, 900, 768, 760, and 390 found no page-level horizontal overflow and preserved the 4:3 game aspect ratio. At 1600 x 1000, the initial image measured 432 x 324 pixels with the sidebar scrollbar present; focus mode measured about 947 x 710. Checks covered resizing, saved width after reload, focus/return via button and Escape, one actual paused game step in focus mode, matching the displayed frame to the current state, and a working WebGL map after returning. No uncaught browser exceptions or WebGL errors were observed. Evidence: `runs/game-panel-browser-check.json`, `runs/game-panel-desktop.png`, `runs/game-panel-focus.png`, and `runs/game-panel-mobile.png`. JavaScript syntax and whitespace checks passed. The UI-only change did not require a new training run or a repeat of the numerical model suite.


## Operator feedback collection and separate candidate training (October 1, 2026)

Added exact-sequence human labeling to the live view and an offline supervised trainer for standard and memory students. A paused, completed, or stopped decision can receive a desired action. The immutable observation stores the exact input vector from the actual forward pass, original action probabilities, and the displayed pre-action PNG. Labels and revisions are written to a local manifest with model/calibration/source provenance. Revisions replace active labels; removal retains an audit record while excluding the example from training. Busy, stale, initial-preview, and disconnected-control observations are rejected. The collector never advances the game or updates model weights.

The trainer verifies collections and parent/data provenance, requires separate train and validation episodes, rejects seed overlap and exact duplicate decision records, and removes training-input duplicates from validation. It uses correction cross-entropy plus earlier human cross-entropy and a parent-policy KL replay term. Original normalization remains fixed. A candidate must improve feedback-validation balanced NLL while retaining earlier human ordinary and balanced accuracy within five percentage points; epoch zero remains eligible. Earlier test labels and game rewards are not used. Laya is not queried and no LoRA/RAG component is added. Teacher rejection metadata is retained. Candidate artifacts are separate; there is no automatic deployment or gameplay-success claim.

**120 tests passed.** Added checks cover immutable observations, label revision/removal, stale/busy/initial/disconnected rejection, read-only model behavior, the same-origin HTTP endpoint, altered recording checksums, split leakage, actual parameter learning on synthetic examples for both standard and memory checkpoints, preservation of parent files and normalization, and epoch-zero retention. The existing memory-runtime test now verifies saved inputs exactly equal those received by the actual model forward pass. Training tests use synthetic fixtures, not claimed human data or a newly trained real-game candidate.

Headless Edge exercised one actual step in `runs/human-feedback-train-v1` (seed 60000), saved and revised a temporary test label, verified that neither the frame nor decision advanced, and removed that label. The collection therefore had **zero active labels** after testing; withdrawn automated test labels cannot enter training. Controls remained available in focus mode, mobile layout had no page overflow, and no uncaught browser errors occurred. Evidence: `runs/human-feedback-browser-check.json`, `runs/human-feedback-desktop.png`, and `runs/human-feedback-mobile.png`. Human collection and real candidate training remain to be done by following the live guide. This iteration validates the workflow, not improved gameplay.


## Guided labeling after the first completed session (October 1, 2026)

The user completed all three episodes in `human-feedback-train-v1`: seeds 60000-60002 each killed one target in 17 decisions with return 35. The action sequences were identical. The feedback manifest still contained only the withdrawn browser-test entry, with **zero active labels** and no separate validation collection. These outcomes are playback results, not new learning; no candidate was trained on them.

Added `--teach` to keep each decision, including a finished episode's final observation, visible until the operator advances. The main button becomes Next decision and advances once. Save & next decision saves the explicit label before requesting one step. Final-run labels remain savable without attempting another step. Teaching mode overrides autoplay; repeated clicks during computation do not queue extra decisions. Ordinary playback behavior is unchanged.

The focused observer, feedback, and memory suites passed **22 tests**, including two new checks for guided stepping, episode-boundary labeling, and repeated-click queue suppression. Edge verified one-step behavior and the Save & next continuation; the save request was intercepted with a synthetic success response so this browser check did not manufacture operator labels. Persisted labeling and its immutability are covered by the HTTP and session tests. Browser evidence is in `runs/teaching-mode-browser-check.json` and `runs/teaching-mode-desktop.png`.

The new plan scanned 25 opening images, selecting eleven distinct frames absent from the checked earlier human/student training and validation recordings. It did not inspect human test frames or gameplay outcomes. Training starts are 60011, 60013, 60014; validation starts are 60015, 60017; later matched gameplay starts are 60020, 60022, 60023, 60025, 60030, 60034. The full checksums and source manifests are in `runs/human-feedback-plan-v2/report.json`. This establishes opening-image uniqueness relative to the checked sources only, not independent tasks or non-overlapping later trajectories. The prepared training and validation sessions are paused for actual operator collection. Real feedback training is still pending labels.


## First real human-feedback training and paired gameplay (October 3, 2026)

The operator saved 35 training labels across starts 60011 and 60013, and 17 validation labels across 60015 and 60017. The training labels are 32 MOVE_RIGHT and 3 ATTACK; validation contains 9 MOVE_RIGHT, 5 MOVE_LEFT, and 3 ATTACK. Training labels disagree with 23 recorded actions, and validation labels disagree with 7. These are explicit operator judgments, not verified optimal actions.

Byte-preserving snapshots in `runs/human-feedback-frozen-v1` freeze the two v3 collections. Their checksums, source paths, parent hash, reserved evaluation starts, and the pre-evaluation training settings are recorded in `snapshot.json`. The 50-epoch run used learning rate 0.00001 and seed 41. Selection retained epoch 9 as `runs/fly-student-human-feedback-v1`. No gameplay outcome selected an epoch or changed these settings.

All 17 feedback validation inputs remain after exact-duplicate checks. The earlier human validation split loses 54 exact training-input duplicates, leaving 127 examples. Feedback validation accuracy stays 10/17 (58.82%) and balanced accuracy stays 48.89%; balanced NLL changes from 1.422208 to 1.419719. Earlier validation accuracy changes from 78/127 (61.42%) to 75/127 (59.06%); balanced accuracy changes from 41.54% to 40.63%. This fits the existing five-percentage-point retention constraint but is not a meaningful demonstrated accuracy improvement.

The actual input, recurrent, and action-readout parameters changed. Parent normalization, parent checkpoint bytes, original fly weights, and the hash-locked runtime implementations were unchanged. The Laya teacher was not loaded for this training or gameplay; its recorded rejection remains unchanged. Tests in this iteration were actual checkpoint loading, training/provenance audits, complete native-engine trajectories, and browser checks for the review artifact; no simulator or policy-runtime source was changed.

Both checkpoints then played all six pre-reserved starts with four game tics per decision and a 75-decision bound. The candidate was already selected before either evaluation. Opening-image uniqueness had been checked against specified older training/validation frames, not human test frames; this remains a small basic-scenario development comparison.

| Model | Target kills / 6 | Mean return | Longest WAIT streak |
|---|---:|---:|---:|
| Memory parent | 2/6 | -220.50 | 23 |
| Human-feedback candidate | 0/6 | -328.33 | 31 |

| Start | Parent return / kills | Candidate return / kills |
|---|---:|---:|
| 60020 | -325 / 0 | -345 / 0 |
| 60022 | 39 / 1 | -335 / 0 |
| 60023 | -310 / 0 | -310 / 0 |
| 60025 | -325 / 0 | -320 / 0 |
| 60030 | -57 / 1 | -315 / 0 |
| 60034 | -345 / 0 | -345 / 0 |

Under the predefined rule (target kills first, mean return for ties), the paired winner is **parent**. The default checkpoint was not changed. This pilot does not establish a general gameplay or biological-topology advantage.

For example, on start 60022 the first action difference is decision 15: the parent chooses ATTACK (29.09%, versus 27.87% RIGHT), while the candidate chooses RIGHT (29.35%, versus 28.67% ATTACK). The parent kills the target on decision 16; the candidate eventually times out. The comparison report verifies matching graph spike totals through the first action difference. This illustrates a trajectory divergence from a small policy change; it does not isolate a biological circuit as the cause.

`runs/human-feedback-comparison-v1/report.json` audits checkpoint hashes, plan/split separation, episode order, trace length, actions against probabilities, action totals, episode returns, parameter differences, normalization, and first action divergences. Raw reports and traces are in `human-feedback-parent-eval-v1` and `human-feedback-candidate-eval-v1`. The candidate and the paired gameplay results are retained separately.

`runs/human-feedback-review-v1/index.html` is a standalone local interactive review of all 52 frozen labeled frames. It displays the operator label, recorded action, and the parent/candidate probabilities on identical stored inputs. Parent probabilities match the recording within 1e-7 absolute / 1e-6 relative tolerance. On training inputs, label agreement changes from 12/35 to 13/35 (one action changes); on validation inputs it stays 10/17 with no changed actions. Edge checked the collection filter, previous/next buttons, slider boundaries, actual image loading, probability summary, and mobile overflow, with no uncaught exceptions. This is fixed-observation inspection, not another autonomous game.

## Laya Vision integration and research workbench (October 3, 2026)

Loaded the actual 201,161,347-parameter `thaitea/laya-vision` checkpoint at revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc` using fork commit `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c`. The isolated worker ran FP32 inference on the local RTX 3060 Laptop GPU, with Hub networking disabled after the verified download. Pillow 12.3.0 and torchvision 0.29.1+cu130 were added; the original installed text-Laya package was retained. Student, action-memory, original checkpoint and simulator hash contracts remained intact.

Three native-engine conditions used seeds 72000, 72001 and 72002, the same basic scenario and bundled Freedoom2 assets, four tics per action, and at most 75 decisions per episode. Both branches were measured in every condition. The only control difference was the fixed probability-mixing coefficient. These are development starts; exact opening separation from earlier training was not established.

| Control | Vision coefficient | Target kills | Mean return | Total decisions |
|---|---:|---:|---:|---:|
| Vision | 1.0 | 3/3 | 83.00 | 15 |
| Vision + existing student | 0.8 | 3/3 | 83.00 | 15 |
| Existing student | 0.0 | 2/3 | -111.00 | 126 |

Vision and the mixture both produced per-seed returns 87, 79 and 83, in 4, 6 and 5 decisions. Their applied actions matched in these runs. The student produced returns 35, -38 and -330 in 17, 34 and 75 decisions. This checks the integration and illustrates the visual teacher's value in this small sample. It does not demonstrate that the fly graph improves the Vision model or that the student has learned from it. No policy or connectome training occurred.

`runs/vision-integration-validation-v1/report.json` audits all 156 decisions: saved PNG against the worker's RGB hash, NPZ checksum, exact fusion and argmax action, previous applied action and all 17 causal memory inputs, Vision logits/temperature against probabilities, and scorer products plus residual and bias against raw logits. Re-evaluating the unchanged student on every saved feature vector gave **zero maximum absolute probability difference**. The source traces and images are under `runs/vision-workbench-v1`.

The new frontend at `flydoom/web/research` replaces the default documented workflow with a notebook-style research instrument. Existing map and checkpoint primitives are reused. The legacy observer remains available separately. All **132 tests passed**, including new checks for mixture extremes and invalid distributions, paired observation/action semantics, immutable student parameters, causal applied-action history, input-drive reconstruction, stop during in-flight Vision inference, repeated step suppression, source hash/path validation, and same-origin/new-run request constraints.

Actual headless Edge checks covered 1600-pixel desktop and 390-pixel mobile widths, large game-image rendering, four action rows, sixteen Vision scorer rows, historical image and map alignment, weight inspection, and a real pointer selection of an engineered cell in the WebGL atlas. The game frame measured 787 pixels wide on desktop and 356 on mobile. There was no page-level horizontal overflow, uncaught JavaScript exception, or WebGL error.

A separate browser lifecycle check ended the initial paused run, created a new run from the form, stepped twice, ran and paused at decision three, verified that it remained paused, inspected a previous observation, ended the run, then created a fresh run and left it paused at decision one. For the inspected cell, observed input drive was 1.2491939068 and the independently reconstructed sum plus bias was 1.2491938472. The full 576-element final Vision scorer vector was exposed. Evidence: `runs/research-browser-check.json`, `runs/research-controls-check.json`, and the desktop, circuit, weights and mobile PNGs in `runs/`.

The active prepared session is under `runs/vision-workbench-v2`; it is an inspection run, not part of the comparison table. The upstream model emits a pad-token configuration warning; observed single-frame predictions passed the numerical checks above. Broader scenario validity, Vision distillation, biological interpretation and a topology advantage remain unestablished. See [the developer architecture and API guide](VISION_WORKBENCH.md).

## Decision playback and engine outcomes (October 3, 2026)

The user's completed inspection run `runs/vision-workbench-v2/run-20261003-031825-391309` contains four decisions: two MOVE_LEFT and two ATTACK, one target kill, and return 87. The old observation panel displayed the image before the final action, which did not convey the terminal result. The scenario itself only exposes strafing and shooting.

Reconstructed all 15 applied decisions from the three-episode hybrid run `runs/vision-workbench-v1/run-20261003-030435-554880` in the native engine. Every decision's RGB hash before action and cumulative return afterward matched the original trace; stored observation checksums and teacher/student probabilities were also checked. The resulting `runs/replays/vision-hybrid-15/replay.json` indexes 66 game frames, six ATTACK decisions, and three engine-confirmed kills. This replays recorded actions, not another neural inference experiment. No terminal image is fabricated: the engine ends immediately on a kill, so the player holds the last available frame and displays an explicit outcome label.

All **136 tests passed** after adding replay capture and API support. New checks cover reconstruction mismatch rejection, native frame persistence without changes to actions or student parameters, frame checksum tampering, path escape, frame bounds and HTTP bytes. Existing simulator/checkpoint hash contracts still pass.

Actual Edge playback checks verified 15 selectable cards, frame progression, pause stability, previous/next controls, slider selection, hit/end labeling, automatic stop on the final decision, and isolation from the live engine. The recorded image measured 841 pixels wide at a 1600-pixel viewport. Both desktop and 390-pixel mobile layouts had no page overflow. No uncaught JavaScript or WebGL errors occurred. Evidence: `runs/replay-browser-check.json`, `runs/replay-desktop.png`, and `runs/replay-mobile.png`.

A fresh paired run at seed 72020 (`runs/vision-workbench-v3/run-20261003-034150-947673`) completed in four decisions, with two LEFT and two ATTACK actions, return 87, and one kill. The last action advanced two tics and had reward delta 99. The browser then refreshed the replay library and loaded all four newly captured decisions. The workbench was left in a new paused run at seed 72000 under `runs/vision-workbench-v3/run-20261003-034327-355208`. These are UI/capture checks, not a new training or generalization result.

## First offline Laya Vision distillation (October 3, 2026)

The experiment plan `runs/vision-distillation-v1/plan.json` reserved collection and evaluation seeds before new recordings began. Existing mixed/student trajectories and six new teacher-controlled episodes supplied training; six other teacher-controlled episodes supplied validation. All 12 new teacher episodes hit the target. After deduplicating within each split and removing four validation rows that repeated a training RGB image or exact student input, 147 training rows and 26 validation rows remained. Labels were the pinned Vision model's complete action distribution; no human target, engine reward, kill count or future action entered the student's inputs.

The separate candidate `runs/fly-student-vision-v1` continues the existing 64-cell memory student with 100 epochs of KL distillation, AdamW learning rate 0.0001 and batches of 32. Epoch 100 had the lowest validation KL; epoch zero was an eligible fallback. Parent mean/scale buffers were identical afterward. Input, recurrent and output weights/biases changed (172,548 scalar entries); the original checkpoint and frozen biological graph remained intact. Candidate SHA256: `434f2d4b8b1b90768f7cf16f85578cc789453459871b03c7c3d63f7b626bd4e2`.

| Saved-observation metric | Parent | Vision-trained candidate |
|---|---:|---:|
| Training teacher agreement | 14.97% | 97.28% |
| Validation teacher agreement | 50.00% | 61.54% |
| Validation KL | 1.14097 | 0.69757 |
| Validation ATTACK agreement | 0/7 | 4/7 |

These are teacher imitation measurements, not game win rates. The large train/validation gap limits the inference. The gameplay comparison below was run only after checkpoint selection, with **no Vision model instantiated or executed**. Both students used the same native scenario, frozen connectome, causal applied-action history and maximum of 75 decisions. The comparison criterion was fixed in advance: target kills first, mean return for ties, parent retained on exact ties.

| Seed | Parent decisions / return / kills | Candidate decisions / return / kills |
|---|---|---|
| 73040 | 22 / 15 / 1 | 17 / 23 / 1 |
| 73041 | 10 / 63 / 1 | 6 / 79 / 1 |
| 73042 | 17 / 35 / 1 | 5 / 83 / 1 |
| 73043 | 12 / 55 / 1 | 8 / 71 / 1 |
| 73044 | 17 / 35 / 1 | 5 / 83 / 1 |
| 73045 | 75 / -325 / 0 | 8 / 71 / 1 |

The candidate wins this pilot: **6/6 kills and mean return 68.33**, versus **5/6 and -20.33** for the parent. Total decisions drop from 153 to 49, with WAIT actions falling from 107 to 6. Seeds 73042 and 73044 have the same opening RGB as a training observation (and each other). Excluding these two leaves 4/4 kills for the candidate and 3/4 for the parent, but does not establish independence from all earlier training, broad gameplay generalization, or a causal advantage of the biological topology. Evaluation seeds must not be recycled as an untouched test set after inspecting these results.

Evidence is in `runs/vision-distillation-eval-v1/report.json`, `run-parent`, `run-candidate`, and `audit.json`. The audit independently recomputes each saved decision through its fixed checkpoint, checks the previous action and all 17 memory inputs, NPZ hashes, action argmax, episode totals and replay frame hashes. The source plan and selected model are unchanged by gameplay evaluation. The default command still selects the original memory parent; the prepared workbench explicitly loads the new candidate with `alpha=0`.

All **146 tests passed**, including new checks for normalization preservation and actual parameter learning, train/validation input or RGB overlap, reserved-seed collisions, altered RGB/NPZ artifacts, mismatched Vision identity, causal memory, and teacher-free native evaluation. Real Edge tests confirmed the two-row training metrics panel, 49-decision/6-kill candidate replay, absent Vision predictions in that replay, final hit indication, no desktop/mobile overflow, and no JavaScript/WebGL errors. A live single step at alpha zero applied the candidate's WAIT proposal even though the observer Vision proposed LEFT, verifying that the student actually controlled the action. Evidence: `runs/distillation-browser-check.json`, `runs/distillation-replay-check.json`, and their PNGs. The server remains open under `runs/vision-student-workbench-v1`, paused after that one decision.


## Synchronized archive signal inspector (October 3, 2026)

The replay now pairs its game view with an interactive signal inspector using the same archived decision. The separate native-observation panel remains removed. `replay_signals.py` locates the recording's exact checkpoint by SHA256, verifies the original trace and NPZ/image hashes, reproduces student activity and probabilities from recorded features, and exposes input/output contribution arithmetic. No missing whole-brain history or Vision inference is fabricated. The 49-decision candidate archive verified locally in 1.46 seconds (excluding interpreter startup).

All 18 targeted replay, signal-inspector and research tests passed. Nine new tests cover input/output reconstruction and rejection of modified weights, NPZ, trace, frame, spike activity, decision identity, probability distributions and source paths. Real Edge checks exercised cell selection, input inspection, mixture changes, direct-output removal, reset, play/pause, decision seeking and switching from a hybrid recording to the student-only candidate. The game remained visible beside the inspector while scrolling. Signal details matched the selected episode/decision; the live engine sequence remained unchanged. Measured input-drive reconstruction error was 2.61e-8 and output-logit reconstruction error was 1.28e-7 for the inspected example. Desktop (1600 px) and mobile (390 px) had no page overflow or uncaught browser errors.

Evidence: `runs/signals-browser-check.json`, `runs/signals-desktop.png`, and `runs/signals-inspector.png`. The workbench is running the trained candidate, paused at its opening, under `runs/vision-signal-workbench-v1`. The local sandbox changes only arithmetic previews; it does not establish causal biological effects or alter a policy, replay, or live game.

## Side-by-side game and connectome observatory (October 3, 2026)

The research interface now uses one game/map workspace with explicit archive and live modes. Recorded decisions drive both panels from the same index. The archive map highlights all retained descending voltage features and the exact student's reconstructed spike activity; missing whole-brain history remains explicitly unavailable. Active-cell rows rank absolute contributions to the inspected action and show each cell's strongest positive output contribution. Selecting a cell exposes input products, output weights and signed logit contributions. The default recording matches the loaded student checkpoint when a student-only archive exists.

All **18 targeted tests passed** across replay, replay signals and research modules. Added assertions verify that descending voltages, rates, previous actions and root IDs match the saved feature vector. Edge checks confirmed the 49-decision trained archive, aligned game/map panels, selected-cell links, numerical input inspection, map picking, seeking, play/pause, live single-step, and archive/live source isolation during pending requests. Archived playback did not advance the live engine. Switching to the older hybrid recording loaded its own checkpoint values. The completed interaction check reported no JavaScript or WebGL errors.

Additional layout checks at 1600, 1440 and 390 pixels confirmed that the game image fits its viewport without covering the footer, the desktop panels align, and the page has no horizontal overflow. Evidence: `runs/observatory-check.json`, `runs/observatory-layout-check.json`, `runs/observatory-desktop.png`, `runs/observatory-laptop.png`, and `runs/observatory-mobile.png`. The prepared server uses the trained candidate at alpha zero under `runs/vision-observatory-v1`; its live session remains paused. These checks validate interface behavior, not additional training or biological causality.

## Locked 20-start Vision student validation (October 3, 2026)

`flydoom.vision_validation` fixed 20 seeds before gameplay, excluding 160 previously declared or recorded seed values found in local experiment metadata. The plan locks 48 artifacts, including both checkpoint files/reports, both retained distillation splits, calibration, Python source and game assets. Plan: `runs/vision-validation-plan-v1/plan.json`; SHA256: `5eff50dd46aa55f67821dd80f5169b3c192468be3438bd12a73e1377e8854c34`. All locked artifacts passed verification before and after execution. No Vision model or optimizer ran, and no default checkpoint was changed.

All three conditions completed the same 20 native-game starts, with at most 75 decisions each. The disconnected condition disables synaptic propagation while preserving visual input drive, student weights and causal action memory.

| Condition | Target hits | Mean return | Decisions | WAIT / ATTACK decisions |
|---|---:|---:|---:|---:|
| Memory parent | 15/20 | -60.40 | 662 | 450 / 44 |
| Vision-distilled candidate | 19/20 | 33.70 | 278 | 24 / 110 |
| Same candidate, disconnected graph | 0/20 | -300.00 | 1500 | 0 / 0 |

The disconnected policy chose RIGHT on all 1500 decisions. Its descending voltage/rate inputs were exactly zero; input cells may still spike because their external drive remains enabled. This demonstrates the fixed candidate's dependence on graph-derived signals under this intervention. It does not establish a biological-topology advantage over a retrained conventional or rewired network, and the intervention changes the student's input distribution.

The 20 seeds produced only **eight distinct opening RGB images**. Thirteen starts repeat the same image found in Vision training; none match retained Vision validation images. All 20 remain in the primary results. Across complete trajectories, parent/candidate/disconnected RGB overlaps with training were 221/65/13 decisions, and exact student-input overlaps were 221/52/0. No trajectory matched the retained validation RGB or input vectors. These checks do not establish independence from the parent's earlier human/text-teacher training.

| Opening group | Starts | Parent decisions / return | Candidate decisions / return |
|---|---:|---|---|
| Training-matched image | 13 | 17 / 35 each | 5 / 83 each |
| Seed 5349712 | 1 | 45 / -82 | 16 / 39 |
| Seed 3587243 | 1 | 75 / -325, no hit | 75 / -390, no hit |
| Seed 6129789 | 1 | 21 / 14 | 5 / 83 |
| Seed 8872304 | 1 | 75 / -330, no hit | 23 / -4 |
| Seed 7480885 | 1 | 75 / -315, no hit | 14 / 36 |
| Seed 6152622 | 1 | 75 / -315, no hit | 54 / -163 |
| Seed 6845968 | 1 | 75 / -310, no hit | 26 / -6 |

A post-hoc descriptive breakdown of the seven openings unmatched to either retained Vision split gives **2/7 parent hits, 6/7 candidate hits, and 0/7 disconnected hits**, with mean returns -237.57, -57.86 and -300.00. This subgroup does not replace the primary analysis. The candidate's remaining failed start uses 53 ATTACK decisions, compared with 7 for its parent; reduced waiting can become repeated ineffective shooting. Another candidate success takes 54 decisions and 26 attacks. These recovery failures justify broader visual teaching data before moving to harder gameplay.

The planned paired bootstrap resampled exact-opening groups 10,000 times with RNG seed 401. Candidate minus parent: hit-rate difference +20 percentage points, descriptive 95% interval +4.55 to +87.50 points; return difference +94.10, interval +53.18 to +249.25. Connected minus disconnected candidate: hit-rate difference +95 points, interval +62.50 to +100; return difference +333.70, interval +159.25 to +374.53. The wide intervals reflect eight opening groups and strongly uneven repetition. They describe this fixed checkpoint and scenario, not variation across training seeds or broad game generalization.

All **41 targeted tests passed**, covering protocol/hash changes, invalid seed bounds, paired episode alignment, grouped statistics, RGB/input overlap, replay integrity and connected/disconnected recording labels. An independent audit reproduced all **2440 decisions** with maximum probability error zero, checked exact spike activity and causal memory/previous-action features, reconciled episode outcomes, and verified **12073 frame hashes**. Edge loaded all three archives, verified their neural data, and exercised active-cell input/output inspection without browser exceptions. Evidence: `runs/vision-validation-v1/report.json`, `audit.json`, `browser-check.json`, and the three `run-*` recordings. The browser workbench remains available under `runs/vision-observatory-v1`; recording refresh exposes the new runs.

M3 remains incomplete: this is one training seed, eight opening images and a fixed-policy disconnection control. Independent training repetitions, more diverse held-out scenes and retrained matched network/topology controls remain necessary. These 20 seeds are now inspected development evidence and must not be recycled as an untouched future test set.

## Map label spacing and narrow-screen layout (October 3, 2026)

At 1366 by 768, the previous map compressed the student grid, action-memory label and previous-action label into overlapping vertical bands. The research map now positions layers from their actual grid bounds with reserved label gaps. The shared renderer exposes a label hook; the legacy observer retains its original label positions. Map height no longer collapses with a short viewport. Panel headings, controls and metadata wrap, and the workspace stacks below 1000 pixels. Probability cards use two columns on narrow screens; inspection tables retain local horizontal scrolling.

Real Edge checks at widths 1600, 1366, 1093, 900, 768 and 390 verified separation between each layer and its label, space above the map footer, no page overflow, and working selected-cell inspection. The 1093-pixel viewport also exercises the effective width of a 1366-pixel display at approximately 125% zoom. No browser exceptions occurred. Evidence: `runs/layout-fix-check.json`, `runs/layout-before.png`, and `runs/layout-fixed-*.png`. These changes affect layout only, not policy weights or recorded decisions.

## Diverse recovery teaching, three continuation seeds (October 3, 2026)

The local plan `runs/vision-recovery-v1/plan.json` was fixed before collection, with SHA256 `117456524f090bce6be4414fb6b795aae3fb7e779eee17a87451da70abad9fa6`. An outcome-free scan of 86 opening frames found 38 distinct images outside 151 known images from the preceding Vision teaching splits and gameplay openings. The first 12 eligible openings became training, the next 6 validation, and the remaining 20 a reserved gameplay set. Selection used only seed and exact RGB hash. This is exact opening separation, not guaranteed independence from the parent's older history or all subsequent trajectories.

The existing Vision student controlled all collection actions at alpha zero while the pinned Vision model labeled the same observations. Each episode was bounded at 24 decisions, so collection kills are not comparable with the earlier 75-decision evaluation. The 12 training episodes yielded 220 decisions; the 6 validation episodes yielded 111. All actual opening hashes matched the locked scan. One validation row was removed for exact training RGB/input overlap, retaining 110 validation samples. No reserved evaluation opening appears in any of the 331 collected RGB observations.

Each continuation started from the same intact `runs/fly-student-vision-v1` weights and normalization. Seeds 101, 202 and 303 independently randomized minibatch ordering for 100 epochs at learning rate 0.0001. Selection used the lowest validation KL within each repetition, including epoch zero. These are three stochastic continuation runs, not three independent network initializations. All three results are retained without cross-seed gameplay selection.

| Repetition | Selected epoch | Initial validation KL | Final validation KL | Initial teacher agreement | Final teacher agreement |
|---|---:|---:|---:|---:|---:|
| 101 | 88 | 0.97396 | 0.51190 | 28.18% | 61.82% |
| 202 | 97 | 0.97396 | 0.51266 | 28.18% | 60.91% |
| 303 | 100 | 0.97396 | 0.50526 | 28.18% | 59.09% |

Training teacher agreement reached 96.82% in every repetition, leaving a substantial training/validation gap. On retained validation examples, ATTACK agreement changed from 43.48% to 47.83%, 52.17% and 47.83%, respectively. Better imitation is not yet evidence of improved recovery during autonomous gameplay. Training on new data alone can forget older behavior, and the teacher's own accuracy on these recovery states has not been independently established.

Each candidate changed 172548 scalar parameter entries in the engineered readout; parent mean/scale buffers were exactly preserved. The biological weights and visual teacher remain fixed. Checkpoints: `runs/fly-student-recovery-v1-seed-101`, `-202`, and `-303`. The independent audit rebuilt the verified datasets, reproduced all selected validation metrics, checked epoch selection and normalization, and confirmed three distinct checkpoint hashes. It also checked the reserved-opening overlap against all collected frames. Evidence: `runs/vision-recovery-v1/report.json`, `audit.json`, `opening-scan.json`, both `run-*` recordings and each candidate's report.

All **24 targeted recovery, distillation and research tests passed**, including eight new recovery tests for history exclusion, duplicate scene/seed separation, insufficient scene diversity, protocol/artifact mutation and actual collected-opening identity. The running workbench API verified neural signals for both collection archives (220 and 111 decisions). No new candidate gameplay evaluation or checkpoint promotion occurred; the browser's active checkpoint remains the earlier tested Vision student. The next step is a locked paired gameplay comparison of all three continuations and their common parent on the 20 reserved openings, with trajectory-overlap reporting before drawing generalization conclusions.

## Complete anatomical cloud and live inspection (October 6, 2026)

The research map replaces its fixed camera scale with a fit computed from all 139,255 projected anchor positions. The CPU fitting and picking equations match the existing WebGL rotation and perspective. Bounds are cached by camera orientation; resizing updates the available rectangle. The enlarged brain view uses the full map area, while the integrated view reserves space for the engineered layers. Unmeasured anchors have higher visibility without being labeled as active. Projection controls expose the source XY, XZ and YZ planes without assigning anatomical directions.

Edge verified 40 combinations: viewport widths 1600, 1366, 1093, 900 and 390; four projections; integrated and enlarged modes. Every anatomical point remained inside the fitted area, the integrated mode retained all 89 engineered nodes, and enlarged mode hid them. Reset restored zoom, pan and the all-anchor filter. Anatomical selection worked in enlarged mode; student selection returned to the circuit and opened four output contributions. There was no page overflow, WebGL error or browser exception. Screenshots were visually inspected at desktop and mobile sizes. Evidence: `runs/brain-fit-check.json`, `runs/brain-fit-*.png`.

A separate live single-step check retained alpha zero and the tested Vision parent. Vision ran as an observer, while the student applied the action. The paired map exposed 3792 spiking biological cells for decision 1, sequence 2; selecting a real root ID fetched connectivity for that exact sequence without closing the enlarged view. The session remained paused. Evidence: `runs/brain-live-check.json` and `runs/brain-live.png`. This is an example of current simulated activity, not physiological validation. Archive views continue to disclose that only descending voltage features were retained. Neither rendering mode reconstructs full cell morphology.

## Reserved recovery gameplay comparison (October 6, 2026)

The locally locked `runs/vision-recovery-gameplan-v1/plan.json` has SHA256 `473d8489c83a522495647a3673fc6590c42ab22c0479681d4967f9ac56aba5c3`. It fixes the earlier Vision student and all three already selected recovery checkpoints, their datasets, calibration, current Python sources, 20 reserved opening hashes and the 75-decision limit. The original teaching lock, including native game assets, also passed verification before planning and after execution. Every condition used the same 20 seeds and matched all 20 distinct reserved opening RGB hashes. This is a local reproducibility lock, not public preregistration.

All 80 episodes completed. Vision was never instantiated by the evaluator, no optimizer ran, biological weights stayed fixed, and no default checkpoint was promoted. Each condition followed its own applied-action history. Neural state and action memory reset at episode boundaries; normalization was exactly preserved from the common parent.

| Condition | Target-hit episodes | Mean return | Decisions | WAIT / LEFT / RIGHT / ATTACK |
|---|---:|---:|---:|---|
| Earlier Vision student, shared parent | 13/20 | -135.15 | 811 | 26 / 178 / 147 / 460 |
| Recovery continuation 101 | 15/20 | -72.30 | 658 | 0 / 246 / 284 / 128 |
| Recovery continuation 202 | 14/20 | -78.15 | 668 | 0 / 260 / 291 / 117 |
| Recovery continuation 303 | 15/20 | -74.70 | 674 | 0 / 253 / 296 / 125 |

The earlier 19/20 result for this parent used different openings, many repeated. It is not a before/after comparison with the present 13/20. Here the valid comparison is within the same 20 reserved starts. Across the three continuation seeds, mean hits were 14.67 with sample SD 0.58; mean return was -75.05 with sample SD 2.94. These are minibatch-order variations from one shared warm start. The 60 candidate episodes share 20 scenes and must not be treated as 60 independent test scenes or three independent initializations.

| Continuation minus parent | Hit-rate difference | Descriptive 95% interval | Return difference | Descriptive 95% interval |
|---|---:|---|---:|---|
| 101 | +10 percentage points | -20 to +40 points | +62.85 | -51.40 to +177.46 |
| 202 | +5 percentage points | -25 to +35 points | +57.00 | -50.61 to +168.05 |
| 303 | +10 percentage points | -20 to +40 points | +60.45 | -51.65 to +173.05 |

Intervals use the predeclared 10,000-draw paired opening bootstrap with RNG seed 401. Every interval includes zero. All three measured mean improvements, but this small evaluation does not establish a reliable superiority claim. No checkpoint was chosen from these gameplay outcomes. Different opening pixels also do not establish broad scene independence or separation from the parent's older human/text-teacher history.

Each candidate lost four parent successes: seeds `11004984`, `11008145`, `11006986` and `11007510`. All three reached 75 decisions without a kill on those starts, with frequent LEFT/RIGHT changes. Continuations 101 and 303 recovered six parent failures each; 202 recovered five. Seed `11005696` failed for parent/101/202 but succeeded for 303 in 58 decisions; seed `11003621` succeeded only for 101, in 33 decisions. These exchanges support collecting observations reached by the new candidates and rehearsing earlier useful behavior. They do not identify a proven causal defect or justify tuning on these starts while still calling them untouched test data.

No opening matched either retained recovery split or either earlier Vision split. Across entire trajectories, all candidate RGB images and exact input vectors were unmatched to recovery training and validation. The parent had one RGB match to recovery validation, at seed `11005013`, and no exact input match. Against earlier Vision training, the parent had two RGB matches and each candidate one, all at seed `11003119`; there were no exact input matches or retained earlier validation matches. Thus opening separation improved substantially over the earlier experiment, while complete RGB trajectory separation is not universal. Older parent history remains unaudited.

All **38 targeted tests passed**, including seven new checks for gameplay protocol/artifact mutation, duplicate openings/seeds, missing conditions and retaining failed-episode scores. Independent auditing reproduced all **2811 decisions** with maximum probability error zero, verified exact mean spike activity and causal memory/previous-action inputs, reconciled rewards and kills, and checked **13864 frame hashes**. Checkpoint, dataset, source, normalization and original game-asset locks remained intact. Evidence: `runs/vision-recovery-gameplay-v1/report.json`, `audit.json`, and the four `run-*` recordings.

The 20 starts are now inspected development evidence. Further tuning needs a separately reserved test set. M3 still requires retrained matched network/topology controls and independent initialization studies; M2 physiology/retinal mapping and M4 biological synaptic training remain separate unresolved work. Better imitation and modest measured gameplay gains do not finish the broader research project.

Edge loaded all four completed archives, matched each archive's checkpoint hash and decision count, and inspected 15 evenly spaced decisions per condition. At all 60 samples, the correct 320-pixel game image loaded, the map matched the decision, Vision was explicitly absent, and student selection opened four readout contributions and restored circuit view from enlarged brain mode. The live parent checkpoint stayed unchanged, with no WebGL or browser exceptions. Evidence: `runs/vision-recovery-gameplay-v1/browser-check.json` and `browser.png`.

## Benchmark-to-live handoff and fresh reservation (October 9, 2026)

The new `flydoom.experiment` entry point verifies completed training and the completed recovery benchmark, including checkpoint identity, ordered openings, recomputed scores and paired intervals. It exposes all four models without promotion. Model switching is explicit, limited to a paused idle or ended session, and rebuilds the live connection catalog before creating a fresh paused alpha-zero game. Vision runs as an observer; the selected student controls actions. Historical simulation, student and benchmark Python sources were preserved.

All **35 targeted tests passed** across experiment, recovery evaluation and Vision research. Checks include incomplete/mismatched benchmark rejection, changed weights, feature-map incompatibility, invalid model/seed inputs, in-flight switching, reservation mutation and protected episode ranges. Edge exercised all four benchmark rows, parent-to-101 switching, exact checkpoint agreement across metadata/map/weights, a real paused single step and selected-cell inspection. That step chose MOVE_LEFT with 8,831 simulated biological spikes. A separate three-decision autonomous smoke run completed through the Run control (one WAIT, two LEFT, return -12, zero kills); this verifies execution only and is not a new performance benchmark. The browser returned to the parent with a fresh paused 75-decision run. Cross-origin model switching and reserved test seeds were rejected. Desktop and 390-pixel mobile checks had no page overflow or browser exceptions. Evidence: `runs/experiment-oct09-browser.json`, `runs/experiment-oct09-browser.png`, `runs/experiment-oct09-mobile.png`, and the runs under `runs/experiment-oct09-v2`.

`runs/vision-recovery-reservation-v2/plan.json` allocates 24 training, 8 validation and 20 future evaluation starts after a scan of 151 images against 529 discovered known RGB hashes. All 52 selected seeds and opening hashes are distinct and excluded hashes are absent. The reservation contains no teacher labels, policy decisions or rewards. Source/game-asset fingerprints and a plan checksum were saved. These are reserved starts, not a newly collected dataset or a newly trained checkpoint. Full-trajectory and semantic independence remain unestablished. The next collection/training protocol still needs to fix rehearsal and selection rules before execution.

## Anatomical geometry and six-action movement pilot (October 9, 2026)

The viewer now combines all 139,255 reference positions with 75 real neuropil meshes (68,860 triangles). The cached surfaces and manifest occupy 1,266,048 bytes. Their coordinates are transformed from nanometers into the same micrometer-centered space as the anchor cloud; fitting includes the complete meshes and selected skeleton. Regions come from the public FlyWire neuropil mesh source. The selected neuron uses the v783 skeleton service documented by [fafbseg](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.get_skeletons.html). Root `720575940603231916` loaded 3,588 vertices and 3,587 segments. This is real selected-cell morphology, not simultaneous reconstruction of all cell branches or synapses. Mesh colors identify geometry rather than activity.

The versioned movement student has 2,635 input features, 64 engineered cells and six outputs: WAIT, LEFT, RIGHT, ATTACK, FORWARD and BACKWARD. Matching parent input weights, normalization and four original logits transfer exactly; new history channels begin with zero weights and the new outputs with bias -2. Six-way probabilities need not equal the old four-way probabilities. Biological connectivity and the pinned visual teacher remain fixed.

The pilot reserved 8 training, 4 validation and 8 evaluation openings before collecting any labels or policy outcomes. Existing discovered opening scans and retained teaching RGB identities were excluded. Collection applied a declared forward/forward/left/backward/backward/right prefix followed by expanded-parent decisions, capped at 24 decisions. Each preceding image was independently labeled by the six-option Vision teacher. This prefix is exploration, not learned autonomous movement. Training retained 165 examples and validation retained 96; no validation row required exact RGB/input deduplication. One warm-start training seed used 80 epochs of KL imitation and selected epoch 80 by lowest validation KL, including epoch zero. There was no rehearsal of older teaching data.

| Metric | Before training | Selected checkpoint |
|---|---:|---:|
| Training KL | 0.97137 | 0.05191 |
| Training teacher agreement | 19.39% | 78.79% |
| Validation KL | 1.08381 | 0.18448 |
| Validation teacher agreement | 10.42% | 47.92% |

The teacher's training top-choice counts were `0 / 37 / 53 / 18 / 57 / 0`; validation counts were `0 / 0 / 57 / 4 / 35 / 0`, in canonical action order. No backward top-choice examples were supplied. The new motor output works, but learned retreat is not demonstrated. The substantial training/validation gap and unknown teacher accuracy also limit interpretation of better imitation.

The completed teacher-free benchmark used the same eight reserved openings for both conditions, with a 48-decision limit per episode:

| Condition | Target-hit episodes | Mean return | Decisions | WAIT / LEFT / RIGHT / ATTACK / FORWARD / BACKWARD |
|---|---:|---:|---:|---|
| Expanded parent | 6/8 | -49.75 | 206 | 13 / 54 / 33 / 106 / 0 / 0 |
| Trained six-action student | 6/8 | -50.75 | 237 | 9 / 96 / 89 / 26 / 17 / 0 |

The student autonomously selected forward 17 times, but did not improve measured overall gameplay. Backward was never selected. The paired hit-rate difference was zero with a descriptive 95% interval of -37.5 to +37.5 percentage points; mean-return difference was -1.00 with interval -76.50 to +106.13. Both intervals include zero. This single-seed eight-opening pilot does not establish a performance gain or meaningful navigation. Its results are not a before/after comparison with the different earlier 20-opening four-action benchmark. Those older checkpoints remain unchanged.

All eight openings matched the locked identities. No benchmark opening or trajectory RGB matched either retained movement teaching split. Different opening pixels do not prove semantic independence or separation from every older parent-training image. An independent artifact audit verified all 261 teacher-frame PNG/RGB hashes, retained targets, causal history and previous-action fields, checkpoint/source/plan fingerprints, saved-model training/validation metrics, and all 16 episodes' probability/argmax/count/reward accounting. This audit did not replay the full connectome a second time. Evidence: `runs/movement-pilot-v1/report.json`, `plan.json`, `audit.json`, `train.npz`, `validation.npz`, and `student.safetensors`.

All **42 targeted tests passed** across movement, experiment, Vision research and recovery evaluation. They cover native forward/backward displacement, six-action causal history, exact parent-logit transfer, geometry parsing and invalid indices, idle-worker release, existing benchmark admission and reservation protection. JavaScript syntax checks passed. Edge exercised the actual movement server against the saved trained checkpoint while the separate gameplay benchmark was still finishing; its temporary test harness relaxed only the completed-benchmark admission condition and did not alter checkpoint or report files. Manual forward/backward checks changed engine position, selection opened all six output weights, and the real skeleton loaded. An autonomous single step applied the student's argmax ATTACK with `manual=false`. Expand view, Escape, new-run reset, idle-memory release and the 390-pixel layout worked without page overflow, WebGL errors or browser exceptions. Desktop and expanded screenshots were inspected visually. Evidence: `runs/movement-browser.json`, `movement-brain-desktop.png`, `movement-brain-fullscreen.png`, and `movement-brain-mobile.png`.

Live history now retains at most eight full neural snapshots; explicit memory release keeps the latest. The older Vision observer loads on demand and unloads after 60 seconds idle. Six-action live inference loads no Vision worker and runs the active graph/student on CPU. After closing the collection worker and test browser, the device reported 97 MiB overall GPU memory and 0% GPU utilization; this is a point-in-time whole-device observation, not a memory guarantee. Closed test browser profiles and pytest cache were removed; datasets, checkpoints, downloaded anatomy and recordings were retained. Current six-action recordings are JSONL/NPZ artifacts and are not yet supported by the older four-action archive inspector.

After the benchmark and artifact audit completed, the normal `flydoom.experiment --movement` launcher admitted the completed checkpoint without any test harness. Metadata matched its saved SHA256. A real autonomous step selected MOVE_RIGHT, the maximum of its six probabilities, with 8,832 simulated biological spikes. The server then ended that smoke run and created a fresh paused run at decision zero. Evidence: `runs/movement-pilot-v1/live-handoff.json`. This smoke check establishes the completed-checkpoint handoff, not an additional performance result.

## Biological-edge optimization and research desk (October 9, 2026)

The requested synaptic-learning stage is now implemented independently of decoder training. The mask contains 293,249 existing nonzero directed pairs into the 1,303 descending output cells, among 15,091,983 pairs in the prepared graph. All 139,255 cells still execute in the original LIF simulator. The previously trained six-action readout, input mapping, timing and visual teacher are fixed.

Two bounded four-parameter searches, `runs/synaptic-pilot-v1` and `runs/synaptic-refine-v1`, retained the original graph: none of their penalized training objectives improved. The first tested coordinate perturbations of 0.15/0.075; the second tested seeded paired joint perturbations of 0.01/0.0025. Reports, candidate histories, plans and unchanged checkpoints are preserved. Their negative result motivated a separate individual-edge proposal, not an automatic acceptance of changed weights.

`runs/synaptic-eligibility-v1` uses direct synaptic-current/voltage eligibility and the frozen readout's surrogate gradient to propose each edge's magnitude gain. The local derivative holds presynaptic spikes and reset schedules fixed and omits indirect recurrent feedback and spike-rate-feature derivatives. It is not full BPTT. A two-cell finite-difference test validates the direct subthreshold derivative and confirms observing eligibility does not alter the forward trajectory. Every proposed candidate is then measured with the actual whole-graph hard-spike simulator from episode resets, using chronological frames and recorded causal action histories.

The predeclared line search tested eight signed step sizes between -0.01 and +0.01, including the original graph. Selection used training KL plus a small magnitude-change penalty only. The accepted step was +0.0001. There were 14,484 nonzero local gradients; float32 application changed **13,867 actual matrix entries**. Per-edge gains range approximately 0.9999–1.0001, comfortably inside the enforced 0.75–1.25 bounds. Group-level mean gains displayed in the interface are summaries, not the parameterization of this final checkpoint.

| Measure | Original graph | Accepted graph |
|---|---:|---:|
| Training KL, 16 frames from 8 episode prefixes | 0.1429181 | 0.1425697 |
| Validation KL, 8 frames from 4 episode prefixes | 0.1743336 | 0.1733302 |
| Development seed 74000, at most 12 decisions | 0 hits / return -53 | 0 hits / return -53 |
| Development seed 74001, at most 12 decisions | 0 hits / return -53 | 0 hits / return -53 |

The imitation gains are small. Validation did not select or tune the checkpoint. These scene prefixes were previously seen by the frozen decoder, and the familiar short game starts are implementation smoke tests rather than a held-out benchmark. No gameplay gain, navigation skill, biological validity, or topology advantage is established. All unselected weights, all transmitter signs, CSR topology and the decoder are unchanged. New live runs keep these trained weights fixed; there is no online optimizer.

An independent reload audit checked all three local source/artifact locks, reconstructed the final patch from exact endpoint IDs and baseline weights, counted all 13,867 changed entries against the original complete matrix, and verified signs, topology and the outside-mask values. It replayed all 24 selected teaching frames with the restored graph and reproduced final probabilities within 1e-7 and both reported KL values. Decoder tensors remained bitwise identical. Evidence: `runs/synaptic-eligibility-v1/audit.json`, `plan.json`, `report.json`, `eligibility.npz`, and `synapses.npz`. This audit did not repeat development gameplay.

The new `flydoom.laboratory` desk loads that exact patch, preserves its SHA256 in run reports and displays its trained graph identity. Its design follows a documented comparison of FlyWire Codex, Neuroglancer, neuPrint, Virtual Fly Brain and SharkViewer; [the research notebook](LAB_DESIGN_RESEARCH.md) maps primary sources to implemented choices and unimplemented features. The anatomical viewport renders all 75 real region surfaces with depth and shading, eight real context-neuron skeletons by default, optional all-neuron anchors and synchronized biological spikes. Skeletons are limited to 16 with released WebGL buffers. Selected-cell voltages, synaptic current, refractory state, directed partners, before/after weights and per-edge gains are visible. The activity view exposes all 64 engineered cells and their six output products. Retained decisions, input/outcome frame switching, source-plane projections, expansion and layer controls make these data inspectable without deleting their detail.

All **52 targeted tests passed** across synaptic learning, movement, experiment, Vision research and recovery evaluation. Edge verified the final 13,867-change checkpoint, 75 surfaces, 139,255 anchors, eight skeletons, actual single-step inference, selected-cell sequence identity, six readout contributions, cell search, training plots/tables, retained-decision review, live following, expansion/Escape and desktop/mobile layout. There were no browser exceptions, WebGL errors or horizontal page overflow. The mobile layer drawer can collapse to give the full width to brain anatomy. Desktop and mobile screenshots were visually reviewed. Evidence: `runs/lab-browser.json`, `lab-desktop.png`, `lab-expanded.png`, `lab-mobile.png`, and `lab-training.png`.

The final camera check projected 545,969 supplied coordinate vertices (anchors, expanded region triangles and loaded skeleton endpoints); none fell outside the viewport at the fitted default view. This count includes repeated triangle/line vertices, not additional neurons. The detailed training panel was also visually reviewed with visible per-edge numerical differences. Evidence: `runs/lab-fit-check.json`, `lab-final-desktop.png`, and `lab-training-detail.png`.

A separate production API smoke check applied actual manual forward and backward movement, continued autonomously to the 12-decision cap, verified the eight-snapshot bound and rejection of an expired snapshot, and checked the synaptic checkpoint hash in the saved live report. It then created a fresh paused 75-decision run and released older neural snapshots. Evidence: `runs/lab-live-check.json`. This mixed manual/autonomous check is not a performance benchmark.
