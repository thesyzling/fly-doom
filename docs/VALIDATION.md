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
