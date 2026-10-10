# Automatic synaptic learning

The project can now run a bounded **experience → candidate → evaluation → promotion** cycle without manual labeling. It changes magnitude gains on existing biological graph edges, while the six-action decoder and Laya Vision stay frozen. This implements a learning mechanism; it does not establish that the resulting policy is a competent Doom player or a biologically valid fly.

## Run it

From the project directory, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m flydoom.laboratory
```

Open `http://127.0.0.1:8770`. The desk loads the approved checkpoint from `runs/learning/registry.json`, or the original synaptic pilot if no candidate has passed. Use **Run**, **Pause**, **Single step** and **End run** to observe the fixed policy. End the current run, then press **Start one learning cycle**. Phase, measurements, benchmark comparison and the promotion decision appear in **Experience to candidate**. **Cancel candidate** requests cancellation at the next bounded operation; the current champion is not replaced. A separate worker closes the Vision process after label collection.

After a successful promotion, **Load approved checkpoint** constructs a new controller and starts a paused live run. **Roll back promotion** restores the previous registry entry; load that approved checkpoint to switch the live desk. Starting another cycle continues from the current approved graph. Rejected candidates remain available for research and never replace it.

The same pipeline runs without a browser:

```powershell
.\.venv\Scripts\python.exe -m flydoom.learning_cycle --cycles 1
```

Default budgets are four training episodes, two validation episodes, four promotion-gate episodes and four final-test episodes, balanced across two tasks. Training trajectories retain up to 12 consecutive decisions rather than the old two-decision prefixes. Evaluation runs for up to 32 decisions; these may be truncated episodes, and the report marks them. The loop tests an input-encoder alternative, one eligibility proposal and two antithetic recurrent-search directions. CPU simulation of the entire graph is the dominant cost.

A larger planned experiment can be started explicitly:

```powershell
.\.venv\Scripts\python.exe -m flydoom.learning_cycle --cycles 3 --train-scenes 16 --validation-scenes 8 --gate-scenes 20 --test-scenes 20 --decisions 50 --benchmark-decisions 75 --proposals 4
```

This is substantially more expensive. Seventy-five four-tic decisions cover the original basic scenario's time horizon; distance-task warm-up consumes some of that horizon. The finite default is an integration pilot, not evidence that training has converged. Ctrl+C cancels a CLI cycle. Do not start multiple learning workers: an exclusive lock prevents competing registry writers.

## The eight implemented components

| Component | Implementation and meaning |
|---|---|
| Longer neural training | Full chronological trajectories and causal action history, with a configurable budget. Every objective evaluation reruns all 139,255 neurons; no cached neural features stand in for changed synapses. |
| Independent benchmark | Before training, reserve disjoint seeds and distinct opening RGB hashes. The champion and candidate play the same gate starts autonomously. A separate final test is measured after the gate decision is recorded. |
| Forward/backward curriculum | Six native buttons, a deterministic exploration schedule and a distance-band shaping task produce measured movement outcomes, including failed movements. Positive progress contributes a bounded auxiliary target. |
| Richer visual input | A versioned, bounded contrast-and-temporal-change adapter competes against the active intensity encoder on training loss. Its coefficient is checkpointed. Zero retains the exact legacy input path. |
| Recurrent learning signal | A local eligibility proposal is supplemented with antithetic perturbations of actual edge gains and full recurrent hard-spike rollouts. This includes indirect feedback in measured losses, but is not exact BPTT or an unbiased high-dimensional gradient estimate. |
| Persistent neural replay | Every live decision saves full population voltage, counts, current, refractory state, decoder drive/activity/features and input/outcome images. Old decisions reopen from disk after RAM eviction and application restart. |
| Anatomical exploration | 78 named measured region surfaces, individual region filtering, 139,255 anchors, sequential loading of up to 32 skeletons, and bounded directed path queries. |
| Automatic learning | Collection, separate candidate optimization, paired evaluation, promotion vetoes, atomic checkpoint registry, cancellation and rollback. Repeating cycles needs no manual labels. |

## What supplies the learning target?

Pinned Laya Vision receives the native RGB image and returns the same six-action probability vector used in the movement pilot. The distance curriculum also measures a potential based on the distance between player and monster: zero within 145–215 engine units, decreasing outside that band. Near and far starts share this single objective. The initialization, warm-up and opening hashes are recorded in the plan.

Only measured **positive** potential change adds probability to the action actually taken. The mixture is `min(0.35, 3 * positive_progress)`. Negative progress does not become a positive label. Native return, kills and distance progress are reported separately; shaped reward is not relabeled as the Doom score.

Engine positions and object identities are reward diagnostics. They are never inputs to the image encoder, fly controller, readout or Laya. This is reward-assisted self-imitation with an external visual teacher, not pure reinforcement learning or a learned world model. Momentum and earlier actions may contribute to measured progress; the signal is not a counterfactual proof that the last button caused the improvement. A stronger retreat policy will need longer, more diverse episodes and possibly more appropriate maps.

## Candidate selection and data separation

- Train/validation/gate/test seeds are separated before labels and gameplay outcomes. Opening hashes are checked again when collecting and evaluating. Known retained seed and image records are excluded by the reservation scanner; this is not a claim to have reconstructed every historical unrecorded frame.
- Exact training-image duplicates can occur later within validation trajectories. They still advance recurrent state, but do not contribute to the validation score. Reported `samples`, `replayed_frames` and `excluded_duplicate_frames` distinguish these counts. At least four independent validation frames are required.
- Encoder and synaptic proposals use training KL only. Validation can veto promotion; it does not select a search direction.
- The gate requires finite paired results, no validation-KL regression, and no per-task mean regression in return, kills or distance progress. At least one gameplay metric must improve. Better imitation alone cannot promote a checkpoint.
- The final test's performance does not tune the candidate or alter the recorded metric-based gate. A subsequent data-integrity audit may veto adoption if teaching and benchmark trajectory images overlap. This veto and its reason are recorded explicitly.
- Descriptive paired bootstrap intervals are available in the benchmark display. Two pairs per task in the default run are far too few for a strong statistical conclusion. Pixel distinctness is not semantic independence; gate success is an engineering acceptance check, not proof of generalization.

The optimizer preserves CSR topology, signs, gains within 0.75–1.25, and every weight outside the plastic mask. A checkpoint verifies the original matrix, exact edge endpoint IDs, student and calibration identities, patch contents and reconstructed trained-matrix hash on load. There are no new invented synapses. Changing a simulated gain does not change the measured anatomical synapse count.

## Recordings and anatomical inspection

The timeline displays every persisted decision, including decisions that have left the eight-entry RAM buffer. Selecting a decision pauses live play and synchronizes the game image, all-neuron activity map, cell inspector and engineered readout. **Refresh** and **Open recording** reopen a previous run after its worker ends. An archive must match the graph, encoder and decoder; a mismatched recording is not silently interpreted using new weights.

Each run writes `neural/manifest.json` and one compressed NPZ per decision. SHA256 validates records on disk read; no pickled arrays are loaded. The per-run neural budget is 512 MiB; crossing it stops recording instead of silently dropping data. One previously read disk snapshot may be cached. **Release idle memory** releases RAM history and that cache, preserving files. Recordings and checkpoints are research artifacts, not disposable caches.

Region names are the filenames of the pinned fafbseg geometry archive. The new named dataset replaces the old numeric-only surface dataset in this desk; old numeric IDs are not guessed to mean named regions. Skeletons are fetched sequentially, with a 32-cell and 24 MiB geometry budget in the renderer, plus the existing 16-file raw download cache. The scene still does not contain all complete neuronal arbors or individual synapse coordinates.

The path search uses at most four hops, 48 strongest outgoing edges per cell and 4,000 visited cells. Its result is a directed graph route, not a demonstrated causal explanation of a decision. A failed bounded search does not prove that no path exists. Path order is numbered on the brain view and can be loaded as real morphology.

## Files for developers

### Sparse consensus follow-up

The desk's **Start one learning cycle** button now runs one bounded sparse-consensus experiment:

```powershell
.\.venv\Scripts\python.exe -m flydoom.synaptic_consensus
```

This collects up to 96 fresh training and 64 validation frames, rehearses 16 old training frames, and evaluates six paired starts each for the gate and final test (up to 32 decisions per model/start). A maximum of 4,096 existing edges can change from the active checkpoint. At least three training episodes must support an edge, with at least 75% sign agreement. Two predeclared gain steps compete using equal-episode training/rehearsal losses; validation never selects a candidate. The adapter, Laya weights and decoder remain fixed. The earlier `flydoom.learning_cycle` CLI retains its original dense perturbation algorithm for explicit reproduction.

New runs use `runs/learning/consensus-*/` and the same registry, cancellation, independent checkpoint audit and UI measurement APIs. **Candidate measurements** includes consensus support, rehearsal KL and actual changed-edge count. Live run creation protects held-out seeds from both cycle formats. The UI chart accepts sparse-search histories without group-gain snapshots.

`python -m flydoom.cycle_diagnosis <completed-cycle>` writes per-episode before/after KL and target-action coverage from verified frames and saved predictions. `python -m flydoom.external_data_audit runs/data-research-20261010` checks downloaded source hashes, action semantics and sample chronology. Neither command trains on external data. See [research findings](RESEARCH.md#dataset-audit-and-sparse-follow-up-october-10-2026).

### Module responsibilities

| File | Responsibility |
|---|---|
| `flydoom/learning_cycle.py` | Reservation, collection, candidate optimization, exact replay, gate, registry and CLI loop |
| `flydoom/synaptic_consensus.py` | Sparse episode-consistent proposals, old-training rehearsal and a bounded follow-up experiment |
| `flydoom/cycle_diagnosis.py` | Per-episode saved-prediction diagnosis and teacher-action coverage |
| `flydoom/external_data_audit.py` | Strict external action compatibility and source-sample audit; no training ingestion |
| `flydoom/learning_curriculum.py` | Deterministic native game setup, distance diagnostics and reward-assisted targets |
| `flydoom/learning_features.py` | Versioned contrast/motion adapter and episode reset |
| `flydoom/learning_job.py` | Local UI worker supervision and cancellation requests |
| `flydoom/learning_audit.py` | Independent reload/identity audit and paired benchmark summaries |
| `flydoom/neural_archive.py` | Atomic neural recording, checksum verification and bounded cache |
| `flydoom/named_anatomy.py` | Pinned named mesh archive and strict PLY parsing |
| `flydoom/graph_paths.py` | Bounded directed search over the actual loaded graph |
| `flydoom/laboratory.py` | Live session, archive and learning APIs |

`runs/learning/cycle-*/` contains a locked plan, status, teaching data or verified recovery references, candidate report, synaptic NPZ, benchmark traces and gate result. `latest.json` tracks the current cycle. `registry.json` exists only after a promotion and retains prior identities for rollback. Failed cycles remain documented. A complete teaching dataset from an interrupted run can be reused with `--experience <cycle-folder>` when it belongs to the same champion; fresh gate and test starts are still reserved.

Audit a completed candidate without retraining:

```powershell
.\.venv\Scripts\python.exe -m flydoom.learning_audit runs/learning/cycle-YYYYMMDD-HHMMSS-NNNNNN
```

The audit independently checks source, frame, label and checkpoint hashes, restores actual graph weights, verifies unselected weights/signs and recorded autonomous action choices, and summarizes the paired benchmarks. It does not rerun every neural loss or game a second time.

## Sources behind these choices

- [ViZDoom scenario definitions](https://vizdoom.farama.org/environments/default/) distinguish the original shooting task from movement and survival tasks. This implementation declares its distance shaping separately.
- [ViZDoom engine API](https://vizdoom.farama.org/api/python/doom_game/) documents episode initialization, seed control, native buttons and diagnostic state.
- [Spall's overview of simultaneous perturbation](https://www.jhuapl.edu/spsa/pdf-spsa/spall_an_overview.pdf) motivates evaluating paired parameter perturbations using full loss measurements. The current optimizer is a small antithetic direct search, not a reproduction of the full SPSA algorithm.
- [Pinned fafbseg named geometry](https://github.com/navis-org/fafbseg-py/blob/d0da95123ee606e204ae2c702e7bc78538646fbd/fafbseg/data/JFRC2NP.surf.fw.zip) supplies the 78 region names and surfaces together. Its SHA256 is hard-coded and checked before rendering.
