# Learning from the student's own gameplay

This experiment records states reached by the existing student, asks the frozen Laya teacher for suggestions afterward, and trains the added 64-cell group on those suggestions. The original fly connections remain fixed. Game scores are evaluation outputs, not training targets. Laya's failed acceptance gate remains recorded: a disagreement is advice, not a verified mistake.

## Why collect another kind of recording?

Human recordings mostly show states reached while a human chooses the buttons. An autonomous student can make different moves and encounter situations missing from those recordings. Recording its own trajectories gives the teacher an opportunity to suggest recovery actions on those states. The teacher sees the actual student history; its suggestions never replace earlier recorded buttons.

The added workflow uses existing dependencies. ViZDoom supplies frames and executes buttons; NumPy stores observations; SciPy propagates activity through the sparse fly graph; Laya supplies probabilities over four actions; PyTorch adjusts the added group's weights; safetensors saves the resulting model. Nothing new needs installing on this prepared machine.

## Files and stages

| File | Purpose |
|---|---|
| `flydoom/correction_data.py` | Records student-controlled episodes, neural features, frames, and causal teacher observations |
| `flydoom/correction_training.py` | Merges recordings, queries the frozen teacher, and trains a separate candidate |
| `tests/test_corrections.py` | Checks history, provenance, split integrity, training, and checkpoint retention |

Each output directory must be new. Existing recordings and checkpoints are never overwritten. These recordings have their own schema and cannot be mistaken for human demonstrations.

1. **Collect.** The saved student controls Doom. The graph and student reset at episode boundaries. Episode roles and reserved evaluation seeds are declared before collection.
2. **Label.** Laya receives each saved observation and strictly past student actions. The recorded score, future frames, and current applied action are excluded from that input. Four action probabilities are saved with checksums. Verified probabilities from earlier identical human training observations can be reused.
3. **Train.** Start from the original student. Each update uses 32 human training observations and 32 student-collected training observations. Half the loss preserves human/teacher imitation; half follows the new teacher probabilities. Suggestions that differ from the recorded action receive three times the correction weight. Mean and scale remain frozen.
4. **Select.** Choose the checkpoint minimizing correction-validation KL plus 0.25 human-validation NLL. Both human validation accuracies must remain within five percentage points of the parent. The unchanged parent remains eligible at epoch zero. Human test metrics and game scores do not select the checkpoint.
5. **Evaluate.** Run the selected candidate and parent on the same reserved seeds and decision limit. Compare target kills first, then mean return. Keep disconnected and direct-policy controls separate. A small pilot is not evidence of general Doom competence.

KL measures the difference between predicted probability distributions; lower means closer agreement with the teacher. NLL penalizes low probability on recorded human actions. Neither metric guarantees better play, which is why the native game comparison is a separate step.

## Reproduce with new output directories

These commands can take several minutes because each decision runs the full neural graph. They reuse the local model and packages. The seed values reproduce development conditions; choose new, disjoint starts for a later assessment, and still check visual overlap.

```powershell
.\.venv\Scripts\python.exe -m flydoom.correction_data --output runs/my-student-observations --seed 56000 --train-episodes 4 --validation-episodes 2 --evaluation-seed 57000 --evaluation-episodes 6
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe -m flydoom.correction_training label --collection runs/my-student-observations --output runs/my-student-suggestions --human-cache runs/fly-student-balanced-v1
.\.venv\Scripts\python.exe -m flydoom.correction_training train --collection runs/my-student-observations --labels runs/my-student-suggestions --output runs/my-corrected-student --epochs 100 --learning-rate 0.00001
.\.venv\Scripts\python.exe -m flydoom.learning play --checkpoint runs/my-corrected-student --episodes 6 --seed 57000 --max-decisions 75 --headless --output runs/my-correction-evaluation
```

To combine multiple compatible collections before labeling:

```powershell
.\.venv\Scripts\python.exe -m flydoom.correction_training merge --collections runs/student-correction-observations-v1 runs/student-correction-recovery-observations-v1 --output runs/my-merged-observations
```

The merger preserves each episode's declared split and requires matching model, data, calibration, and reserved evaluation seeds. It rejects duplicate episode seeds.

## Visual overlap matters even with different seeds

During collection, seeds 56000, 56001, 56003, and 56004 produced identical opening frames and the same 18-decision trajectory. Distinct seeds alone do not guarantee distinct scenes in this basic scenario. Correction validation therefore excludes exact feature-vector plus causal teacher-state duplicates of either training source. This removes exact repeated inputs, not near duplicates or all related scenes. The older human validation split is reused as development data and is not certified visually independent by this filter.

The recovery collection deliberately reuses failed development starts 54000-54002 for training and 54003 for validation. Earlier scores on those starts are no longer independent evidence for this candidate. Reserved starts 57000-57005 are excluded from collection and checkpoint selection, but visual overlap must still be reported. See [the measured results](VALIDATION.md).

## Inspect a completed candidate

```powershell
.\.venv\Scripts\python.exe -m flydoom.live_brain --checkpoint runs/fly-student-corrected-v2 --port 8767
```

The browser starts paused. **Step once** advances one decision; **Run** plays autonomously. The observer displays actual simulated activity and learned weights. It does not update weights during play. The original checkpoint remains available independently.

The completed comparison kept five kills in six episodes for both models and improved mean return from -21.33 to -11.83. WAIT usage increased, so this is not a demonstrated solution to prolonged waiting. Direct Laya killed all six targets. The original student remains the default; the corrected candidate is available for inspection. Full results and limitations are recorded in [VALIDATION.md](VALIDATION.md).
