# First neural connection to Doom

This adapter tests whether real game pixels can drive the existing LIF model and whether its output can determine game buttons. It does not train weights, reconstruct natural fly vision, or validate biological motor functions. The previous photoreceptor-to-descending test failed to produce output spikes, so this bridge explicitly bypasses the retina.

## Follow one decision

1. **Read the screen.** ViZDoom supplies a 320 x 240 RGB frame. The HUD is disabled in both visible and headless modes. The weapon remains in the image. Only this frame enters the controller; reward, ammunition, enemy coordinates, labels, and other game variables do not enter action selection.
2. **Encode pixels.** `encode_frame` in `flydoom/bridge.py` averages the three color channels, divides by 255, and averages each of 64 spatial bins on an 8 x 8 grid. Values range from 0 for black to 1 for white. This is a deliberately simple brightness encoder, not object recognition.
3. **Drive input neurons.** Select the 6,878 `visual_projection` cells with positive model transmitter signs. Sort by exact integer root ID and cyclically assign them to the 64 image bins. Multiply the assigned brightness by a default 40 mV-equivalent input gain. The assignment has no claimed relation to visual receptive fields or anatomical location. No descending output cell receives direct pixel drive.
4. **Compute neural activity.** Keep all 139,255 neurons and the existing signed sparse connectivity. Hold this drive constant for 50 ms of neural time, using 100 steps of 0.5 ms. Voltage, synaptic current, delayed spikes, and refractory state persist between decisions. They reset to rest between episodes. No weights are updated.
5. **Read the output.** Sort the 1,303 `descending` cells by exact root ID, then cyclically assign them to LEFT, RIGHT, and ATTACK. Group sizes are 435, 434, and 434. These are arbitrary engineering groups, not discovered behavioral functions. Each score is spike count divided by group size and neural window duration in seconds. Pick the largest score. Silence or numerically tied maxima produce WAIT, with no random fallback.
6. **Act and repeat.** Send the selected button for up to four Doom tics. Read the resulting screen for the next decision. Reward is recorded after acting; it does not modify the controller.

The game calls follow ViZDoom's [official Python API](https://vizdoom.farama.org/api/python/doom_game/). The adapter uses synchronous `PLAYER` mode. The game waits during neural computation; visible mode displays individual action tics at approximately 35 tics per second. Neural time (50 ms per decision), game time (up to 4/35 seconds per decision), and wall-clock time are distinct. This is an engineering time mapping, not calibrated sensorimotor timing.

## Run and inspect

```powershell
.venv/Scripts/python.exe -m flydoom.brain_doom --visible
```

The initial graph verification takes time before the game window opens. The default experiment ends at 24 decisions unless the game finishes sooner. A `decision_limit` ending is a truncated trial, not a won or lost episode. The window closes at the end. Computation currently takes about 1.3 seconds per neural decision on this CPU; the pauses are expected. The command does not start the browser brain viewer.

For a shorter trial, add `--max-decisions 6`. Use `--episodes 2` to run two independently reset episodes with successive game seeds. `--seed` controls Doom randomness; mappings and action selection are deterministic. `--brain-ms` changes neural integration duration per decision, and `--input-gain-mv` changes external stimulation strength. Both alter the experiment, not just display speed. The default output path is unique; an explicit `--output` must name a directory that does not already exist.

The terminal line `L=7.91 R=7.51 A=7.05 Hz/cell` means that the three artificially assigned output populations averaged those firing rates over the decision window. LEFT wins this example. This measures modeled activity; it is not confidence, intent, or understanding of the target.

Each output folder contains:

| File | Contents |
|---|---|
| `report.json` | Control condition, model parameters, versions, graph provenance, output hashes, episode rewards, stopping reasons, and completion status |
| `mapping.json` | Exact input/output neuron IDs as strings, input-bin assignments, and mapping rules |
| `decisions.jsonl` | One JSON object per applied decision: 64 image features, group rates and counts, chosen button vector, neural time, game tics, reward, compute time, and voltage extrema after reset |

The trace is flushed after each applied decision. Ctrl+C during play produces an `interrupted` report and retains completed decision records. An interrupted decision may have partially advanced internal state or game tics and is not logged as complete. Setup failures before game initialization may leave a directory without a completed report. Reports do not store every neuron's trajectory or raw screenshots and are not compatible with the probe playback viewer.

## Controls

Run these separately. Add `--visible` to watch the expected waiting behavior:

```powershell
.venv/Scripts/python.exe -m flydoom.brain_doom --control disconnected --max-decisions 6
.venv/Scripts/python.exe -m flydoom.brain_doom --control zero-input --max-decisions 6
```

`disconnected` still delivers pixel-derived drive to input neurons but disables recurrent transmission. Starting from rest, disjoint output cells remain silent and select WAIT. `zero-input` still reads and records image features but applies zero drive; the connected model stays at rest and selects WAIT. The connected and control runs start with the same default game seed, but later frames can diverge after different actions. These are closed-loop controls, not matched-frame replays.

These checks establish that neural transmission is necessary for this adapter's actions. They do not establish an advantage from fruit-fly topology, target recognition, learning, or sensible play. A controller can repeatedly choose one action because the readout has not learned a task. No tuning has been performed to maximize game reward.

## What remains unresolved

Photoreceptor signaling, retinotopy, receptor-specific effects, and physiological calibration remain unresolved. The existing model can produce excessively negative voltages; the bridge reports these without clipping. Finite states in a short trial do not prove long-term stability. M2 remains incomplete, and this bounded integration test does not move the project into validated training.

Next work should measure image sensitivity with matched-frame controls, calibrate dynamics, and compare fixed and trained readouts under a separate evaluation protocol. Successful wiring of software components is the current result; a trained Doom-playing model remains the longer-term goal.
