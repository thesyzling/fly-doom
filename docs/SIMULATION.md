# First neural simulation: assumptions and experiments

Date: September 27, 2026. This is a diagnostic implementation of a uniform current-based leaky integrate-and-fire (LIF) network. It is not a validated reconstruction of fly physiology, a natural visual system, or a trained Doom player.

## What was added

- `flydoom/simulation.py`: verified graph loading, provisional transmitter signs, and time-stepped neuron dynamics.
- `flydoom/brain_probe.py`: four bounded stimulation experiments and reproducible reports.
- `tests/test_simulation.py`: analytic voltage checks, signal direction and delay, inhibition, refractoriness, reset, sign rules, and disconnected controls.

These use the existing NumPy and SciPy installation. No additional dependency or GPU framework was installed. The first benchmark runs on the CPU.

## From a wiring diagram to a changing state

The data provides synapse counts `C[target, source]`. A neuron also needs a changing state: here, a membrane voltage `v`, a decaying synaptic drive `g`, and a refractory timer. A threshold crossing emits a spike. That spike reaches connected cells after a delay.

Our subthreshold equations are:

```text
dv/dt = (rest - v + g + external_drive) / membrane_tau
dg/dt = -g / synapse_tau
W[target, source] = C[target, source] * source_sign * mv_per_contact
```

Voltage and drive are expressed in mV-equivalent units; `external_drive` is not a physical current in amperes. Each step integrates these linear equations analytically while external drive remains constant. Incoming spikes add `W @ arriving_spikes` to `g` at the start of an interval. Threshold crossings are detected at its end. Threshold timing therefore remains discretized even though subthreshold integration is analytic.

After a spike, voltage resets. It remains clamped during complete refractory intervals, while synaptic drive continues receiving inputs and decaying. Resetting a network clears voltage, drive, timers, and pending spikes so that trials cannot leak state into one another.

Parameters are uniform across all neurons:

| Parameter | Value |
|---|---:|
| Rest / reset voltage | -52 mV |
| Spike threshold | -45 mV |
| Membrane time constant | 20 ms |
| Synaptic drive time constant | 5 ms |
| Requested refractory duration | 2.2 ms |
| Requested transmission delay | 1.8 ms |
| Weight per synaptic contact | 0.275 mV |
| Simulation step | 0.5 ms |

Delay and refractory duration are rounded up to whole steps: effective values are 2.0 ms and 2.5 ms. A spike emitted at time `t` arrives at `t + effective_delay`; integration in the receiving cell begins then. Voltage is clamped for the full effective refractory duration, with integration resuming in the following interval.

The numerical parameter starting points are informed by [Shiu et al.'s reference implementation](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py). Our code is independently implemented and differs in numerical scheduling, input method, transmitter handling, and refractory synaptic behavior. We use deterministic external drive, not their Poisson stimulation. This is not a replication. The distinction between spike suppression and state clamping is also described in the [Brian 2 refractoriness documentation](https://brian2.readthedocs.io/en/stable/user/refractoriness.html).

## Transmitter signs are explicit assumptions

The original unsigned graph is unchanged. A separate float32 matrix uses a provisional presynaptic sign:

1. `R1-6`, `R7`, and `R8` use an inhibitory visual-channel approximation, overriding automated predictions. Histamine-mediated inhibition in fly visual pathways motivates this choice; see [Alejevski et al., Nature Communications 2019](https://www.nature.com/articles/s41467-018-08116-7). It is a modeling rule, not a new annotation in the source data.
2. Otherwise, curated `known_nt` labels take priority when their supported fast transmitters agree in sign. Acetylcholine maps to +1; GABA, glutamate, and histamine map to -1. These are provisional uniform rules, not receptor-specific measurements.
3. Conflicting or unsupported curated identities produce zero output. If curated labels are absent, supported `top_nt` predictions provide the same provisional signs.
4. Unknown or modulatory-only predictions produce zero output. No dopamine learning rule is implemented.

The photoreceptor rule omits cotransmission and target-specific signaling, including pathways discussed in [a study of R8 cotransmission](https://www.nature.com/articles/s41586-023-06681-6). Normally graded visual cells are also represented by LIF units here. Together with absent tonic activity and a natural retinal encoder, these are substantial visual-system limitations.

The sign audit reports 85,530 positive-output neurons, 51,454 negative-output neurons, and 2,271 neurons with silenced outputs. All 139,255 neurons and 15,091,983 structural edges remain represented; 14,744,809 edges have nonzero effective weights. Retaining the structural graph does not mean that every connection transmits in this approximation. Explicit zeros remain stored for auditability.

## Running the experiment

From the project directory:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m flydoom.brain_probe
```

The probe loads and verifies the prepared graph and annotations. Each condition begins from a fresh resting state and lasts 200 ms of simulated time. Nonempty stimulation runs use a 20 mV-equivalent pulse from 20 to 80 ms. No game frames or game state are used.

The output directory `runs/brain-probe/` contains `report.json` and one `.npz` per condition. Reports include parameter values, source and output hashes, transmitter rules, versions, timing, voltage summaries, and response traces. Each new `.npz` records per-neuron total spike counts, exact stimulated IDs, `sample_times_ms`, `spike_bins`, and `voltage_mv`. Temporal arrays have shape `[samples, neurons]`; the initial frame has zero spikes and resting voltages. Each later spike bin covers the interval since the preceding sample, while voltage is sampled at the interval endpoint after any reset. Sampling uses the first full step at or above 10 ms, plus the final step. At default settings there are 21 samples including time zero. Spike bins sum exactly to each neuron's total. This is not a record of individual spike timestamps or of all voltage extrema between samples.

The optional capture adds memory for recording arrays, which is excluded from `persistent_network_mib`. The browser can replay these saved samples, including subthreshold voltage responses, without rerunning the model. Repeating a run in the same output directory replaces its outputs; use `--output runs/another-probe` to preserve a separate experiment.

## First results

The same 1,303 neurons annotated `super_class == descending` serve as a diagnostic output population. The standalone probe does not assign them to game buttons. The later [engineering bridge](BRIDGE.md) uses an explicitly artificial assignment.

| Condition | Directly stimulated neurons | Spikes outside stimulated cells | Descending spikes | Other cells with voltage response |
|---|---:|---:|---:|---:|
| No input | 0 | 0 | 0 | 0 |
| R1-6 pulse | 8,452 | 0 | 0 | 5,897 |
| R1-6 pulse, recurrent transmission disabled | 8,452 | 0 | 0 | 0 |
| Positive visual projection pulse | 6,878 | 17,112 | 1,290 | 111,095 |

The R1-6 pulse caused 42,260 spikes in the stimulated cells in both connected and disconnected conditions. A voltage response is an absolute deviation greater than 0.0001 mV from rest. The connected pulse produces inhibition downstream, but no downstream spikes in this resting, simplified model. The excitatory control bypasses the retina; its success does not demonstrate vision.

All four bounded runs remained finite. However, the connected photoreceptor condition reached about -125.1 mV, and the excitatory control reached about -425.4 mV. The latter retained a maximum deviation of about 84.1 mV at the end. These large excursions are a model calibration warning: being finite does not imply physiological plausibility, recovery to rest, or long-term stability. We did not clip voltages or hide these observations.

At the default step size, connected conditions took approximately 12.5–13.0 ms of CPU compute per 0.5 ms simulated step, excluding graph loading and reporting. This is about 25–26 times slower than simulated real time. Timing is machine- and load-dependent. The persistent matrix, state, and delay buffer occupy about 119.9 MiB; this excludes annotation objects, temporary arrays, loading peaks, and interpreter overhead. It is smaller than the earlier count matrix partly because simulation weights use float32 instead of int64 counts.

## Milestone status

The numerical kernel and bounded transmission diagnostics are implemented. M2 remains incomplete: photoreceptor-to-descending spiking, natural visual encoding, physiological calibration, and adequate runtime have not passed. A later [bounded engineering bridge](BRIDGE.md) attaches Doom pixels directly to visual projection cells, bypassing the unresolved retinal pathway. It does not validate these missing model properties. Baseline activity, graded visual signaling, stimulation, and synaptic effects still need calibration before validated training.
