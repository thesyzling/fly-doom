# Fly Doom

A research project to train a model based on the real fruit fly connectome to control Doom.

**Status: the experimental neural game loop and a staged local learning pipeline are implemented.** Laya and CUDA PyTorch are installed, the local model is downloaded, and engineering gain checks have passed. Human demonstrations are still needed before training a Doom policy. Natural visual transmission and physiological calibration remain unresolved. Start with the [learning guide](docs/LEARNING.md); see also the [bridge guide](docs/BRIDGE.md), [research plan](docs/RESEARCH.md), and [simulation assumptions](docs/SIMULATION.md).

Current tools include data preparation, annotation matching, an experimental neural simulator, stimulation diagnostics, recorded activity playback, a random Doom baseline, a bounded neural game controller, human demonstration recording, local Laya adaptation, and training of an additional spiking neuron group.

Project documentation, source comments, docstrings, and application messages are written in English. Upstream research data retains its original contents.

## Start here

On the prepared machine, run this from the project directory to record your play and begin the learning pipeline:

```powershell
.\.venv\Scripts\python.exe -m flydoom.learning run
```

Press **ENTER** before each of 10 episodes. Use **LEFT/RIGHT** or **A/D** to move and **SPACE** to fire. Try to align with the target and shoot. **ESC** stops recording and keeps completed episodes. When recording finishes, the command extracts neural features and trains the Laya decision head. It trains the additional 64 spiking cells only if Laya passes the validation gate, then prints the command for watching the student play. Training is an experiment; a failed gate is a possible result. See [LEARNING.md](docs/LEARNING.md) for timing, separate commands, and what each stage means.

The experimental bridge implements the following loop:

```mermaid
flowchart LR
    A[Doom screen] --> B[Convert pixels into neural input]
    B --> C[Compute activity across fly connections]
    C --> D[Select a game action from neural activity]
    D --> A
```

The first working adapter converts screen brightness into drive for positive-sign visual projection cells and reads descending spikes to select actions. Image-to-cell and cell-to-action assignments are explicit engineering conventions, not known biological mappings. The learning pipeline instead trains an additional group from descending voltages and spike rates, with the fly graph frozen and Laya used as a teacher during training.

The local dataset contains **139,255 neurons**, **15,091,983 directed neuron pairs**, and **54,492,922 synaptic contacts**. The prepared sparse connectivity matrix occupies approximately **173 MiB** in memory; this excludes simulation and training memory. Every neuron matches an annotation row, although some annotation fields are missing. See the [validation record](docs/VALIDATION.md).

A **neuron** is a nerve cell that receives and transmits signals. A **synapse** is a communication site between cells. A **connectome** describes which neurons are connected. A **simulation** computes how activity changes over time along those connections. **Training** changes selected model parameters using experience from the game. These are separate steps.

### What each file does

| File or directory | Purpose | What to know now |
|---|---|---|
| `README.md` | The project guide you are reading | Start here |
| `docs/RESEARCH.md` | Sources, design decisions, and experiment plan | Read for scientific and technical context |
| `docs/VALIDATION.md` | What has been run and verified | Progress and evidence record |
| `flydoom/data.py` | Downloads and checks data, builds the graph, and matches annotations | The current data preparation code |
| `flydoom/doom_smoke.py` | Sends random actions to Doom to check the integration | Not a fly model or training procedure |
| `flydoom/bridge.py` | Encodes images and decodes neural spikes | Fixed, artificial mappings; no learned weights |
| `flydoom/brain_doom.py` | Runs the neural model and Doom together | Start here to watch the experimental neural controller |
| `docs/BRIDGE.md` | Explains the adapter, controls, timing, and limitations | Guide to the first game connection |
| `flydoom/learning.py` | Starts recording, preparation, teacher training, and student training | Run `python -m flydoom.learning run` |
| `flydoom/learning_data.py` | Records your buttons and replays images through the graph | Keeps episodes in separate learning splits |
| `flydoom/laya_teacher.py` | Downloads Laya and adapts its decision head | Frozen text encoder; supervised imitation |
| `flydoom/student.py` | Trains and runs the extra 64 spiking cells | Uses Laya during training, without loading it during play |
| `flydoom/calibration.py` | Selects a smaller global synaptic gain | Engineering stability checks, not biological validation |
| `docs/LEARNING.md` | Explains the learning experiment and commands | Read before the first recording |
| `flydoom/simulation.py` | Loads the graph and computes changing voltages and spikes | Experimental model with explicit assumptions |
| `flydoom/brain_probe.py` | Runs four stimulation and control experiments | Checks signal transmission before game integration |
| `flydoom/viewer.py`, `flydoom/web/` | Local browser viewer for recorded trials | Play recorded activity, scrub time, and inspect neurons |
| `docs/SIMULATION.md` | Explains equations, sign rules, and measured limitations | Guide to the first neural model |
| `flydoom/__init__.py` | Marks the directory as a Python package | No manual changes needed |
| `tests/test_data.py` | Checks the data code against small examples with known answers | Automated checks that catch mistakes |
| `tests/test_simulation.py` | Checks dynamics, delays, reset, and controls | Numerical model tests |
| `pyproject.toml` | Project metadata and required Python libraries | The project's dependency list |
| `requirements.lock.txt` | Exact installed library versions | Recreates the package versions used here |
| `requirements-training.lock.txt` | Exact training environment versions | Windows, Python 3.13, CUDA 13.0 PyTorch build |
| `models/laya-base/` | Downloaded local Laya checkpoint and checksums | Large model files, excluded from Git |
| `data/raw/` | Original files downloaded from the researchers | No need to open them in a code editor |
| `data/processed/` | Data prepared for our program | Inputs for later computations |
| `runs/` | Results from game runs | Experiment output |
| `.venv/` | The project's Python environment and libraries | Automatically managed |
| `.uv-cache/`, `.pytest_cache/`, `__pycache__/` | Installation and runtime caches | No need to inspect them now |
| `.gitignore` | Defines files excluded from Git tracking | Excludes large data and temporary files |
| `_vizdoom.ini` | Local settings generated by the game engine | Automatically generated |

In a fresh checkout, `data/` and `runs/` appear when their corresponding commands run. File extensions also help: `.py` is Python code, `.md` is documentation, and `.json` stores structured records. `.npy`, `.npz`, and `.feather` store numerical data for efficient program access.

### Tools we installed and why

We used the existing Python 3.13.11 installation on this machine. The existing `uv` tool created `.venv` and installed the project dependencies there.

| Tool or library | Role in this project | Why we use it |
|---|---|---|
| Python | Runs our scripts | Provides a shared environment for data processing and the game interface |
| `uv` | Creates the environment and installs packages | Keeps dependency installation and management straightforward |
| NumPy | Stores and processes arrays of neuron IDs and counts | Supports efficient array operations and exact 64-bit integer IDs |
| SciPy | Builds and stores the sparse connectivity matrix | Stores existing connections without allocating every possible neuron pair |
| PyArrow | Reads the research archive's Feather table | Reads the source format and selects the columns we need |
| ViZDoom | Provides the Doom engine interface | Lets Python read screen frames and send game actions |
| pygame-ce | Displays the demonstration window and reads keyboard input | Lets you provide examples by playing |
| Laya | Scores the four possible actions from text descriptions of coarse images | Provides the local teacher we can specialize |
| PyTorch | Computes gradients and updates selected weights | Trains the Laya head on the GPU and the small spiking group on the CPU |
| Transformers and Hugging Face Hub | Load the text encoder, tokenizer, and downloaded checkpoint | Supply Laya's pretrained components |
| Safetensors | Stores numerical model weights | Saves and reloads the teacher head and student |
| pytest | Runs automated checks | Detects data preparation errors using known examples |

The lock files also contain transitive dependencies such as Gymnasium. Standard-library modules such as `pathlib`, `hashlib`, `json`, `csv`, and `urllib.request` come with Python and did not require separate installation. Training dependencies are optional for the original simulator and viewer; the learning setup is documented separately below.

### Understanding connectivity data

A fictional example for learning:

| Source neuron | Target neuron | Synapse count |
|---|---|---:|
| A | B | 3 |
| B | C | 2 |

The first row means there are three synaptic contacts from A to B. It does not imply a connection from B to A. Three contacts also do not establish an electrical effect exactly three times as strong.

`data.py` converts this list into a matrix suitable for computation. We use `A[target, source]`: the entry in row B and column A is 3. Most neuron pairs are not directly connected, so we avoid storing all the zero entries. This is a **sparse matrix**. It retains every neuron while reducing memory use.

After downloading an archive file, its digital fingerprint, or **checksum**, is compared with the publisher's value. This checks that the file matches the archive; it does not establish biological validity. Preparation writes `manifest.json` to record the sources and neuron, connection, and synapse counts.

### Reading a command

```powershell
.venv/Scripts/python.exe -m flydoom.data prepare
```

- `.venv/Scripts/python.exe`: run the Python interpreter in this project's environment.
- `-m flydoom.data`: run the `data.py` module in the `flydoom` package.
- `prepare`: select the data preparation operation. Use `download` to select downloading or `annotate` to match neuron annotations.

Run commands in a PowerShell terminal opened in the project directory. There is no need to repeat a command that has already completed successfully unless you intend to rerun that step.

## Setup (PowerShell)

The following is the **base environment** setup for a fresh checkout. Syncing its lock file removes optional training packages. For the learning environment, use the [training setup instructions](docs/LEARNING.md#setup-on-a-fresh-machine).

Requires Python 3.11+ and `uv`. Tested on this machine with Python 3.13.

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv venv .venv
uv pip sync --python .venv/Scripts/python.exe requirements.lock.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m flydoom.doom_smoke --episodes 3
```

The random-policy report is written to `runs/doom-smoke.json`. It is explicitly marked `trained: false` and `connectome_loaded: false`. The game runs without a visible window; this command does not start training.

### Watch the game window

To watch the random controller play the basic target-shooting scenario:

```powershell
.venv/Scripts/python.exe -m flydoom.doom_smoke --visible --episodes 5 --log-actions --output runs/my-visible-doom.json
```

This opens a separate ViZDoom window with a 640 x 480 view and HUD. It does not appear inside the brain viewer. The controller randomly chooses `WAIT`, `MOVE_LEFT`, `MOVE_RIGHT`, or `ATTACK`; the terminal prints each decision. The basic scenario allows sideways movement and shooting, not free navigation through a full Doom level. Episodes can end early when the target is hit, so the whole demo may be short. The window closes after the requested episodes finish.

The visible mode advances one game tic at a time, paced around the normal 35-tic-per-second rate. Add `--speed 0.5` to watch at half speed. This changes wall-clock presentation, not the controller's decision frequency in game time. Rendering or OS scheduling can make playback slower than requested. Sound remains disabled. Press Ctrl+C in the terminal to stop; interrupted runs do not write a completed report. You do not need to steer with your keyboard: the random controller sends the actions.

**This command still uses random control.** The separate neural controller below connects the graph to the game. The random report records the presentation settings separately and keeps `trained: false` and `connectome_loaded: false`. Implementation uses the existing ViZDoom interface; see its [game control documentation](https://vizdoom.farama.org/api/python/doom_game/).

### Connect the neural model to Doom

With the prepared graph and annotations already present, run:

```powershell
.venv/Scripts/python.exe -m flydoom.brain_doom --visible
```

After loading the graph, a separate Doom window opens. Every decision prints `L`, `R`, and `A`: mean spike rates per output neuron assigned to left, right, and attack. The largest rate selects the action; silence or a tied maximum selects `WAIT`. There is no random-action fallback. These groups were assigned by us, not identified as biological Doom controls.

The default run stops after **24 decisions or an earlier game ending**, not necessarily a complete episode. It computes 50 ms of neural activity per decision and applies the chosen action for up to 4 game tics. The CPU needs approximately 1.3 seconds per decision on this machine, during which the game holds its current frame. This is expected computation time. The neural and game clocks are separate experimental time scales.

Each run creates a new timestamped `runs/brain-doom-.../` folder with `report.json`, `mapping.json`, and a per-decision `decisions.jsonl` trace. Ctrl+C during play saves partial results. To preserve an explicitly named run, use `--output runs/my-neural-game`; an existing directory is rejected. The existing browser viewer still displays separate probe recordings, not live Doom activity.

**This is an untrained engineering bridge that bypasses the retina.** It is not evidence of natural fly vision or competent play. Nothing new needs installing: NumPy encodes pixels, SciPy carries signals through the sparse graph, and ViZDoom reads frames and applies actions. See [the step-by-step bridge guide and control commands](docs/BRIDGE.md).

## Real connectivity data

The two connectivity archive files require approximately 853 MB of downloads. Processing needs additional disk space and RAM; annotations are downloaded separately by `annotate` if missing.

```powershell
.venv/Scripts/python.exe -m flydoom.data download
.venv/Scripts/python.exe -m flydoom.data prepare
.venv/Scripts/python.exe -m flydoom.data annotate
```

Source: [FlyWire publication archive](https://zenodo.org/records/10676866). The downloader checks published MD5 values. Preparation records source and output SHA-256 values and neuron, edge, and synapse counts in `data/processed/fafb783/manifest.json`.

`annotate` downloads the [authors' v2.1.0 annotations](https://github.com/flyconnectome/flywire_annotations/tree/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0) if necessary, checks a pinned SHA-256 value, and orders rows by neuron identity. The outputs are `neuron_annotations.tsv` and `annotations_manifest.json`. A `.tsv` file is a text table with tab-separated columns. `top_nt` is the predicted neurotransmitter; missing values remain missing.

The prepared matrix uses `A[target, source]`. Its weights are **unsigned synapse counts**, not physiological strengths. Neuron IDs remain exact integers. The experimental simulator builds a separate signed weight matrix using the explicit assumptions in [the simulation guide](docs/SIMULATION.md).

## First neural simulation

```powershell
.venv/Scripts/python.exe -m flydoom.brain_probe
```

This CPU experiment uses existing NumPy and SciPy dependencies. It stimulates selected cells for a short period and compares connected and disconnected conditions. Results go to `runs/brain-probe/report.json`. No Doom pixels, learned policy, or training are involved. The first run showed downstream voltage responses, but photoreceptor stimulation did not produce descending spikes, and some voltage excursions were excessive. See [the full results and next steps](docs/SIMULATION.md).

## View the brain in your browser

After a completed probe, run:

```powershell
.venv/Scripts/python.exe -m flydoom.viewer
```

Keep the terminal open and visit **http://127.0.0.1:8765**. The default viewer reads the existing `runs/brain-probe` experiment. To view a different completed experiment:

```powershell
.venv/Scripts/python.exe -m flydoom.viewer --run-dir runs/my-first-probe
```

An interrupted probe without `report.json` cannot be displayed. Complete it first. If port 8765 is in use, pass `--port 8766` and open the corresponding address. Press Ctrl+C in the terminal to stop the viewer.

Choose a condition from **Experiment**. Drag the map to rotate, scroll to zoom, and click a point to inspect the cell. The **Show** filter can isolate spiking, directly stimulated, or descending cells. Cyan identifies stimulated spiking cells, orange other spiking cells, and pink spiking descending cells. Gray cells emitted no spikes; they may still have changed voltage.

### Play activity over time

New probes record per-neuron spike counts in successive intervals and voltage snapshots, including the initial resting state. To create your own replay, stop the old viewer with Ctrl+C and run these commands in order:

```powershell
.venv/Scripts/python.exe -m flydoom.brain_probe --output runs/my-replay
.venv/Scripts/python.exe -m flydoom.viewer --run-dir runs/my-replay
```

Wait for the probe to finish all four conditions before starting the viewer. The terminal prints simulation progress. Open http://127.0.0.1:8765 and reload the page. A temporal recording starts playing once on load; use **Pause**, **Play recording**, **Replay**, or the **Simulation time** slider to explore it. Selecting a different experiment starts its recording. Clicking a neuron pauses playback for inspection.

**Color by → Voltage change** shows endpoint voltage relative to rest: blue below rest, orange above rest, gray near rest. Color intensity saturates at ±20 mV for visibility, while actual values remain available in the inspector. This color mapping does not clip simulation voltages. **Spikes per interval** uses the cyan/orange/pink spike colors described above and shows counts since the previous saved sample. The cells remain in place; their activity changes. Default recordings sample every 10 ms, with each saved sample displayed for 400 ms of viewing time (about 40 times slower than simulated time). There is no interpolation. The final frame stays visible until replayed.

The sidebar totals and bars summarize the whole trial; the map and trace cursor follow the selected time. The stimulus indicator reports whether input was applied during the selected interval. These are recorded results, not a live simulation. Older runs with only total counts remain viewable but show disabled playback controls and instructions to create a new recording.

The map uses all 139,255 real neuron anchor positions, converted from the authors' 4 x 4 x 40 nm voxel coordinates to micrometers before display normalization. These are reference points on neurons, not their full shapes, somas, synaptic locations, or connections. Source scan orientation is retained; anatomical left/right is not inferred from screen position. See the [authors' coordinate definitions](https://github.com/flyconnectome/flywire_annotations/blob/ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0/supplemental_files/Supplemental_files_columns.md).

**Playback uses measured per-neuron recordings; it is not live simulation.** Legacy files without temporal data show whole-trial totals only. Group bars summarize annotated cell superclasses, not anatomical brain regions or inferred mental functions. The viewer uses Python's built-in HTTP server and the browser's WebGL support, with no new packages, CDN, or external web service. It listens only on the local machine and verifies data/result hashes before serving results.

### Windows test directory permissions

If pytest reports access denied for its temporary or cache directory, use a fresh temporary location and disable the cache:

```powershell
$testTemp = Join-Path $env:TEMP ("fly-doom-tests-" + [guid]::NewGuid().ToString("N"))
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp "$testTemp"
```

This still runs every test. Use the fresh generated path: pytest manages the directory passed to `--basetemp` as disposable test storage.

Large data files, the environment, and run outputs are excluded from version control. Third-party data and game assets retain their own licenses; this repository does not relicense or bundle them.
