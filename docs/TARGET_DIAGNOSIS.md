# Inspect target-side information

The human-feedback candidate did not improve the reserved gameplay comparison. Before collecting more action labels or changing the policy again, this diagnostic asks whether target-side information is accessible at successive stages of the visual pathway.

## Open the completed inspection

On this prepared machine, open the saved page without starting another simulation:

```powershell
Start-Process .\runs\target-side-probe-v2\index.html
```

Select **Validation episodes**, then use **Next** or the slider. The large image shows the recorded Doom frame and a green scoring box. The smaller grid shows the actual 64 brightness values delivered to the fly graph. Below them are separate diagnostic LEFT/RIGHT predictions. These are not the playing model's actions. Near-center and repeated images remain inspectable but are excluded from accuracy calculations.

If a browser restricts local file access, serve the same folder:

```powershell
.\.venv\Scripts\python.exe -m http.server 8799 --bind 127.0.0.1 --directory runs/target-side-probe-v2
```

Keep that terminal open and visit **http://127.0.0.1:8799**.

## What the experiment measures

The three representations are:

| Stage | Input to diagnostic classifier | Dimensions |
|---|---|---:|
| Color comparison | Mean RGB channels in 8 by 8 spatial bins | 192 |
| Actual fly input | Existing `encode_frame` brightness bins | 64 |
| Fly output | Descending-cell voltage and spike-rate features | 2,606 |

The Laya observation description contains a blue-pixel heuristic, while the fly encoder averages color channels into brightness. The RGB comparison checks whether keeping color helps this particular diagnostic; it does not change the live encoder or use the heuristic as ground truth.

ViZDoom supplies the visible Cacodemon bounding box through its [labels interface](https://vizdoom.farama.org/main/api/python/gameState/). Its horizontal center supplies LEFT below 47.5% of the screen width and RIGHT above 52.5%. Centered, absent, or ambiguous targets are excluded from the binary task. Privileged object information supplies only the scoring target. The fly simulator receives RGB pixels through its unchanged brightness encoder, and the diagnostic classifiers receive only their named feature arrays.

Twenty-four predetermined seeds, 71000–71023, provide four observations each. The first sixteen episodes form the diagnostic training set; the last eight are validation. The character follows prescribed alternating strafe actions, mirrored on alternate episodes, with 8, 16, and 8 game tics between observations. It does not shoot. The graph retains its state between the four observations and resets at each episode boundary. These trajectories are different from autonomous play.

Each representation gets a separate small linear classifier. This trains measurement tools, not the playing student or biological connection weights. NumPy solves a class-balanced ridge system with a fixed penalty of 0.1. Feature means and scales come only from training examples. The kernel is normalized by feature dimension. No validation-based hyperparameter search is performed. Twenty shuffled-training-label repetitions provide a descriptive control, not a significance test.

Exact RGB repetitions are removed, giving priority to training over validation. Balanced accuracy averages recall for LEFT and RIGHT, so always guessing one side scores 50% when both classes are present. The report also records ordinary accuracy, class counts, confusion matrices, and exact cross-split feature matches. Related backgrounds and nearby scenes can still remain after exact deduplication.

## Reproduce the experiment

```powershell
.\.venv\Scripts\python.exe -m flydoom.target_probe --output runs/my-target-probe
```

Use a new output directory. It takes several minutes to run the frozen graph on all 96 observations. Existing NumPy, ViZDoom, and simulation dependencies are sufficient; no package installation is needed. A plan is saved before collection. The finished folder contains images, observations, feature arrays, a measured report with checksums, and the interactive page. Existing policy checkpoints and protected runtime files are hashed before and after the experiment.

A good output decoder means the measured features contain accessible side information in this sample. It does not establish that the current trained readout uses that information effectively. A poor linear decoder does not prove that all information was lost. This experiment cannot establish game success, generalization to other maps, physiological validity, or an advantage of biological topology over a control network.
