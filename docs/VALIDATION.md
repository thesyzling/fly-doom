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
