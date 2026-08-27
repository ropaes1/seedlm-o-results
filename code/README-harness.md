# SeedLM+O: seeded pseudorandom bulk + exact outlier side-channel

Compression research: can LFSR-seeded random bases + an outlier
side-channel compete with real K-quants at ~3 bpw? Current answer: within
2.1x and closing (see
[../docs/experiment/EXPERIMENT.md](../docs/experiment/EXPERIMENT.md), the
status of record; read that first).

## Map

- **[../docs/experiment/](../docs/experiment/)**: `EXPERIMENT.md`
  (status of record, measured numbers, branch tree, decision rules) and
  `PROXY-ALIGNMENT.md` (the proxy-vs-KL analysis).

## Code (this directory)

`runner.py` (stages, CLI, selftest; run `--selftest` after any change),
`lfsr_core.py` (fit math incl. W1/W2/W3), `comparators.py`, `swap_eval.py`,
`salience.py`, `hf_sync.py` (laptop-to-pod transport via a Hugging Face
repo), `run_phase3_pod.sh` (pod launcher; pulls the harness from that
repo's `harness/` folder), `run_phase3_local.ps1` / `run_w1_local.ps1`
(legacy local launchers, kept for provenance; long c12p4-class runs were
done on rented GPUs).

Results are in this repo under
[../results/](../results/); `results/{slug}/summary.md` is the
human-readable verdict per model.
