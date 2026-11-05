# RUN_MECHANISM.md

Exact commands for this round's additions: the 1500-clock dataset, the
teacher-forced logit metric (folded into every Experiment A mode below),
`--position_mode text_all`, `--component mlp`, and `knockout.py`. All of
this is Qwen2.5-VL-3B only for now. Every command below has been tested
offline against fake models (see the scratchpad test files referenced
inline); none of it has been run against real weights yet.

**Order matters**: run `--verify` for a mode before its full sweep, every
time. `--verify` prints PASS/FAIL for each check and exits with a nonzero
code on any FAIL -- it does not run the sweep.

## 1. Generate the 1500-clock dataset

```
python clocks.py --n_clocks 1500
```

Writes to `data_balanced_1500/` (25 images per minute value, 0-59). Same
generator as `data_balanced/`, just a bigger balanced set. The pair builder
(`intervene.py`'s `build_pairs`, unchanged) works on this directly -- point
`--data_csv`/`--images_dir` at it in every command below.

## 2. Logit metric (folded into Experiment A -- no separate command)

The teacher-forced `logit_diff` column is computed automatically for every
Experiment A trial (default sweep, `--position_mode text_all`, and
`--component mlp` all include it) -- there's nothing extra to run. Verify
it once, standalone, before trusting it on a real sweep:

```
python intervene.py --experiment none --verify --verify_pairs 3 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500
```

This runs the existing vision/decoder-patch checks AND (since
`--skip_logit_metric` isn't passed) `verify_logit_metric`: the
teacher-forced log-prob of the model's own greedy answer must match
`generate(output_scores=True)`'s own per-step scores within 1e-3, for a
few images. If this fails, don't trust `logit_diff` anywhere else in this
round's outputs.

To skip the logit metric entirely on any run below (cost control -- it
adds 2 extra forward passes per trial, 2 per pair), add `--skip_logit_metric`.

## 3. `--position_mode text_all`

```
# verify first (3 checks, exits -- does not sweep)
python intervene.py --experiment a --position_mode text_all --verify \
    --verify_pairs 3 --verify_layers 10,20 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500

# smoke test
python intervene.py --experiment a --position_mode text_all \
    --layers 16,18,20,22,24 --max_pairs 2 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500

# full run (readout-window layers; same --n_pairs budget logic as the
# default sweep applies here -- see heads.py's power-note precedent if
# you want statistically powered numbers, not just a screening pass)
python intervene.py --experiment a --position_mode text_all \
    --layers 14,16,18,19,20,21,22,23,24,26 --n_pairs 20 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500
```

Writes to `intervene_output_textpos/<model_short_name>/experiment_a_trials.csv`
(default `--out_dir`, unless you override it).

## 4. `--component mlp`

```
# verify first (3 checks, exits -- does not sweep)
python intervene.py --experiment a --component mlp --verify \
    --verify_pairs 3 --verify_layers 10,20 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500

# smoke test (remember: layer 0 and the FINAL layer are refused --
# mlp_module_for_layer raises a clear error if you pass them)
python intervene.py --experiment a --component mlp \
    --layers 16,18,20,22,24 --max_pairs 2 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500

# full run
python intervene.py --experiment a --component mlp \
    --layers 14,16,18,19,20,21,22,23,24,26 --n_pairs 20 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500
```

Writes to `intervene_output_mlp/<model_short_name>/experiment_a_mlp_trials.csv`.

## 5. `knockout.py`

```
# verify first (5 checks, exits -- does not sweep)
python knockout.py --verify --verify_images 3 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500

# smoke test
python knockout.py --max_images 10 --start_layers 0,20,34 \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500

# full run (all 11 start layers, the full 1500-image set)
python knockout.py \
    --data_csv data_balanced_1500/data.csv --images_dir data_balanced_1500
```

Writes to `knockout_output/<model_short_name>/knockout_trials.csv` and
`knockout_summary.csv` (per-start-layer bootstrap CIs + a permutation test
against the negative control, via `analyze_replication.py`'s
`bootstrap_ci`/`permutation_test`, reused unmodified).

**Cost note**: `knockout.py` runs the FULL 1500-image set x 11 start layers
= 16500 `generate()` calls by default (plus 16500 more forward passes for
the logit metric, unless `--skip_logit_metric`) -- size `--max_images`
and/or `--start_layers` down for a first real-hardware timing check before
committing to the full run, the same way every other script in this
project recommends (`heads.py`'s `--max_pairs`, `intervene.py`'s
`--max_pairs`).

## Offline test coverage (this round)

All of the above was tested against fake models before being described
here, not just written and assumed correct:
- `test_logit_textpos_mlp.py`: teacher-forced logit metric (hand-computed
  cross-check + `generate(output_scores=True)` agreement),
  `--position_mode text_all`'s 3 verify checks, `--component mlp`'s
  off-by-one-safe layer indexing and 3 verify checks, and
  `run_experiment_a_mlp` end to end.
- `test_knockout.py`: `force_eager`, the knockout hook's exact-zero/
  byte-identical-elsewhere behavior against a REAL (if tiny) eager
  attention implementation, and all 5 of `run_verification`'s checks.

Both are in the scratchpad, not the repo (per this project's established
practice for offline-only test harnesses) -- see the conversation this
round came from if you need to re-run them locally without a GPU.
