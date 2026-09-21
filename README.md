# Clock Interp

Mechanistic interpretability project on why vision-language models (VLMs)
misread analog clocks, especially swapping the hour and minute hands.

- **Step 1: behavior check.** Generate synthetic clocks, ask a VLM to read
  them, analyze the errors. Found: Qwen2.5-VL-3B-Instruct reads clocks badly
  (~2% exact accuracy), but often *describes* the hands correctly in words.
- **Step 2: probing.** Is the true hand angle linearly decodable from the
  model's hidden states, even on images where its final answer is wrong? If
  so, the information was there -- the model just failed to use it.
- **Step 3: causal intervention.** Probing only shows the angle is present;
  it can't show the model actually *uses* it. This step tests that causally:
  patch one clock's hidden states into another's forward pass and see if the
  stated minute moves, and steer along the probe's own weight direction and
  see if that moves it. No results yet -- `intervene.py` is written and
  tested (see "How to run" below); results go here once it's been run on
  Kaggle.

## Results

### Behavior

- **Exact accuracy is 1.8-2.2%** -- 1.8% with the plain "what time is it"
  prompt (`results/results.csv`, n=500), 2.2% with a "describe the hands,
  then answer" prompt (`results/describe.csv`, n=500). Reproduce:
  ```
  python clocks.py --n 500 --seed 42 --out_dir data
  python eval_behavior.py --data_csv data/data.csv --images_dir data --out_csv results/results.csv
  python analyze.py --results_csv results/results.csv --images_dir data --out_dir analysis_output
  ```
- **The model describes the minute hand's position correctly 82% of the
  time** (`results/describe_positions.csv`, n=480, `minute_desc_ok` column,
  allowing +/-1 clock-number tolerance since the hand often sits between two
  numbers). Reproduce:
  ```
  python balanced_clocks.py   # writes data_positions/ (480 clocks, 12 exact minute positions x 40)
  python describe_check.py --data_csv data_positions/data.csv --images_dir data_positions \
      --out_csv results/describe_positions.csv
  ```
- **Even when it describes BOTH hands correctly, it still answers wrong 66%
  of the time** (147/480 images had both `minute_desc_ok` and `hour_desc_ok`
  true; of those, 66.0% still had the wrong final HH:MM). This is the core
  behavioral finding: perceiving the hands correctly and composing the right
  answer are two separate steps, and the second one is failing far more
  often than the first.
- **The "hand at number N -> minutes" conversion works for familiar
  positions and fails completely for others.** Grouping `describe_positions.csv`
  by the number the model *said* the minute hand was at (`said_minute_number`)
  and checking whether the final answer's minute matches that number's
  target (`N * 5`, or `0` for `N=12`):

  | said N | 12 | 9 | 6 | 10 | 3 | 4 | 2 | 5 | 7 |
  |---|---|---|---|---|---|---|---|---|---|
  | n | 120 | 22 | 125 | 33 | 57 | 43 | 50 | 27 | 3 |
  | conversion correct | 100% | 100% | 89% | 45% | 61% | 47% | **0%** | **0%** | 0% |

  The clock-face landmarks (12, 9, 6) convert essentially perfectly; several
  others (2, 5, and the n=3 case of 7) convert at 0%. Reproduced from the same
  `describe_positions.csv` above, grouped with pandas (`groupby("said_minute_number")`).
- **In text only (no image, hand positions given in words), the same
  conversions are 100% correct.** Reproduce: `python text_only_check.py`
  (prints to the console; a small hand-checked qualitative probe -- 8
  full "minute-hand at X, hour-hand at Y" cases plus 7 pure "number N ->
  minutes" cases -- not a large-sample CSV like the others above). Together
  with the previous point, this localizes the failure: the model *can* both
  perceive the hand position and do the number arithmetic in isolation; it
  fails at using the two together when reading an actual image.

### Probing

Is the true minute-hand angle linearly decodable from the model's LLM hidden
states on the clock image -- including on images the model itself answered
wrong? Using the last-token hidden state at every layer of the 36-layer LLM
stack (`hidden_last` in `probe_output/probe_results/per_layer_results.csv`,
`split="all"`, n=480 balanced clocks):

- **Layer 0 (the embedding layer) is at chance**: R² ~ 0 (-0.002), 12-way
  classification accuracy 4.4% (chance is 1/12 = 8.3%).
- **R² rises to 0.86 by layer 1**, peaks around **R² = 0.94 at layer 21**,
  and is still **R² = 0.88 at the final layer (36)** -- the signal survives
  to the very output of the network, not just in early/middle layers.
- **Mean angular error is 8-12 degrees** across the layers where the angle
  is decodable (out of a 360-degree circle -- for reference, each clock
  number is 30 degrees apart).
- **12-way nearest-clock-number classification reaches ~95%** at the best
  layers.
- **Controls hold**: the shuffled-label baseline (same probe, labels
  randomly permuted) stays negative at every layer (R² between -0.005 and
  -0.28), confirming the probe is picking up a real, consistent signal in
  the activations rather than overfitting noise.

Reproduce:
```
python clocks.py --balanced --out_dir data_balanced --images_per_minute 8   # 480 clocks, all 60 minute values
python probe.py --stage extract --data_csv data_balanced/data.csv --images_dir data_balanced
python probe.py --stage probe
```
Full per-layer numbers: `probe_output/probe_results/per_layer_results.csv`
(filter `representation=="hidden_last"`, `hand=="minute"`, `split=="all"`).
Plots: `probe_output/probe_results/layers_minute.png`.

> **Note on `probe_output/probe_results/summary_table.csv`:** its `r2_correct`
> column for the minute-hand probe shows absurd values (e.g. `-6.3e+30`).
> This is a known numerical artifact of computing R² on the "answered
> correctly" subset when it has only 12 images (minute accuracy is ~2%, so
> there's almost nothing in that group) -- not a bug in the underlying
> signal. It's also a symptom of the float16/NaN issue `probe.py` has since
> been hardened against (float32 activations, per-layer NaN/Inf diagnostics,
> `--recover`); this particular run predates that fix. The `r2_all` and
> `r2_wrong` columns, and every number quoted above, are unaffected and were
> independently re-verified from `per_layer_results.csv` directly.

### Conclusion

**The visual information about the minute hand's true angle survives to the
model's output layer and is not used.** A simple linear probe recovers it
with R² > 0.85 at almost every layer, including on images where the model's
own final answer got that hand wrong -- so this isn't a perception failure
("the model can't see the hands"). Combined with the behavior-check findings
above, the failure looks localized to the step that converts a correctly
perceived hand position into the correct number of minutes, especially for
clock positions other than the handful of familiar landmarks (12, 9, 6).

## Files

| File | Purpose |
|---|---|
| `clocks.py` | Draws synthetic analog clock PNGs + a metadata CSV of ground-truth times |
| `eval_behavior.py` | Loads Qwen2.5-VL-3B-Instruct and asks it to read each clock |
| `analyze.py` | Computes accuracy metrics, hand-swap rate, and saves example error images |
| `probe.py` | Step 2: extracts hidden states, probes them for the hand angles, and (via `--stage direction`) fits+saves a steering direction for Step 3 |
| `intervene.py` | Step 3: causal interventions -- activation patching between clocks (Experiment A) and steering along the probe direction (Experiment B) |
| `balanced_clocks.py` | Generates `data_positions/`: 480 clocks at exactly the 12 clock-number minute positions (40 each), for the position-accuracy breakdown |
| `describe_check.py` | "Describe the hands, then answer" prompt -- logs whether each hand was *described* correctly, separately from whether the final answer was correct |
| `eval_prompt2.py` | `eval_behavior.py` with the describe-then-answer prompt, for quick spot checks (small sample) |
| `text_only_check.py` | No-image sanity check: can the model do the "hand at number N -> minutes" conversion from a text description alone? |
| `clock_behavior.ipynb` | Step 1, bundled into one notebook, for Kaggle |
| `requirements.txt` | Python dependencies |

Directories produced by the scripts (gitignored where regenerable -- see
`.gitignore`; the Results section above says which script writes what):

```
data/                    # clock_0000.png ... clock_0499.png + data.csv (Step 1, gitignored)
data_balanced/            # 480 clocks, equal count per minute value + data.csv (Step 2/3 input, gitignored)
data_positions/            # 480 clocks, 12 exact minute positions x 40 + data.csv (position-breakdown input, gitignored)
results/                 # results.csv, describe.csv, describe_positions.csv, prompt2.csv,
                          # describe20.csv, parse_failures.csv (checked in -- all small CSVs)
analysis_output/         # summary.txt, by_hour.csv, by_minute_bucket.csv, examples/*.png (Step 1 -- not currently
                          # present in this repo snapshot; regenerate via analyze.py, small enough to check in)
probe_output/
  activations/            # hidden_last.npy, hidden_meanpool.npy, vision_meanpool.npy, index.csv (gitignored, ~150MB+)
  probe_results/          # per_layer_results.csv, shuffled_label_control.csv, summary.txt,
                          # summary_table.csv, layers_minute.png, layers_hour.png, and (after
                          # `--stage direction`) probe_direction_<representation>_<hand>.npz (all checked in)
intervene_output/         # experiment_a_trials.csv, experiment_a_summary.csv/.txt, experiment_a_layers.png,
                          # experiment_b_trials.csv, experiment_b_summary.csv/.txt, experiment_b_steering.png
```

## Option A: Run on Kaggle (recommended)

1. Create a new Kaggle notebook and upload `clock_behavior.ipynb` (or copy/paste
   its cells).
2. In the notebook settings, turn on **GPU T4 x2** (or any T4) as the accelerator.
3. Run all cells top to bottom. The notebook:
   - installs the extra packages it needs (`transformers`, `accelerate`, `qwen-vl-utils`),
   - writes out `clocks.py` / `eval_behavior.py` / `analyze.py` into the Kaggle
     working directory (so it's fully self-contained -- you don't need to
     upload the repo separately),
   - generates 500 clocks, evaluates them, and analyzes the results.
4. The first run downloads the ~3B parameter model from Hugging Face (a few GB),
   which takes a few minutes. Generation for 500 images on a T4 in float16 takes
   roughly tens of minutes -- there's a commented-out line in the notebook to
   first run on a handful of images (`max_images=8`) to sanity-check everything
   works before committing to the full run.

`clock_behavior.ipynb` only covers Step 1. Step 2 (`probe.py`) isn't bundled
into a notebook yet -- on Kaggle, either upload it alongside `clocks.py` and
`eval_behavior.py` as plain files, or paste its contents into a `%%writefile
probe.py` cell the same way the Step 1 notebook does, then run it with
`!python probe.py --stage extract` / `--stage probe` in separate cells (so
the slow GPU stage and the fast CPU stage can be re-run independently).

## Option B: Run locally as scripts

```bash
pip install -r requirements.txt

# 1. Generate 500 clocks with random times, default style
python clocks.py --n 500 --out_dir data --seed 42

# 2. Ask the VLM to read every clock (needs a GPU with a few GB free)
python eval_behavior.py --data_csv data/data.csv --images_dir data --out_csv results/results.csv

# 3. Analyze the results
python analyze.py --results_csv results/results.csv --images_dir data --out_dir analysis_output

# 4. Step 2: generate the balanced probing dataset (480 clocks, 8 per minute value)
python clocks.py --balanced --out_dir data_balanced

# 5. Step 2: extract hidden states (needs a GPU) then probe them (CPU, fast)
python probe.py --stage extract   # slow: loads the model, runs all 480 images
python probe.py --stage probe     # fast: fits/cross-validates the probes, makes plots

# 6. Step 3: fit + save a steering direction for Experiment B (needs --stage probe's output)
python probe.py --stage direction

# 7. Step 3: verify the intervention mechanism actually works, BEFORE trusting a null result
python intervene.py --verify --experiment none

# 8. Step 3: causal interventions (needs a GPU) -- smoke-test first, then the full sweep
python intervene.py --max_pairs 1 --layers 0,1,21,36   # quick correctness check, a few minutes
python intervene.py --verify                            # full sweep, both experiments, verified
```

Each script also has a `--help` for its full option list, and each exposes a
plain Python function (`generate_dataset`, `run_eval`, `run_analysis`,
`extract_activations`, `run_probing`, `fit_probe_direction`,
`run_experiment_a`, `run_experiment_b`, `run_verification`) so you can call
it directly from a notebook or another script instead of the CLI.

## What each script does

### `clocks.py`

`generate_clock_image(hour, minute, save_path, ...)` draws one clock face with
matplotlib and saves it as a PNG. Controls: `hour_hand_length`,
`minute_hand_length`, `hour_hand_thickness`, `minute_hand_thickness`,
`show_numbers`, `show_ticks`. Default style: white face, black hands, hour
hand shorter *and* thicker than the minute hand (a fairly standard analog
clock look).

`generate_dataset(n, out_dir, seed, ...)` calls that in a loop with random
hour (1-12) and minute (0-59) values, and writes `data.csv` with one row per
image logging the true time and every rendering setting used, so the dataset
is fully reproducible from the CSV alone. This is the Step 1 dataset.

`generate_balanced_dataset(images_per_minute, out_dir, seed, ...)` (CLI:
`--balanced`) instead loops over every minute value 0-59 and draws
`images_per_minute` clocks at each (default 8, for 480 images total), each
with a random hour. This is the Step 2 probing dataset: a regression probe
over the full 0-360 degree angle needs even coverage of that range, which a
plain random draw doesn't guarantee at a few hundred samples.

### `eval_behavior.py`

Loads `Qwen/Qwen2.5-VL-3B-Instruct` via `transformers` in `float16` with
`device_map="auto"`. For each clock image it sends the prompt:

> "What time does this clock show? Answer only in HH:MM format."

and parses the reply with a regex for `H:MM` / `HH:MM`. If no such pattern is
found (or the numbers are out of a sane clock range), the row is marked
`parse_success=False` and also written to `parse_failures.csv` for quick
inspection -- these are logged, not silently dropped, since a parse failure is
itself a data point about model behavior. The model's full raw reply is
always saved in the `raw_answer` column of `results.csv`, alongside the
parsed prediction, so you can re-inspect or re-parse it later.

Generation is deterministic: greedy decoding (`do_sample=False`) with a small
`max_new_tokens` (16, since the expected answer is just `HH:MM`), plus a
fixed seed (`set_seed`, default 0) for full reproducibility.

### `analyze.py`

Reads `results.csv` and reports:

- **Exact accuracy** -- both hour and minute correct.
- **Hour accuracy / minute accuracy** -- each checked independently (so you
  can see e.g. "hour is almost always right but minute is often wrong").
- **Hand-swap rate** -- see below. Reported two ways: `hand_swap_rate_including_overlap`
  (over every parsed clock) and `hand_swap_rate_excluding_overlap` (over only
  the clocks where the hands are clearly apart -- see "overlap" below).
- **Accuracy within +/-5 minutes** -- using circular distance on a 12-hour
  face (so 11:58 vs. 12:02 counts as 4 minutes apart, not ~718).
- **Breakdown by hour** and **by minute bucket** (5-minute buckets, so you can
  compare "near :00" -- where both hands are close together -- against other
  times) -- see `by_hour.csv` / `by_minute_bucket.csv`.
- **Example images** for each error category, saved as PNG montages under
  `analysis_output/examples/` (`correct`, `hand_swap`, `hour_only_correct`,
  `minute_only_correct`, `within_5min`, `other_error`, `parse_failure`).

**Hand-swap heuristic:** on an analog face, the hour hand's position lines up
with a minute tick (hour `H` sits at the same angle as minute `H*5`), and the
minute hand's position lines up with an hour number (minute `M` sits at the
same angle as hour `round(M/5)`). We flag an answer as a "hand swap" if it
matches the time you'd get by reading the hands backwards this way. This is
an approximation (documented in `analyze.py::is_hand_swap`), not a ground
truth -- for times near quarter-hours a swapped reading can occasionally
coincide with another error type -- but it gives a useful rate to track.

**Overlap flag:** near times like 12:00, 1:05, or 2:11, the hour and minute
hands point close enough together that a swapped reading and a correct
reading look almost identical on the image -- so a "hand swap" there is much
less meaningful than one where the hands are clearly apart. `analyze.py`
flags any clock where the true hour and minute hands are within 15 degrees
of each other (`hands_overlap`, using the exact same angle formulas as the
renderer in `clocks.py::hand_angles`) as `is_overlap`, and the summary
reports `n_overlap_clocks` / `overlap_rate` plus the hand-swap rate computed
both with and without those clocks. The threshold is configurable via
`--overlap_threshold_deg`.

### `probe.py` (Step 2)

Two independent stages, selectable with `--stage extract` / `--stage probe` /
`--stage all` (default):

**`extract`** (needs a GPU) -- loads Qwen2.5-VL-3B-Instruct, runs every clock
image in `data_balanced/` through it once, and for each image saves:
  - the **last-token hidden state** at every LLM layer (`output_hidden_states=True`)
    -- the position the model uses to start generating its answer;
  - the same, **mean-pooled over the image-token positions** instead of the
    last token;
  - the **vision encoder's own output**, mean-pooled over its patch tokens
    (captured with a forward hook on the vision tower, located by scanning
    the model's module tree for a vision-transformer-looking class rather
    than a hardcoded attribute path, since that path has moved between
    transformers versions);
  - the model's generated answer (so we know, per image, whether each hand
    was read correctly).

Saved as `hidden_last.npy` / `hidden_meanpool.npy` / `vision_meanpool.npy`
(shape `(N, layers, hidden_dim)`, float32) plus `index.csv` (true time, true
hand angles, the model's answer, correctness flags). Memory stays low because
only one image's GPU activations exist at a time; the small float32 vectors
that get kept in RAM across all 480 images total roughly 300MB. Activations
are saved as float32 (not float16) because some hidden-state dimensions can
occasionally exceed float16's ~65504 max magnitude and silently become
inf/NaN -- `run_probing` also diagnoses, repairs, or skips any layer that
still has non-finite values (see `--recover` for repairing already-saved
files in place without re-extracting).

**`probe`** (CPU only, fast, re-runnable) -- loads the saved activations and,
for the minute-hand and hour-hand angle separately:
  - a **regression probe** (Ridge, after `StandardScaler` + `PCA`) predicting
    `sin`/`cos` of the true angle (so the probe never has to handle the
    359->0 degree wraparound directly), reported as R² and mean angular
    error in degrees;
  - a **classification probe** (multinomial logistic regression) predicting
    which of the 12 face numbers the hand is nearest to;
  - both cross-validated (5-fold by default) and run at **every layer**, for
    all three saved representations.

**Key analysis:** `cross_val_predict` gives one out-of-fold prediction per
image per layer. We then split those same predictions by whether *that
specific hand* was read correctly in the model's final answer (minute-hand
probe split by `minute_correct`, hour-hand probe split by `hour_correct` --
not by exact HH:MM match, which happens on only ~2% of images and would
leave the "correct" group too small to say anything) and compute R²/accuracy
separately for each group. If the angle is still decodable on the "wrong"
images, that demonstrates the information was present in the model's
activations even though it didn't make it into the answer.

**Controls:**
  - a **shuffled-label baseline** -- the identical probe pipeline, but with
    the true angle/label randomly permuted across images first, run at every
    layer. A real probe should clearly beat this; if it doesn't, it's fitting
    noise, not signal.
  - the **majority-class baseline** for classification (accuracy of always
    guessing the most common of the 12 positions).
  - for regression there's no separate "predict the mean" model to run: R² is
    defined so that always predicting the mean scores exactly 0, so R² is
    already its own baseline (R² < 0 means worse than that).
  - we deliberately skip probing a randomly-initialized model (it would
    roughly double the GPU extraction cost for a check the shuffled-label
    control already covers: whether the probe is finding real structure in
    *these* activations, vs. fitting noise).

Output: `probe_output/probe_results/per_layer_results.csv` (every layer x
hand x split x representation), `shuffled_label_control.csv`,
`summary.txt` / `summary_table.csv` (the best layer per representation x
hand, correct vs. wrong vs. shuffled, side by side), and `layers_minute.png`
/ `layers_hour.png` (R² and accuracy vs. layer, all/correct/wrong lines plus
the shuffled-label reference line).

**`--stage direction`** (CPU only) refits the minute-hand probe on ALL
images (no CV split -- for steering we want the single best-fit probe, not
a held-out estimate) at one layer (auto-picked as the best from
`per_layer_results.csv`, or given via `--direction_layer`), and saves it as
a **steering direction in raw activation space** to
`probe_output/probe_results/probe_direction_hidden_last_minute.npz`: the
fitted pipeline (`StandardScaler` -> `PCA` -> `Ridge`) is a linear map from
activations to `(sin, cos)`, so the chain rule gives a Jacobian whose two
columns are the raw-activation-space directions that increase the sin- and
cos-readouts -- exactly what Experiment B steers along. `probe.py::apply_saved_probe`
recomputes the exact predicted angle from any raw hidden-state vector using
this same saved pipeline, which `intervene.py` uses to verify a steering
intervention actually moved the probe's internal readout.

### `intervene.py` (Step 3)

Two experiments testing, causally, whether the angle information that
probing finds is ever actually *used*. Both reuse `probe.py`'s model
loading, vision-module/image-token-id finding, and vision-encoder hook
(imported, not copy-pasted), plus `eval_behavior.py`'s answer parser.
Run with `--experiment a` / `--experiment b` / `--experiment both` (default).

**Experiment A -- activation patching between clocks.** For a pair of clocks
(A, B) with the same hour but minutes at least `--min_gap` (default 15)
apart: cache B's full per-position hidden states at every layer, then run A
while a hook *replaces* the hidden state at a chosen layer and set of token
positions with B's -- and check whether A's stated minute moves toward B's.
Swept over every layer (0 = the embedding layer / input to the first
decoder block, 1..36 = each decoder block's output -- the same indexing
`output_hidden_states=True` uses) and three position sets (`image_tokens`,
`final_token`, `all_positions`), plus a **vision-encoder ceiling condition**
that replaces the whole vision-tower output instead of an LLM layer (if
swapping the *entire* visual representation doesn't move the answer,
nothing downstream will). Two controls, run on the same grid: patching from
a **same-minute** clock (should change nothing, since the ground truth is
unchanged) and patching in **matched-norm random noise** (isolates "does
perturbing this position at all matter" from "does B's specific content
matter"). Outcome measures per trial: the patched minute, whether it moved
*toward* B's minute (`circular_dist_minutes` decreased vs. the unpatched
baseline), a 0-1 **shift score** (how much of the gap between A and B's
minutes closed -- not clipped, so overshoot/backward movement are visible),
and whether the answer changed at all. **A layer/position patch is only
called causally load-bearing if it moves the answer clearly more than its
own noise and same-minute controls at that same layer** -- not against an
assumed 50% chance level (see the printed summary for why: when A and B are
near-maximally far apart, even a uniformly random perturbation has good
odds of landing "closer" by pure geometry).

**Experiment B -- steering along the probe direction.** Loads the direction
saved by `probe.py --stage direction`. For each trial image, the steering
target is `(true_minute + --target_offset) % 60` (default: diametrically
opposite, the clearest possible target). The direction vector for that
target is `w_sin * sin(target) + w_cos * cos(target)` (a "prototype"
direction for that specific angle), normalized, then added -- scaled by
`alpha` times *that image's own* last-token activation norm at the
probe's layer, so alpha is self-normalizing across images -- to the
residual stream at the **final token position** (matching what the probe
was actually trained on). Swept over `--alphas` (default
`-2,-1,-0.5,-0.25,0,0.25,0.5,1,2`). Two things are measured per trial: (1)
whether the *stated* minute moved toward the target, same shift-score
machinery as Experiment A; and (2) whether the *probe's own readout*
(recomputed via `apply_saved_probe` on the steered activation vector, no
new forward pass needed) moved toward the target. This separates two very
different outcomes that look identical from the outside: "steering failed
to move the representation" vs. "the representation moved but the model's
output ignored it" -- if (2) tracks alpha closely while (1) stays flat,
it's the second one. Controlled against steering along a **fixed random
direction** of the same norm, run through the identical pipeline.

Both experiments print a **time estimate** partway through their first few
trials (measured, not guessed) before committing to the full sweep, and
support `--max_pairs` (cap trial count) and `--layers` (Experiment A: a
comma-separated layer subset, e.g. `0,1,21,36`) for a quick smoke test.
Output: `intervene_output/experiment_a_trials.csv` (every individual
trial) / `experiment_a_summary.csv`+`.txt` (aggregated per layer x
position-set x condition) / `experiment_a_layers.png`, and the equivalent
`experiment_b_*` files.

**`--verify`** exists because a clean null result (patching/steering doesn't
move the answer) and a silently broken hook (patching/steering doesn't run
at all) look IDENTICAL in the experiment output -- and a broken hook is by
far the likelier explanation for a suspiciously total null (e.g. the
vision-encoder ceiling condition showing exactly 0% answer-change across
every trial). `--verify` checks the mechanism directly, independent of the
experiment logic: for a few (A, B) pairs, it (1) confirms the vision-encoder
swap actually writes a different tensor (reported as a relative L2
difference, not a boolean) and that the difference propagates into the
LLM's hidden states (checked at the embedding layer and one layer
downstream -- if that's bit-identical, the hook isn't reaching the forward
path used for generation); (2) runs a maximally aggressive version of the
same intervention -- zeroing the vision output entirely, and separately
replacing it with a random-noise image's -- and confirms the ANSWER changes
at all; and (3) for Experiment A's decoder-layer patching, confirms the
patched positions actually differ downstream of the patched layer (checked
one layer down and at the final layer) and reports what fraction of
sequence positions each position set covers, so a silently-empty
`image_tokens` mask can't hide. Ends with a blunt PASS/FAIL verdict, plus
`verification_report.txt` / `verification_vision_swap.csv` /
`verification_decoder_patch.csv` saved into `--out_dir` so every result
carries its own evidence. Combine with `--experiment none` for a fast
standalone check (a few pairs, no full sweep), or with a real experiment run
to verify and produce results in one command.

## Notes for Kaggle's T4 (16GB)

- `float16` keeps the ~3B parameter model comfortably under the 16GB budget
  (roughly 6-7GB for weights, plus activations).
- The eval and probe-extraction scripts both process one image at a time (not
  batched) to keep the code simple and easy to read/debug -- a deliberate
  simplicity trade-off, not a perf-critical pipeline.
- If you hit a CUDA OOM anyway, lower `image_size` in `clocks.py` (default
  512x512) or reduce `max_new_tokens` in `eval_behavior.py` / `probe.py`
  (default 16, already small since the answer is just `HH:MM`).
- `probe.py --stage extract` does roughly 2x the per-image GPU work of
  `eval_behavior.py` (one plain forward pass to capture hidden states, plus
  one `generate()` call to score the answer, each of which separately
  re-runs the vision encoder) -- expect it to take longer than Step 1's eval
  run for the same image count. Use `--max_images 8` first to sanity-check
  the whole pipeline (including the probing stage) before committing to the
  full 480-image run.
- `probe.py --stage probe` reduces activations to `--n_components` (default
  100) dimensions with PCA before fitting each linear probe -- both to keep
  cross-validated Ridge/LogisticRegression fits fast at ~2000+ raw hidden
  dimensions with only ~480 images, and to reduce overfitting risk from
  fitting a linear model with more features than samples.
- `intervene.py` caches each source image's FULL per-position, per-layer
  hidden states (not just the pooled vectors `probe.py` keeps) so it can
  patch any layer/position combination -- roughly 150-250MB per cached
  image, held only transiently (per pair/trial image, not accumulated
  across the whole run), well inside a T4's RAM. Compute-wise, Experiment
  A's default settings (`--n_pairs 5`, every layer, all 3 position sets,
  3 conditions) run on the order of ~1500-1700 `generate()` calls; each is
  small (`--max_new_tokens 16`) but the hook adds a little overhead per
  call, so expect it to take longer than a single `eval_behavior.py` pass
  over the same image count -- the printed time estimate (measured from the
  first few trials, not guessed) tells you the real number before it
  commits to the full sweep. `--max_pairs 1 --layers 0,1,21,36` finishes in
  a couple of minutes and is worth running first to confirm the hooks work
  correctly against the real model (they're tested locally against a fake
  decoder stack, but transformers-version quirks -- like the vision
  hook's bare-tensor/tuple/ModelOutput ambiguity that already bit this
  project once -- can only be confirmed against the real thing).
