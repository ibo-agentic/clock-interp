# Clock Interp

Mechanistic interpretability project on why vision-language models (VLMs)
misread analog clocks, especially swapping the hour and minute hands.

- **Step 1: behavior check.** Generate synthetic clocks, ask a VLM to read
  them, analyze the errors. Found: Qwen2.5-VL-3B-Instruct reads clocks badly
  (~2% exact accuracy), but often *describes* the hands correctly in words.
- **Step 2: probing.** Is the true hand angle linearly decodable from the
  model's hidden states, even on images where its final answer is wrong? If
  so, the information was there -- the model just failed to use it.

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
| `probe.py` | Step 2: extracts hidden states and probes them for the hand angles |
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
data_balanced/            # 480 clocks, equal count per minute value + data.csv (Step 2 probing input, gitignored)
data_positions/            # 480 clocks, 12 exact minute positions x 40 + data.csv (position-breakdown input, gitignored)
results/                 # results.csv, describe.csv, describe_positions.csv, prompt2.csv,
                          # describe20.csv, parse_failures.csv (checked in -- all small CSVs)
analysis_output/         # summary.txt, by_hour.csv, by_minute_bucket.csv, examples/*.png (Step 1 -- not currently
                          # present in this repo snapshot; regenerate via analyze.py, small enough to check in)
probe_output/
  activations/            # hidden_last.npy, hidden_meanpool.npy, vision_meanpool.npy, index.csv (gitignored, ~150MB+)
  probe_results/          # per_layer_results.csv, shuffled_label_control.csv, summary.txt,
                          # summary_table.csv, layers_minute.png, layers_hour.png (checked in)
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
```

Each script also has a `--help` for its full option list, and each exposes a
plain Python function (`generate_dataset`, `run_eval`, `run_analysis`,
`extract_activations`, `run_probing`) so you can call it directly from a
notebook or another script instead of the CLI.

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
