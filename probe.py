"""
probe.py -- STEP 2: is the hand angle encoded in the model's hidden states,
even on images where its final HH:MM answer is wrong?

Background (from the Step 1 behavior check, see results/ and README.md):
Qwen2.5-VL-3B-Instruct reads clocks badly (exact accuracy ~2%), but it often
DESCRIBES the hands correctly in words, and correctly converts "hand at
number N" -> minutes when asked in text only. This script checks whether the
true hand angle is linearly decodable from the model's internal activations
on the clock IMAGE itself -- including on images where the model's own
final answer got that hand wrong. If so, the information was there; the
model just failed to use it when composing its answer.

This script has two stages, selectable with --stage (default: both):

  extract  (needs a GPU) Run every clock image through the model once,
           saving hidden states from every LLM layer plus the vision
           encoder's output. This is the slow, expensive part.

  probe    (CPU only)    Load the saved activations and fit linear probes
           for the hour- and minute-hand angle, layer by layer, comparing
           probe performance on images the model answered correctly vs.
           wrong. This is fast and can be re-run repeatedly (e.g. to try a
           different --n_components) without re-extracting activations.

Usage:
    python probe.py --stage extract   # run this on the Kaggle GPU
    python probe.py --stage probe     # then this, as many times as you like

Activations are saved as float32. `run_probing` also checks every layer for
NaN/Inf before handing it to PCA (loudly reporting counts, repairing via
column-mean imputation if only partially bad, or skipping the layer
entirely if there's nothing usable), so a bad layer degrades gracefully
instead of producing a silent garbage number. If you'd rather not repeat the
(slow, GPU) extraction after finding bad values, run:
    python probe.py --recover         # repairs saved activations in place
    python probe.py --stage probe     # then re-probe as usual

Expects the balanced dataset from clocks.py's --balanced mode:
    python clocks.py --balanced --images_per_minute 8 --out_dir data_balanced
(480 images, an equal number at every minute value 0-59, so the regression
probe sees even coverage of the full 0-360 degree angle range.)
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clocks import hand_angles
from eval_behavior import MODEL_ID, PROMPT, SEED, parse_time_answer

DATA_CSV = "data_balanced/data.csv"
IMAGES_DIR = "data_balanced"
ACTIVATIONS_DIR = "probe_output/activations"
RESULTS_DIR = "probe_output/probe_results"
N_COMPONENTS = 100
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def minute_number(minute):
    """Which of the 12 face numbers the minute hand is nearest to (1-12).
    E.g. minute=32 -> nearest to "6" (30 minutes)."""
    n = round(minute / 5) % 12
    return 12 if n == 0 else n


def hour_number(hour):
    """Which of the 12 face numbers the hour hand is nearest to (1-12)."""
    n = hour % 12
    return 12 if n == 0 else n


# ---------------------------------------------------------------------------
# STAGE 1: extract activations (needs the model + a GPU)
# ---------------------------------------------------------------------------

def find_vision_module(model):
    """Locate the vision tower by scanning the module tree for a class whose
    name looks like a vision transformer, instead of hardcoding an attribute
    path like `model.visual` -- that path has moved between transformers
    versions, but the class naming has stayed recognizable. The first match
    in `named_modules()` is the outermost vision module (PyTorch yields
    parents before their children), which is what we want to hook.
    """
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "VisionTransformer" in cls_name or ("Vision" in cls_name and cls_name.endswith("Model")):
            return module
    raise AttributeError("Could not find the vision tower in this model's module tree.")


def find_image_token_id(model, processor):
    """The placeholder token id used for each image patch in the input
    sequence -- needed to select which hidden-state positions are image
    tokens (for the mean-pooled-over-image-tokens representation)."""
    token_id = getattr(model.config, "image_token_id", None)
    if token_id is not None:
        return token_id
    return processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def load_model(model_id=MODEL_ID):
    """Load Qwen2.5-VL-3B-Instruct + its processor in float16 with
    device_map="auto". Shared by extract_activations (below) and
    intervene.py's causal-intervention experiments, so there's exactly one
    place that knows how to load the model.

    transformers is imported here (not at module level) so that anything
    that only needs the CPU-only parts of this file (e.g. `--stage probe`)
    still works in an environment without transformers/a GPU installed.
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"Loading {model_id} in float16 ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def make_vision_hook(vision_holder):
    """Build a forward hook that captures the vision tower's output into
    `vision_holder["out"]` (a plain dict the caller owns and can clear/read
    around each image). Depending on the transformers version this module
    returns a bare tensor, a tuple, or a ModelOutput object -- handle all
    three. (Confirmed necessary in practice: the real Qwen2.5-VL vision
    tower on Kaggle returned a ModelOutput here, not a bare tensor.)

    Factored out so intervene.py can register the exact same hook, and (for
    Experiment A's vision-encoder patch) a variant that also overwrites the
    output instead of only reading it.
    """
    def hook(module, inputs_, output):
        tensor = output
        if hasattr(tensor, "last_hidden_state"):
            tensor = tensor.last_hidden_state
        elif isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not torch.is_tensor(tensor):
            return
        vision_holder["out"] = tensor.detach()
    return hook


@torch.no_grad()
def process_one_image(model, processor, image_path, image_token_id, vision_holder,
                       prompt=PROMPT, max_new_tokens=16):
    """Run one clock image through the model. Returns:
      - last_token_vecs: list of (hidden_dim,) float32 arrays, one per LLM
        layer (index 0 = embedding output, then each transformer block),
        taken at the LAST prompt token -- the position the model uses to
        start generating its answer.
      - meanpool_vecs: same, but mean-pooled over the image-token positions
        instead of just the last token.
      - vision_vec: (vision_dim,) float32 array, the vision encoder's own
        output for this image, mean-pooled over its patch tokens.
      - raw_reply: the model's generated text answer, used later to score
        whether the final answer was correct.

    We keep everything float32 (not float16) once it leaves the GPU: float16
    has a max magnitude of ~65504, and some hidden-state dimensions in real
    transformers occasionally blow past that ("activation outliers"), which
    silently turns into inf/NaN if stored as float16. float32 has no such
    problem at these magnitudes.
    """
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(model.device)

    # --- forward pass: this is what we probe ---
    vision_holder.clear()
    outputs = model(**inputs, output_hidden_states=True)

    input_ids = inputs["input_ids"][0]
    image_mask = (input_ids == image_token_id)
    if not image_mask.any():
        # Should not happen, but never silently pool the wrong thing.
        print(f"WARNING: no image tokens found for {image_path}; pooling the full sequence instead.")
        image_mask = torch.ones_like(input_ids, dtype=torch.bool)

    last_token_vecs, meanpool_vecs = [], []
    for h in outputs.hidden_states:
        h0 = h[0]  # drop the batch dim -> (seq_len, hidden_dim)
        last_token_vecs.append(h0[-1].float().cpu().numpy())
        meanpool_vecs.append(h0[image_mask].mean(dim=0).float().cpu().numpy())

    vision_out = vision_holder.get("out")
    if vision_out is None:
        vision_vec = None
    else:
        pooled = vision_out if vision_out.dim() == 1 else vision_out.mean(dim=0)
        vision_vec = pooled.float().cpu().numpy()

    del outputs  # free the hidden_states tuple before generate() runs

    # --- generate: only used to score whether the final answer was correct ---
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
    raw_reply = processor.decode(trimmed, skip_special_tokens=True).strip()

    return last_token_vecs, meanpool_vecs, vision_vec, raw_reply


def _warn_nonfinite(name, arr):
    """Loudly report any NaN/Inf in a freshly extracted activation array,
    broken down by layer if the array has a layer axis (N, L, H) rather than
    just (N, H). This runs once right after extraction, on the full array,
    so problems are visible immediately instead of surfacing later as a
    cryptic PCA warning during probing."""
    if arr.ndim == 3:
        n_images, n_layers, _ = arr.shape
        bad_layers = []
        for layer in range(n_layers):
            layer_slice = arr[:, layer, :]
            n_bad = int((~np.isfinite(layer_slice)).sum())
            if n_bad:
                n_bad_images = int((~np.isfinite(layer_slice)).any(axis=1).sum())
                bad_layers.append((layer, n_bad, n_bad_images))
        if bad_layers:
            print(f"\n{'!' * 70}")
            print(f"WARNING: {name} has non-finite (NaN/Inf) values in {len(bad_layers)}/{n_layers} layer(s):")
            for layer, n_bad, n_bad_images in bad_layers:
                print(f"  layer {layer}: {n_bad} non-finite value(s) across {n_bad_images}/{n_images} images")
            print("run_probing() will report, repair, or skip these layers automatically;")
            print(f"or run `python probe.py --recover` to repair the saved files in place.")
            print(f"{'!' * 70}\n")
    else:
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            n_bad_images = int((~np.isfinite(arr)).any(axis=1).sum())
            print(f"\n{'!' * 70}")
            print(f"WARNING: {name} has {n_bad} non-finite value(s) across {n_bad_images}/{len(arr)} images.")
            print(f"{'!' * 70}\n")


def extract_activations(data_csv=DATA_CSV, images_dir=IMAGES_DIR, out_dir=ACTIVATIONS_DIR,
                         model_id=MODEL_ID, max_images=None, seed=SEED):
    """Run every clock image through the model once, saving:
      - hidden_last.npy      (N, L+1, H) float32 -- last-token hidden state per layer
      - hidden_meanpool.npy  (N, L+1, H) float32 -- image-token-mean-pooled hidden state per layer
      - vision_meanpool.npy  (N, V)      float32 -- vision encoder output, mean-pooled
      - index.csv            one row per image: true time, true angles, the
                              model's answer, and whether it was correct

    Memory note: we only ever hold ONE image's activations on the GPU at a
    time (`process_one_image` processes and immediately moves its results to
    CPU float32 numpy arrays). The Python lists below accumulate those small
    per-image vectors across the loop -- for 480 images at a ~2048-dim
    hidden size that's roughly 300MB total, which is fine to keep in host
    RAM even on a modest Kaggle instance.

    transformers is imported here (not at module level) so that `--stage
    probe`, which never needs the model, works even in an environment
    without transformers/a GPU installed.
    """
    from transformers import set_seed
    set_seed(seed)

    df = pd.read_csv(data_csv)
    if max_images is not None:
        df = df.head(max_images)

    model, processor = load_model(model_id)

    image_token_id = find_image_token_id(model, processor)
    vision_module = find_vision_module(model)
    vision_holder = {}
    vision_module.register_forward_hook(make_vision_hook(vision_holder))

    last_all, meanpool_all, vision_all, rows = [], [], [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting activations"):
        path = os.path.join(images_dir, r["filename"])
        last_vecs, meanpool_vecs, vision_vec, raw_reply = process_one_image(
            model, processor, path, image_token_id, vision_holder)

        last_all.append(np.stack(last_vecs))
        meanpool_all.append(np.stack(meanpool_vecs))
        vision_all.append(vision_vec if vision_vec is not None else np.zeros_like(last_vecs[0]))

        true_hour, true_minute = int(r["hour"]), int(r["minute"])
        hour_angle, minute_angle = hand_angles(true_hour, true_minute)
        pred_hour, pred_minute, ok = parse_time_answer(raw_reply)

        rows.append({
            "filename": r["filename"],
            "true_hour": true_hour,
            "true_minute": true_minute,
            "hour_angle_deg": hour_angle,
            "minute_angle_deg": minute_angle,
            "hour_number": hour_number(true_hour),
            "minute_number": minute_number(true_minute),
            "raw_answer": raw_reply,
            "pred_hour": pred_hour,
            "pred_minute": pred_minute,
            "parse_success": ok,
            "hour_correct": bool(ok and pred_hour == true_hour),
            "minute_correct": bool(ok and pred_minute == true_minute),
            "exact_correct": bool(ok and pred_hour == true_hour and pred_minute == true_minute),
        })

        if len(rows) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(out_dir, exist_ok=True)
    hidden_last = np.stack(last_all).astype(np.float32)
    hidden_meanpool = np.stack(meanpool_all).astype(np.float32)
    vision_meanpool = np.stack(vision_all).astype(np.float32)

    _warn_nonfinite("hidden_last.npy", hidden_last)
    _warn_nonfinite("hidden_meanpool.npy", hidden_meanpool)
    _warn_nonfinite("vision_meanpool.npy", vision_meanpool)

    np.save(os.path.join(out_dir, "hidden_last.npy"), hidden_last)
    np.save(os.path.join(out_dir, "hidden_meanpool.npy"), hidden_meanpool)
    np.save(os.path.join(out_dir, "vision_meanpool.npy"), vision_meanpool)

    index_df = pd.DataFrame(rows)
    index_df.to_csv(os.path.join(out_dir, "index.csv"), index=False)

    print(f"\nSaved activations for {len(index_df)} images to '{out_dir}/':")
    print(f"  hidden_last.npy:     {hidden_last.shape}")
    print(f"  hidden_meanpool.npy: {hidden_meanpool.shape}")
    print(f"  vision_meanpool.npy: {vision_meanpool.shape}")
    print(f"  exact accuracy on this set:  {index_df['exact_correct'].mean():.1%}")
    print(f"  minute read correctly:       {index_df['minute_correct'].mean():.1%}")
    print(f"  hour read correctly:         {index_df['hour_correct'].mean():.1%}")
    return index_df


# ---------------------------------------------------------------------------
# STAGE 2: probes (CPU only, works from saved activations)
# ---------------------------------------------------------------------------

def build_targets(index_df):
    """Regression targets (sin/cos of each hand's true angle -- this avoids
    making the probe deal with the 359->0 degree wraparound directly) and
    classification targets (nearest-of-12-face-numbers) for every image."""
    minute_rad = np.radians(index_df["minute_angle_deg"].to_numpy())
    hour_rad = np.radians(index_df["hour_angle_deg"].to_numpy())
    return {
        "minute": {
            "regression": np.stack([np.sin(minute_rad), np.cos(minute_rad)], axis=1),
            "classification": index_df["minute_number"].to_numpy(),
        },
        "hour": {
            "regression": np.stack([np.sin(hour_rad), np.cos(hour_rad)], axis=1),
            "classification": index_df["hour_number"].to_numpy(),
        },
    }


def make_regression_probe(n_components):
    # PCA first: activations are high-dimensional (~2000+) but we only have
    # a few hundred images, so we reduce dimensionality before fitting a
    # linear probe. This is both much faster and less prone to overfitting
    # than fitting Ridge directly on the raw activations.
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=0)),
        ("ridge", Ridge(alpha=10.0)),
    ])


def make_classification_probe(n_components):
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=0)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def cv_predict(X, y, probe_fn, n_components, n_splits, seed):
    """Out-of-fold predictions for every row of X (each prediction comes
    from a probe that never saw that row during training)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(probe_fn(n_components), X, y, cv=kf)


def regression_metrics(y_true, y_pred, mask=None):
    """R^2 (on sin/cos jointly) and mean angular error in degrees. R^2 = 0 is
    exactly what you'd get by always predicting the mean angle, so it is
    already its own "predict the mean" baseline -- see module docstring."""
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return float("nan"), float("nan")
    r2 = r2_score(y_true, y_pred)
    true_angle = np.degrees(np.arctan2(y_true[:, 0], y_true[:, 1]))
    pred_angle = np.degrees(np.arctan2(y_pred[:, 0], y_pred[:, 1]))
    diff = np.abs(true_angle - pred_angle) % 360
    diff = np.minimum(diff, 360 - diff)
    return r2, float(diff.mean())


def classification_metrics(y_true, y_pred, mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 1:
        return float("nan")
    return accuracy_score(y_true, y_pred)


def diagnose_and_clean_layer(X, label):
    """Check one layer's activation matrix (N, H) for NaN/Inf, repair it if
    it's only partially bad, and drop zero-variance columns before PCA sees
    it. Returns (X_clean, status, note, kept_mask):
      - status "ok":       X had no non-finite values; nothing changed
                            except dropping any zero-variance columns.
      - status "repaired": X had some non-finite values, which were replaced
                            with that column's mean (computed from its
                            finite entries); any column that was entirely
                            non-finite (no mean to compute) was dropped.
      - status "skipped":  nothing usable was left (X_clean is None) -- the
                            caller should skip this layer rather than fit a
                            probe on it.
      - kept_mask: bool array of length X.shape[1] (the ORIGINAL feature
        count passed in), True for columns that survived into X_clean (all
        False if skipped). Lets a caller scatter a direction/coefficient
        vector computed on X_clean back into full raw-activation space --
        used by `fit_probe_direction` for the causal-intervention probe.
    `X` must already be float64 (the caller casts before calling this).
    """
    n_features = X.shape[1]
    n_bad = int((~np.isfinite(X)).sum())
    note = ""

    if n_bad == 0:
        status = "ok"
        kept_mask = np.ones(n_features, dtype=bool)
    else:
        n_bad_images = int((~np.isfinite(X)).any(axis=1).sum())
        print(f"  {label}: {n_bad} non-finite value(s) across {n_bad_images}/{len(X)} images")

        # Per-column mean computed only from that column's finite entries.
        # A column that's entirely non-finite makes nanmean warn about an
        # "empty slice" -- expected and handled below (has_any_finite),
        # so we suppress just that specific warning rather than the result.
        finite = np.isfinite(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_mean = np.nanmean(np.where(finite, X, np.nan), axis=0)

        # A column with NO finite entries anywhere has no mean to impute
        # from -- drop it outright rather than inventing a value.
        has_any_finite = finite.any(axis=0)
        if not has_any_finite.any():
            return None, "skipped", f"all {X.size} values in this layer were non-finite", np.zeros(n_features, dtype=bool)

        X = X[:, has_any_finite]
        col_mean = col_mean[has_any_finite]
        bad_mask = ~np.isfinite(X)
        X = np.where(bad_mask, np.broadcast_to(col_mean, X.shape), X)
        kept_mask = has_any_finite.copy()

        n_dropped_allbad = int((~has_any_finite).sum())
        status = "repaired"
        note = f"{n_bad} non-finite value(s) in {n_bad_images} image(s) imputed with column mean"
        if n_dropped_allbad:
            note += f"; {n_dropped_allbad} column(s) entirely non-finite, dropped"

    # Zero-variance columns carry no information and are exactly what makes
    # PCA's explained_variance_ratio_ divide by zero -- drop them regardless
    # of whether this layer needed any NaN repair.
    keep = X.var(axis=0) > 0
    n_dropped_zerovar = int((~keep).sum())
    if n_dropped_zerovar:
        note = (note + "; " if note else "") + f"{n_dropped_zerovar} zero-variance column(s) dropped"
    X = X[:, keep]
    kept_mask[kept_mask] = keep  # narrow the mask by this second filter too

    if X.shape[1] == 0:
        return None, "skipped", (note + "; no usable columns remained" if note else "no usable columns remained"), np.zeros(n_features, dtype=bool)

    return X, status, note, kept_mask


def clean_representation(activations, prefix):
    """Diagnose and clean every layer of one activation array ONCE, so the
    result can be reused across both hands and both the real and
    shuffled-label probes -- this avoids re-scanning for NaN/Inf and
    printing duplicate warnings for the same layer four times over.
    Returns a list of (X_clean_or_None, status, note), one per layer."""
    n_layers = activations.shape[1]
    print(f"Cleaning {prefix} ({n_layers} layer(s)) ...")
    cleaned = []
    for layer in range(n_layers):
        X = activations[:, layer, :].astype(np.float64)
        cleaned.append(diagnose_and_clean_layer(X, f"{prefix} layer {layer}"))
    n_skipped = sum(1 for _, status, _, _ in cleaned if status == "skipped")
    n_repaired = sum(1 for _, status, _, _ in cleaned if status == "repaired")
    if n_skipped or n_repaired:
        print(f"  -> {prefix}: {n_repaired} layer(s) repaired, {n_skipped} layer(s) skipped "
              f"(of {n_layers} total)")
    return cleaned


def probe_one_representation(cleaned_layers, prefix, targets, hand, correct_mask,
                              n_components, n_splits, seed):
    """Cross-validated probe for one hand's angle, on one already-cleaned
    activation array (see `clean_representation`), at every layer. Returns a
    long-form DataFrame with one row per (layer, split), where split is
    'all' / 'correct' / 'wrong'.

    We call `cross_val_predict` exactly ONCE per layer (an out-of-fold
    prediction for every image), then compute the all/correct/wrong metrics
    from those same predictions. So "correct" vs "wrong" is a post-hoc
    grouping of genuinely held-out predictions, not a separate, weaker
    train/test split -- this is the comparison behind the key finding.
    """
    y_reg, y_clf = targets[hand]["regression"], targets[hand]["classification"]

    rows = []
    for layer, (X, status, note, _kept_mask) in enumerate(cleaned_layers):
        if status == "skipped":
            for split_name, mask in [("all", None), ("correct", correct_mask), ("wrong", ~correct_mask)]:
                n = int(mask.sum()) if mask is not None else len(y_reg)
                rows.append({
                    "representation": prefix, "layer": layer, "hand": hand, "split": split_name,
                    "n": n, "r2": np.nan, "angular_error_deg": np.nan, "clf_accuracy": np.nan,
                    "status": "skipped", "note": note,
                })
            continue

        layer_n_components = min(n_components, X.shape[1])
        reg_pred = cv_predict(X, y_reg, make_regression_probe, layer_n_components, n_splits, seed)
        clf_pred = cv_predict(X, y_clf, make_classification_probe, layer_n_components, n_splits, seed)

        for split_name, mask in [("all", None), ("correct", correct_mask), ("wrong", ~correct_mask)]:
            r2, ang_err = regression_metrics(y_reg, reg_pred, mask)
            acc = classification_metrics(y_clf, clf_pred, mask)
            n = int(mask.sum()) if mask is not None else len(y_reg)
            rows.append({
                "representation": prefix, "layer": layer, "hand": hand, "split": split_name,
                "n": n, "r2": r2, "angular_error_deg": ang_err, "clf_accuracy": acc,
                "status": status, "note": note,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STAGE 2 controls
# ---------------------------------------------------------------------------

def shuffled_label_control(cleaned_layers, prefix, targets, hand, n_components, n_splits, seed):
    """Same probe pipeline, but with the labels randomly shuffled across
    images first. This breaks any true image<->angle correspondence, so a
    probe that still scores well here would mean it's fitting noise (e.g.
    leakage or overfitting at small sample size), not real signal -- this is
    the "shuffled-label baseline" control. Uses the same already-cleaned
    layers as the real probe (see `clean_representation`)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(targets[hand]["classification"]))
    y_reg = targets[hand]["regression"][perm]
    y_clf = targets[hand]["classification"][perm]

    rows = []
    for layer, (X, status, note, _kept_mask) in enumerate(cleaned_layers):
        if status == "skipped":
            rows.append({
                "representation": prefix, "layer": layer, "hand": hand, "split": "shuffled_labels",
                "n": len(y_reg), "r2": np.nan, "angular_error_deg": np.nan, "clf_accuracy": np.nan,
                "status": "skipped", "note": note,
            })
            continue

        layer_n_components = min(n_components, X.shape[1])
        reg_pred = cv_predict(X, y_reg, make_regression_probe, layer_n_components, n_splits, seed)
        clf_pred = cv_predict(X, y_clf, make_classification_probe, layer_n_components, n_splits, seed)
        r2, ang_err = regression_metrics(y_reg, reg_pred)
        acc = classification_metrics(y_clf, clf_pred)
        rows.append({
            "representation": prefix, "layer": layer, "hand": hand, "split": "shuffled_labels",
            "n": len(y_reg), "r2": r2, "angular_error_deg": ang_err, "clf_accuracy": acc,
            "status": status, "note": note,
        })
    return pd.DataFrame(rows)


def majority_class_baseline(y_clf):
    """Accuracy of always predicting the single most common class -- the
    floor a classification probe needs to beat. (For a perfectly balanced
    12-way task this would be ~8.3%; we compute it from the actual label
    counts rather than assuming exact balance.)"""
    counts = pd.Series(y_clf).value_counts()
    return counts.iloc[0] / counts.sum()


# ---------------------------------------------------------------------------
# Orchestration: run every probe + control, plot, summarize
# ---------------------------------------------------------------------------

def safe_n_components(n_images, n_splits, requested):
    """Cap PCA components so it always fits inside a single CV training
    fold, even if this is run on a small --max_images smoke test."""
    train_size = n_images - n_images // n_splits
    return max(2, min(requested, train_size - 1))


def make_layer_plots(results_df, control_df, out_dir):
    """One PNG per hand: a 2x2 grid of (representation) x (R^2, accuracy),
    each showing the all/correct/wrong split lines across layers plus the
    shuffled-label control as a dashed reference line."""
    for hand in ("minute", "hour"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
        for col, prefix in enumerate(("hidden_last", "hidden_meanpool")):
            sub = results_df[(results_df["hand"] == hand) & (results_df["representation"] == prefix)]
            ctrl = control_df[(control_df["hand"] == hand) & (control_df["representation"] == prefix)]

            ax_r2, ax_acc = axes[0, col], axes[1, col]
            for split_name, color in [("all", "gray"), ("correct", "tab:green"), ("wrong", "tab:red")]:
                s = sub[sub["split"] == split_name]
                ax_r2.plot(s["layer"], s["r2"], label=split_name, color=color, marker="o", markersize=3)
                ax_acc.plot(s["layer"], s["clf_accuracy"], label=split_name, color=color, marker="o", markersize=3)
            ax_r2.plot(ctrl["layer"], ctrl["r2"], "--", color="black", label="shuffled labels", linewidth=1)
            ax_acc.plot(ctrl["layer"], ctrl["clf_accuracy"], "--", color="black", label="shuffled labels", linewidth=1)

            ax_r2.axhline(0, color="lightgray", linewidth=1, zorder=0)
            ax_r2.set_title(prefix)
            ax_r2.set_ylabel("regression R^2 (sin/cos)")
            ax_acc.set_ylabel("classification accuracy")
            ax_acc.set_xlabel("layer")

        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{hand.capitalize()}-hand angle probe performance across layers")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"layers_{hand}.png"), dpi=120)
        plt.close(fig)


def print_summary(results_df, control_df, majority_baselines, index_df, out_dir):
    """Build, print, and save the headline comparison table: for every
    representation x hand, the best layer's R^2/accuracy on all/correct/
    wrong images, next to the shuffled-label control and majority-class
    baseline. The `data` column flags whether that best layer's numbers came
    from clean data, data with NaN/Inf repaired (see `diagnose_and_clean_layer`),
    or (rare) every layer being unusable."""
    lines = ["=== PROBE SUMMARY (best layer per representation x hand) ==="]
    lines.append(
        f"n images: {len(index_df)}   "
        f"minute read correctly in final answer: {index_df['minute_correct'].mean():.1%}   "
        f"hour read correctly: {index_df['hour_correct'].mean():.1%}"
    )
    lines.append("R^2 = 0 is what always predicting the mean angle would score, so it's already")
    lines.append("its own baseline. 'shuffled' re-runs the same probe with labels randomly")
    lines.append("permuted across images, to check the probe isn't just fitting noise.")
    lines.append("'data' flags whether the best layer's numbers came from clean activations")
    lines.append("('ok'), activations with NaN/Inf repaired ('REPAIRED' -- treat with extra")
    lines.append("caution), or found no usable layer at all ('NONE USABLE').")
    lines.append("")

    header = (f"{'representation':<17}{'hand':<8}{'layer':>6}{'data':>10}{'n_all':>7}{'n_corr':>7}{'n_wrong':>8}"
              f"{'R2_all':>8}{'R2_corr':>9}{'R2_wrong':>10}{'R2_shuf':>9}"
              f"{'acc_all':>9}{'acc_corr':>10}{'acc_wrong':>11}{'acc_shuf':>10}{'acc_majrty':>11}")
    lines.append(header)

    summary_rows = []
    notes = []
    for (prefix, hand), group in results_df.groupby(["representation", "hand"], sort=False):
        all_rows = group[(group["split"] == "all") & (group["status"] != "skipped")]
        if len(all_rows) == 0:
            lines.append(f"{prefix:<17}{hand:<8}{'--':>6}{'NONE USABLE':>10}   "
                          f"(every layer was skipped -- see warnings above / --recover)")
            notes.append(f"  NOTE: {prefix}/{hand} had NO usable layers at all; run --recover "
                         "on the saved activations, or re-run extraction.")
            # Still add a row to the CSV (all-NaN numeric columns) so a
            # script reading summary_table.csv sees this combo was attempted
            # and failed, rather than silently missing it.
            summary_rows.append({
                "representation": prefix, "hand": hand, "layer": None, "data_status": "none_usable",
                "n_all": None, "n_correct": None, "n_wrong": None,
                "r2_all": float("nan"), "r2_correct": float("nan"), "r2_wrong": float("nan"),
                "r2_shuffled": float("nan"),
                "acc_all": float("nan"), "acc_correct": float("nan"), "acc_wrong": float("nan"),
                "acc_shuffled": float("nan"), "acc_majority_baseline": majority_baselines[hand],
            })
            continue

        best_layer = all_rows.loc[all_rows["r2"].idxmax(), "layer"]
        at_best = group[group["layer"] == best_layer].set_index("split")
        data_status = at_best.loc["all", "status"]  # same for every split of a given layer

        ctrl_at_best = control_df[(control_df["representation"] == prefix) &
                                   (control_df["hand"] == hand) &
                                   (control_df["layer"] == best_layer)]
        r2_shuf = ctrl_at_best["r2"].iloc[0] if len(ctrl_at_best) else float("nan")
        acc_shuf = ctrl_at_best["clf_accuracy"].iloc[0] if len(ctrl_at_best) else float("nan")

        row = {
            "representation": prefix, "hand": hand, "layer": int(best_layer), "data_status": data_status,
            "n_all": int(at_best.loc["all", "n"]),
            "n_correct": int(at_best.loc["correct", "n"]),
            "n_wrong": int(at_best.loc["wrong", "n"]),
            "r2_all": at_best.loc["all", "r2"], "r2_correct": at_best.loc["correct", "r2"],
            "r2_wrong": at_best.loc["wrong", "r2"], "r2_shuffled": r2_shuf,
            "acc_all": at_best.loc["all", "clf_accuracy"], "acc_correct": at_best.loc["correct", "clf_accuracy"],
            "acc_wrong": at_best.loc["wrong", "clf_accuracy"], "acc_shuffled": acc_shuf,
            "acc_majority_baseline": majority_baselines[hand],
        }
        summary_rows.append(row)

        data_label = "REPAIRED" if data_status == "repaired" else "ok"
        lines.append(
            f"{prefix:<17}{hand:<8}{row['layer']:>6}{data_label:>10}"
            f"{row['n_all']:>7}{row['n_correct']:>7}{row['n_wrong']:>8}"
            f"{row['r2_all']:>8.2f}{row['r2_correct']:>9.2f}{row['r2_wrong']:>10.2f}{row['r2_shuffled']:>9.2f}"
            f"{row['acc_all']:>9.1%}{row['acc_correct']:>10.1%}{row['acc_wrong']:>11.1%}"
            f"{row['acc_shuffled']:>10.1%}{row['acc_majority_baseline']:>11.1%}"
        )
        if data_status == "repaired":
            note = at_best.loc["all", "note"]
            notes.append(f"  NOTE: {prefix}/{hand} best layer ({int(best_layer)}) used REPAIRED data: {note}")
        if row["n_correct"] < 20:
            notes.append(
                f"  NOTE: {prefix}/{hand} has only {row['n_correct']} 'correct' images -- "
                "its correct-group number is high-variance, interpret with caution."
            )

    if notes:
        lines.append("")
        lines.extend(notes)

    text = "\n".join(lines)
    print("\n" + text)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)
    return summary_df


def _load_representation_array(activations_dir, representation):
    """Load one saved activation array by representation name, always
    shaped (N, layers, H) -- vision_encoder gets a size-1 "layers" axis
    added so every caller can use the same per-layer code regardless of
    representation. Loaded as-saved (float32 from a current
    extract_activations run, or possibly float16 from an older one) --
    diagnose_and_clean_layer casts each layer up to float64 itself, so we
    don't need to touch dtype here.
    """
    filename = {
        "hidden_last": "hidden_last.npy",
        "hidden_meanpool": "hidden_meanpool.npy",
        "vision_encoder": "vision_meanpool.npy",
    }[representation]
    arr = np.load(os.path.join(activations_dir, filename))
    if representation == "vision_encoder":
        arr = arr[:, None, :]
    return arr


def run_probing(activations_dir=ACTIVATIONS_DIR, out_dir=RESULTS_DIR,
                 n_components=N_COMPONENTS, n_splits=N_SPLITS, seed=SEED):
    index_df = pd.read_csv(os.path.join(activations_dir, "index.csv"))
    hidden_last = _load_representation_array(activations_dir, "hidden_last")
    hidden_meanpool = _load_representation_array(activations_dir, "hidden_meanpool")
    vision_meanpool = _load_representation_array(activations_dir, "vision_encoder")

    n_components = safe_n_components(len(index_df), n_splits, n_components)
    targets = build_targets(index_df)

    # Correct/wrong is defined PER HAND: the minute-hand probe splits by
    # whether the model's predicted minute matched the true minute
    # (regardless of the hour digit it gave), and likewise for hour. This is
    # both a more direct test of "was THIS hand read correctly" and much
    # better powered than requiring the full HH:MM to match exactly, which
    # our Step 1 behavior check found happens on only ~2% of images.
    masks = {
        "minute": index_df["minute_correct"].to_numpy(),
        "hour": index_df["hour_correct"].to_numpy(),
    }
    representations = {
        "hidden_last": hidden_last,
        "hidden_meanpool": hidden_meanpool,
        "vision_encoder": vision_meanpool,
    }

    # Clean each representation's layers ONCE (checks for NaN/Inf, repairs
    # or skips as needed, drops zero-variance columns) and reuse that across
    # both hands and both the real and shuffled-label probes below, instead
    # of re-diagnosing the same layer four times over.
    cleaned_by_rep = {prefix: clean_representation(acts, prefix) for prefix, acts in representations.items()}

    result_frames, control_frames = [], []
    for prefix, cleaned_layers in cleaned_by_rep.items():
        for hand in ("minute", "hour"):
            print(f"Probing {prefix} / {hand} hand ({len(cleaned_layers)} layer(s)) ...")
            result_frames.append(probe_one_representation(
                cleaned_layers, prefix, targets, hand, masks[hand], n_components, n_splits, seed))
            control_frames.append(shuffled_label_control(
                cleaned_layers, prefix, targets, hand, n_components, n_splits, seed))

    results_df = pd.concat(result_frames, ignore_index=True)
    control_df = pd.concat(control_frames, ignore_index=True)
    majority_baselines = {
        hand: majority_class_baseline(targets[hand]["classification"]) for hand in ("minute", "hour")
    }

    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, "per_layer_results.csv"), index=False)
    control_df.to_csv(os.path.join(out_dir, "shuffled_label_control.csv"), index=False)

    make_layer_plots(results_df, control_df, out_dir)
    summary_df = print_summary(results_df, control_df, majority_baselines, index_df, out_dir)

    print(f"\nFull per-layer results, plots, and summary written to '{out_dir}/'.")
    return results_df, control_df, summary_df


def _repair_slice(X):
    """Replace non-finite values in a single (N, H) slice with that column's
    mean, computed from its finite entries. A column with NO finite values
    anywhere falls back to 0.0 (there's nothing to average). Returns
    (X_repaired, n_bad_values, n_all_bad_columns)."""
    finite = np.isfinite(X)
    n_bad = int((~finite).sum())
    if n_bad == 0:
        return X, 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_mean = np.nanmean(np.where(finite, X, np.nan), axis=0)
    all_bad_cols = ~np.isfinite(col_mean)
    col_mean = np.where(all_bad_cols, 0.0, col_mean)
    X_fixed = np.where(finite, X, np.broadcast_to(col_mean, X.shape))
    return X_fixed, n_bad, int(all_bad_cols.sum())


def recover_activations(activations_dir=ACTIVATIONS_DIR):
    """Repair already-saved activation .npy files IN PLACE, without redoing
    the (slow, GPU) extraction: cast to float32, and replace any non-finite
    (NaN/Inf) value with that column's mean (computed per layer, from the
    finite entries in that same column; a column with no finite values
    anywhere falls back to 0.0, since there's nothing to average).

    This fixes files that were saved by an older/buggy extraction run (e.g.
    one that stored float16, which can silently overflow to inf on some
    hidden-state dimensions). `run_probing` also does its own per-layer
    NaN/Inf handling on load regardless, so this is a convenience to avoid
    re-extracting -- not a strict prerequisite for probing to work.
    """
    print(f"Recovering activations in '{activations_dir}/' ...")
    for name in ("hidden_last.npy", "hidden_meanpool.npy", "vision_meanpool.npy"):
        path = os.path.join(activations_dir, name)
        if not os.path.exists(path):
            print(f"  {name}: not found, skipping")
            continue

        arr = np.load(path).astype(np.float64)
        total_bad, total_all_bad_cols = 0, 0

        if arr.ndim == 3:
            for layer in range(arr.shape[1]):
                fixed, n_bad, n_all_bad = _repair_slice(arr[:, layer, :])
                arr[:, layer, :] = fixed
                total_bad += n_bad
                total_all_bad_cols += n_all_bad
        else:
            arr, total_bad, total_all_bad_cols = _repair_slice(arr)

        arr = arr.astype(np.float32)
        np.save(path, arr)

        if total_bad:
            msg = f"  {name}: repaired {total_bad} non-finite value(s)"
            if total_all_bad_cols:
                msg += f" ({total_all_bad_cols} column(s) had no finite values at all, filled with 0.0)"
            print(msg + f"; re-saved as float32 {arr.shape}")
        else:
            print(f"  {name}: no non-finite values found; re-saved as float32 {arr.shape}")
    print("Done. You can now run `python probe.py --stage probe` without re-extracting.")


# ---------------------------------------------------------------------------
# STAGE "direction": fit + save a steering direction, for intervene.py
# ---------------------------------------------------------------------------

def fit_probe_direction(activations_dir=ACTIVATIONS_DIR, out_dir=RESULTS_DIR,
                         representation="hidden_last", hand="minute",
                         layer=None, n_components=N_COMPONENTS, seed=SEED):
    """Refit the regression probe for one (representation, hand) at ONE
    layer, on ALL available images (no cross-validation split -- for a
    causal-intervention experiment we want the single best-fit probe, not a
    held-out generalization estimate), and save it as a STEERING DIRECTION
    in raw activation space, so intervene.py can load and use it without
    needing scikit-learn itself.

    If `layer` is None, reads probe_output/probe_results/per_layer_results.csv
    (must already exist -- i.e. run `--stage probe` first) and auto-picks
    the layer with the highest R^2 on split='all' for this representation/hand.
    """
    index_df = pd.read_csv(os.path.join(activations_dir, "index.csv"))
    acts = _load_representation_array(activations_dir, representation)

    if layer is None:
        results_path = os.path.join(out_dir, "per_layer_results.csv")
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"{results_path} not found -- run `python probe.py --stage probe` first, "
                "or pass --direction_layer explicitly.")
        results_df = pd.read_csv(results_path)
        sub = results_df[(results_df["representation"] == representation) &
                          (results_df["hand"] == hand) & (results_df["split"] == "all") &
                          (results_df["status"] != "skipped")]
        if len(sub) == 0:
            raise ValueError(f"No usable layers found for {representation}/{hand} in {results_path}.")
        layer = int(sub.loc[sub["r2"].idxmax(), "layer"])
        print(f"Auto-selected layer {layer} (highest R^2 on the 'all' split) for {representation}/{hand}.")

    X_raw = acts[:, layer, :].astype(np.float64)
    X_clean, status, note, kept_mask = diagnose_and_clean_layer(X_raw, f"{representation} layer {layer}")
    if status == "skipped":
        raise ValueError(f"Layer {layer} of {representation} has no usable data: {note}")

    targets = build_targets(index_df)
    y = targets[hand]["regression"]  # (N, 2): sin, cos of the true angle

    n_components_eff = min(n_components, X_clean.shape[1], X_clean.shape[0] - 1)
    scaler = StandardScaler().fit(X_clean)
    X_scaled = scaler.transform(X_clean)
    pca = PCA(n_components=n_components_eff, random_state=seed).fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    ridge = Ridge(alpha=10.0).fit(X_pca, y)
    r2_train = ridge.score(X_pca, y)  # train-set R^2 -- a sanity number, not held-out

    # The fitted pipeline is a LINEAR map from raw activations to (sin, cos):
    #   x_scaled = (x - scaler.mean_) / scaler.scale_
    #   z        = (x_scaled - pca.mean_) @ pca.components_.T
    #   y_pred   = z @ ridge.coef_.T + ridge.intercept_
    # Chaining these (chain rule) gives the Jacobian dy_pred/dx in RAW
    # activation space -- its two columns are the directions that increase
    # the sin-readout and cos-readout, respectively. This is exactly what
    # "steer along the probe direction" means: moving activations along a
    # combination of these two columns changes what the probe reads off as
    # the angle.
    M = pca.components_.T @ ridge.coef_.T           # (n_kept_features, 2)
    J = M / scaler.scale_[:, None]                   # (n_kept_features, 2)

    # Scatter back into the FULL raw activation space (kept_mask marks which
    # of the original columns survived cleaning; dropped columns get
    # direction 0, since the probe never looked at them).
    n_features_full = acts.shape[2]
    w_sin_full = np.zeros(n_features_full)
    w_cos_full = np.zeros(n_features_full)
    w_sin_full[kept_mask] = J[:, 0]
    w_cos_full[kept_mask] = J[:, 1]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"probe_direction_{representation}_{hand}.npz")
    np.savez(
        out_path,
        representation=representation, hand=hand, layer=layer,
        w_sin_full=w_sin_full, w_cos_full=w_cos_full, kept_mask=kept_mask,
        n_components=n_components_eff, r2_train=r2_train, n_images=len(X_clean),
        # Raw pipeline parameters too (in the kept-columns, reduced space) --
        # lets a caller recompute the EXACT predicted angle from any raw
        # hidden-state vector via `apply_saved_probe`, as a rigorous check
        # that a steering intervention actually moved the probe's readout
        # (rather than trusting the linear Jacobian approximation above,
        # which is exact for this pipeline but good to double-check).
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        pca_components=pca.components_, pca_mean=pca.mean_,
        ridge_coef=ridge.coef_, ridge_intercept=ridge.intercept_,
    )
    print(f"Saved steering direction for {representation}/{hand} at layer {layer} "
          f"(train R^2={r2_train:.3f}, n={len(X_clean)}) to '{out_path}'.")
    return out_path


def load_probe_direction(path):
    """Load a direction saved by `fit_probe_direction`, as a plain dict of
    numpy arrays / native Python scalars (unwraps 0-d arrays with .item())."""
    data = np.load(path, allow_pickle=False)
    return {k: (data[k].item() if data[k].shape == () else data[k]) for k in data.files}


def apply_saved_probe(direction_data, h_full):
    """Given a raw (n_features_full,) hidden-state vector `h_full` and a
    direction dict from `load_probe_direction`, recompute the probe's
    predicted angle (in degrees) using the exact saved pipeline parameters.
    Used by intervene.py to verify a steering intervention actually moved
    the represented angle, independent of whether the model's stated
    answer changed."""
    mask = direction_data["kept_mask"]
    h = h_full[mask]
    h_scaled = (h - direction_data["scaler_mean"]) / direction_data["scaler_scale"]
    z = (h_scaled - direction_data["pca_mean"]) @ direction_data["pca_components"].T
    y = z @ direction_data["ridge_coef"].T + direction_data["ridge_intercept"]
    return float(np.degrees(np.arctan2(y[0], y[1])))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Probe Qwen2.5-VL-3B's hidden states for the true clock-hand angles.")
    parser.add_argument("--stage", choices=["extract", "probe", "direction", "all"], default="all",
                         help="'extract': GPU activation extraction only. 'probe': CPU probing + "
                              "analysis only, from already-saved activations. 'direction': refit one "
                              "probe on all images and save it as a steering direction for "
                              "intervene.py (run this explicitly -- 'all' does NOT include it, since "
                              "it's a separate downstream need, not part of the standard pipeline). "
                              "'all': extract + probe.")
    parser.add_argument("--recover", action="store_true",
                         help="repair already-saved activations in place (cast to float32, impute "
                              "non-finite values with the column mean) instead of running a stage; "
                              "ignores --stage. Run '--stage probe' afterward.")
    parser.add_argument("--data_csv", type=str, default=DATA_CSV)
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--activations_dir", type=str, default=ACTIVATIONS_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--max_images", type=int, default=None,
                         help="only process the first N images (for a quick smoke test)")
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS,
                         help="PCA components kept before the linear probe")
    parser.add_argument("--n_splits", type=int, default=N_SPLITS, help="cross-validation folds")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--direction_representation", type=str, default="hidden_last",
                         choices=["hidden_last", "hidden_meanpool", "vision_encoder"],
                         help="(--stage direction) which saved representation to fit the steering direction on")
    parser.add_argument("--direction_hand", type=str, default="minute", choices=["minute", "hour"],
                         help="(--stage direction) which hand's angle to fit the steering direction for")
    parser.add_argument("--direction_layer", type=int, default=None,
                         help="(--stage direction) layer to fit at; default: auto-pick the best layer "
                              "from per_layer_results.csv (requires --stage probe to have run first)")
    args = parser.parse_args()

    if args.recover:
        recover_activations(activations_dir=args.activations_dir)
        return

    if args.stage in ("extract", "all"):
        extract_activations(data_csv=args.data_csv, images_dir=args.images_dir,
                             out_dir=args.activations_dir, model_id=args.model_id,
                             max_images=args.max_images, seed=args.seed)

    if args.stage in ("probe", "all"):
        run_probing(activations_dir=args.activations_dir, out_dir=args.results_dir,
                    n_components=args.n_components, n_splits=args.n_splits, seed=args.seed)

    if args.stage == "direction":
        fit_probe_direction(activations_dir=args.activations_dir, out_dir=args.results_dir,
                             representation=args.direction_representation, hand=args.direction_hand,
                             layer=args.direction_layer, n_components=args.n_components, seed=args.seed)


if __name__ == "__main__":
    main()
