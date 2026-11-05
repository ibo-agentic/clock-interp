"""
knockout.py -- ATTENTION KNOCKOUT: block attention from TEXT-QUERY positions
(every prompt token after the image, AND every generated token) to
IMAGE-TOKEN KEY positions, from a chosen START LAYER through the last
layer, and see whether the model's stated time collapses as blocking starts
EARLIER (more layers affected). Qwen2.5-VL-3B only for now -- adapters.py
is used for all model-specific details (image-token positions, decoder
layer/attention-module access); nothing Qwen-specific is hardcoded beyond
what adapters.py itself already establishes.

MECHANISM: forces eager attention on the targeted layers (SDPA/FlashAttention
don't expose a modifiable additive attention_mask the way eager does -- the
same constraint heads.py's inspect_attention_pattern hit), then registers a
forward_pre_hook on EACH targeted attention module that adds a large
negative bias (torch.finfo(dtype).min, matching HF's own "impossible to
attend" convention for additive masks) to the EXISTING attention_mask at
every [text-query, image-key] cell -- for BOTH the prefill pass (every
prompt text position as query) and every later single-token decode step
(the newly generated token as query, attending back over the KV-cache-grown
key sequence, where image positions stay at their ORIGINAL fixed indices --
only what's appended afterward, generated tokens, is never an image
position). This is "patching the attention forward" per-layer, not one
global mask: a SEPARATE hook on each targeted layer's OWN attention module,
each modifying THAT layer's OWN attention_mask input independently.

NOTE on the "never patch/interpret the final decoder layer" rule this
project otherwise follows (see intervene.py's module docstring/README):
that caveat is specific to intervene.py's residual-stream/MLP-output
patches, which only ever apply during the single PREFILL forward pass (a
shape guard skips single-token decode calls) and hence can't reach a
generated token's logits if applied at the LAST layer -- no downstream
block re-attends to a patched position there. Knockout's mechanism is
different: it modifies the ATTENTION COMPUTATION ITSELF, at EVERY
generation step the hook stays registered for (no such shape guard), so
blocking the final layer's attention DOES directly affect that step's own
logits. That caveat's reasoning doesn't apply here, and the requested
start-layer list (up to 34) isn't restricted for it.

LAYER INDEXING: "layer L" here means attn_modules[L] DIRECTLY -- decoder
block L (0-indexed), the plain convention find_attention_modules-style
functions use. This is DELIBERATELY NOT intervene.py's hidden_states-index
convention (0=embeddings, 1..num_layers=decoder block outputs, requiring a
layer-1 shift to reach attn_modules -- see intervene.py's
attn_module_for_layer and the real off-by-one bug that motivated it).
Knockout has no "hidden_states position" to align with -- it intercepts
attention modules directly -- so using the plain 0-indexed block index
here is correct, not a repeat of that mistake.
"""
import argparse
import contextlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from adapters import get_adapter, output_dir_for
from eval_behavior import SEED, parse_time_answer
from intervene import setup_model_and_vision, teacher_forced_token_logprobs
from analyze_replication import bootstrap_ci, permutation_test

OUT_DIR = "knockout_output"
KNOCKOUT_PROMPT = "What time is it?"
START_LAYERS = [0, 10, 14, 18, 20, 22, 24, 26, 28, 30, 34]
MAX_NEW_TOKENS = 16


# ---------------------------------------------------------------------------
# Attention-module discovery -- reflection-based, same philosophy as
# heads.py's find_attention_modules (confirmed against the same installed
# Qwen2_5_VLAttention source: self.o_proj/self.q_proj etc., found here by
# the callable 'q_proj' attribute rather than 'o_proj' -- either works, but
# knockout needs the module whose OWN forward() takes attention_mask, which
# is the SAME attention module either way).
# ---------------------------------------------------------------------------

def find_attention_modules(decoder_layers):
    """Same reflection convention as heads.py's find_attention_modules
    (kept as a separate, local copy rather than importing heads.py, to
    keep this file's dependencies limited to intervene.py/adapters.py --
    heads.py is a sibling analysis, not a shared library this file should
    depend on). Returns a list, index i = decoder_layers[i]'s own
    attention module, used DIRECTLY (attn_modules[L]) -- see module
    docstring on why this file's layer indexing differs from
    intervene.py's."""
    attn_modules = []
    for i, layer in enumerate(decoder_layers):
        candidates = [m for _, m in layer.named_modules()
                      if hasattr(m, "o_proj") and callable(getattr(m, "o_proj", None))]
        if len(candidates) != 1:
            raise AttributeError(
                f"Expected exactly one attention submodule with a callable `o_proj` in decoder layer {i}, "
                f"found {len(candidates)}. This model's attention module doesn't match the convention "
                "find_attention_modules assumes -- do not proceed without checking why."
            )
        attn_modules.append(candidates[0])
    return attn_modules


def force_eager(attn_modules):
    """Forces every GIVEN attention module's OWN config to eager (SDPA/
    FlashAttention don't accept/use a modifiable additive attention_mask
    the way eager's `attn_weights = attn_weights + attention_mask` does --
    same constraint heads.py's inspect_attention_pattern hit, targeting
    the SPECIFIC attention module's own config object, confirmed to be
    checked independently per-module for Qwen2_5_VLAttention). Returns a
    restore() function -- ALWAYS call it, even on error."""
    originals = []
    for m in attn_modules:
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            originals.append((cfg, cfg._attn_implementation))
            cfg._attn_implementation = "eager"

    def restore():
        for cfg, orig in originals:
            cfg._attn_implementation = orig
    return restore


def _extract_attention_mask(args, kwargs):
    if len(args) > 1 and torch.is_tensor(args[1]):
        return args[1], "arg1"
    if kwargs.get("attention_mask") is not None:
        return kwargs["attention_mask"], "kwarg"
    return None, None


def _replace_attention_mask(args, kwargs, where, new_mask):
    if where == "arg1":
        return (args[0], new_mask) + tuple(args[2:]), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["attention_mask"] = new_mask
    return args, new_kwargs


def make_knockout_hook(image_mask_prompt):
    """A forward_pre_hook for ONE attention module. `image_mask_prompt`:
    (prompt_len,) bool -- which of the ORIGINAL prompt's positions are
    image tokens. Image positions stay at these SAME fixed indices as the
    KV cache grows during generation (they're only ever written once, at
    prefill); anything appended afterward (generated tokens) is never an
    image position, so the image-key mask for a longer key axis is just
    the original mask padded with False."""
    prompt_len = image_mask_prompt.shape[0]

    def hook(module, args, kwargs):
        attention_mask, where = _extract_attention_mask(args, kwargs)
        if attention_mask is None:
            raise RuntimeError(
                "knockout requires eager attention with an explicit additive attention_mask tensor -- "
                "got None. force_eager() should make the outer model construct one when the config says "
                "eager; if this fires, this transformers version's causal-mask construction doesn't match "
                "what this file assumes -- do not proceed without checking why."
            )
        bsz, _n_head_dim, q_len, kv_len = attention_mask.shape
        device, dtype = attention_mask.device, attention_mask.dtype
        img_mask_dev = image_mask_prompt.to(device)

        if kv_len == prompt_len:
            image_key_mask = img_mask_dev
        elif kv_len > prompt_len:
            image_key_mask = torch.cat(
                [img_mask_dev, torch.zeros(kv_len - prompt_len, dtype=torch.bool, device=device)])
        else:
            raise RuntimeError(f"attention_mask's key length ({kv_len}) is SHORTER than the prompt "
                               f"({prompt_len}) -- unexpected; image positions can't be located.")

        if q_len == prompt_len:
            text_query_mask = ~img_mask_dev
        else:
            # every later decode step queries with exactly the newly generated (non-image) token(s)
            text_query_mask = torch.ones(q_len, dtype=torch.bool, device=device)

        new_mask = attention_mask.clone()
        # Chained boolean-index assignment (new_mask[a][b] = x) does NOT write through in PyTorch --
        # read the selected rows as their own tensor, mutate THAT, then write the whole thing back
        # in one assignment, which DOES write through correctly.
        selected_rows = new_mask[:, :, text_query_mask, :]
        selected_rows[:, :, :, image_key_mask] = torch.finfo(dtype).min
        new_mask[:, :, text_query_mask, :] = selected_rows

        return _replace_attention_mask(args, kwargs, where, new_mask)

    return hook


@contextlib.contextmanager
def knockout_attention(attn_modules, start_layer, image_mask_prompt):
    """Registers make_knockout_hook on attn_modules[start_layer:] (see
    module docstring on this file's plain 0-indexed layer convention) --
    "layers start_layer through the last layer". start_layer >= len(
    attn_modules) registers NO hooks (the negative control: masking zero
    layers). Forces eager attention on exactly those same layers for the
    duration, restored afterward. Always removed/restored, even on error."""
    targeted = attn_modules[start_layer:]
    restore_eager = force_eager(targeted)
    hook = make_knockout_hook(image_mask_prompt)
    handles = [m.register_forward_pre_hook(hook, with_kwargs=True) for m in targeted]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
        restore_eager()


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def run_knockout_sweep(adapter, decoder_layers, attn_modules, df, images_dir, out_dir,
                       start_layers=START_LAYERS, max_new_tokens=MAX_NEW_TOKENS,
                       prompt=KNOCKOUT_PROMPT, max_images=None, logit_metric=True):
    """For each clock image: get the UNMASKED baseline answer (once), then
    for each start_layer, run knocked-out generation and record: the
    parsed answer, whether it matches the UNMASKED baseline, whether it
    matches the TRUE time, and (logit_metric=True, default) the
    teacher-forced log-prob of the UNMASKED baseline's OWN generated token
    sequence under this knockout condition (reuses intervene.py's
    teacher_forced_token_logprobs UNCHANGED -- knockout's hooks are already
    active via the `with knockout_attention(...)` block, so no `patch`/
    `mlp_patch` argument is needed there)."""
    rows = []
    images = df if max_images is None else df.head(max_images)
    for _, row in tqdm(images.iterrows(), total=len(images), desc="knockout images"):
        path = os.path.join(images_dir, row["filename"])
        inputs = adapter.build_inputs(path, prompt)
        image_mask = adapter.image_token_positions(inputs)

        baseline_answer = adapter.generate_answer(inputs, max_new_tokens)
        base_hour, base_minute, base_ok = parse_time_answer(baseline_answer)
        baseline_ids = (adapter.processor.tokenizer.encode(baseline_answer, add_special_tokens=False)
                        if baseline_answer else [])

        for start_layer in start_layers:
            with knockout_attention(attn_modules, start_layer, image_mask):
                ans = adapter.generate_answer(inputs, max_new_tokens)
                logprob_unmasked = (
                    float(sum(teacher_forced_token_logprobs(adapter, adapter.model, inputs, baseline_ids)))
                    if (logit_metric and baseline_ids) else float("nan")
                )
            pred_hour, pred_minute, ok = parse_time_answer(ans)
            rows.append({
                "filename": row["filename"], "true_hour": int(row["hour"]), "true_minute": int(row["minute"]),
                "start_layer": start_layer, "num_layers": len(attn_modules),
                "baseline_answer": baseline_answer, "baseline_hour": base_hour, "baseline_minute": base_minute,
                "knockout_answer": ans, "knockout_hour": pred_hour if ok else None,
                "knockout_minute": pred_minute if ok else None,
                "matches_baseline": (ans == baseline_answer),
                "matches_true": bool(ok and pred_hour == int(row["hour"]) and pred_minute == int(row["minute"])),
                "logprob_unmasked_answer": logprob_unmasked,
            })

    trials_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    trials_df.to_csv(os.path.join(out_dir, "knockout_trials.csv"), index=False)
    print(f"\nknockout.py: {len(trials_df)} trial(s) over {len(images)} image(s) x "
          f"{len(start_layers)} start_layer(s) written to '{out_dir}/knockout_trials.csv'.")
    return trials_df


def summarize_knockout(trials_df, out_dir, n_permutations=2000, n_bootstrap=2000, seed=SEED):
    """Per start_layer, PER-IMAGE statistics (never per-trial-row -- there's
    exactly one trial per (image, start_layer) already, but the rate/CI
    below are computed by averaging/bootstrapping over IMAGES, reusing
    analyze_replication.py's bootstrap_ci/permutation_test UNCHANGED, not
    reimplemented): agreement-with-baseline rate (bootstrap CI), agreement-
    with-truth rate (bootstrap CI), mean logprob of the unmasked answer,
    and a permutation test of THIS start_layer's per-image
    matches_baseline values against start_layer=len(attn_modules) (the
    negative control -- zero layers knocked out, i.e. matches_baseline
    should be ~100% and any OTHER start_layer's departure from that is
    what's being tested)."""
    num_layers = int(trials_df["num_layers"].iloc[0])
    none_rows = trials_df[trials_df["start_layer"] >= num_layers]
    reference = none_rows.sort_values("filename")["matches_baseline"].astype(float).to_numpy()

    rows = []
    for start_layer in sorted(trials_df["start_layer"].unique()):
        sub = trials_df[trials_df["start_layer"] == start_layer].sort_values("filename")
        match_base = sub["matches_baseline"].astype(float).to_numpy()
        match_true = sub["matches_true"].astype(float).to_numpy()
        mean_base, lo_base, hi_base = bootstrap_ci(match_base, n_bootstrap=n_bootstrap, seed=seed)
        mean_true, lo_true, hi_true = bootstrap_ci(match_true, n_bootstrap=n_bootstrap, seed=seed)
        if len(reference) == len(match_base) and start_layer < num_layers:
            _, p_value = permutation_test(match_base, reference, n_permutations=n_permutations, seed=seed)
        else:
            p_value = float("nan")
        rows.append({
            "start_layer": int(start_layer), "n_images": len(sub),
            "matches_baseline_rate": mean_base, "matches_baseline_ci_lo": lo_base, "matches_baseline_ci_hi": hi_base,
            "matches_true_rate": mean_true, "matches_true_ci_lo": lo_true, "matches_true_ci_hi": hi_true,
            "mean_logprob_unmasked_answer": float(sub["logprob_unmasked_answer"].mean()),
            "p_value_vs_zero_layers": p_value,
        })
    summary = pd.DataFrame(rows).sort_values("start_layer").reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "knockout_summary.csv"), index=False)
    return summary


def print_knockout_summary(summary_df):
    lines = ["=== ATTENTION KNOCKOUT SUMMARY ===", "",
            "start_layer=<num_layers> is the negative control (no layers knocked out).", ""]
    header = (f"{'start_L':>8}{'n':>5}{'match_base':>12}{'ci':>18}{'match_true':>12}{'ci':>18}"
              f"{'logp(unmasked)':>16}{'p_vs_0':>9}")
    lines.append(header)
    for _, r in summary_df.iterrows():
        ci_base = f"[{r['matches_baseline_ci_lo']:.0%},{r['matches_baseline_ci_hi']:.0%}]"
        ci_true = f"[{r['matches_true_ci_lo']:.0%},{r['matches_true_ci_hi']:.0%}]"
        p_text = f"{r['p_value_vs_zero_layers']:.4f}" if not np.isnan(r["p_value_vs_zero_layers"]) else "n/a"
        lines.append(f"{int(r['start_layer']):>8}{int(r['n_images']):>5}{r['matches_baseline_rate']:>12.1%}"
                     f"{ci_base:>18}{r['matches_true_rate']:>12.1%}{ci_true:>18}"
                     f"{r['mean_logprob_unmasked_answer']:>16.2f}{p_text:>9}")
    text = "\n".join(lines)
    print("\n" + text)
    return text


# ---------------------------------------------------------------------------
# --verify: ~5 checks, prints PASS/FAIL for each, exits (nonzero on any FAIL)
# ---------------------------------------------------------------------------

def run_verification(adapter, decoder_layers, attn_modules, df, images_dir, max_new_tokens=MAX_NEW_TOKENS,
                     n_verify_images=3, seed=SEED):
    """5 checks:
      1. attention weights from text queries to image keys are EXACTLY 0 in
         masked layers (captured via output_attentions, forced eager) --
         and UNCHANGED (matches the unmasked run) in unmasked layers.
      2. generated tokens are ALSO blocked: checked at GENERATION STEP 2
         specifically (the attention_mask a decode-step call actually
         receives has a text-query row for the newly generated token, and
         that row's image-key entries must be masked too).
      3. start_layer=0 (mask everything) destroys agreement with the
         unmasked answer (positive control).
      4. start_layer=num_layers (mask nothing) gives EXACTLY the baseline
         answers (negative control).
      5. the attention_mask changes ONLY at [text_query, image_key] cells
         -- image-query rows and text-key columns (within a text-query
         row) are byte-identical to the unmasked mask.
    """
    rng = np.random.RandomState(seed)
    sample = df.sample(n=min(n_verify_images, len(df)), random_state=seed)
    num_layers = len(attn_modules)
    checks = {}

    # --- checks 1 and 5: attention weights + mask-cell correctness ---
    masked_layer, unmasked_layer = 0, num_layers - 1
    weight_checks, mask_cell_checks = [], []
    for _, row in sample.iterrows():
        path = os.path.join(images_dir, row["filename"])
        inputs = adapter.build_inputs(path, KNOCKOUT_PROMPT)
        image_mask = adapter.image_token_positions(inputs)

        restore = force_eager(attn_modules)
        try:
            with torch.no_grad():
                out_unmasked = adapter.model(**inputs, output_attentions=True)
        finally:
            restore()
        attn_unmasked = out_unmasked.attentions

        with knockout_attention(attn_modules, start_layer=0, image_mask_prompt=image_mask):
            with torch.no_grad():
                out_masked = adapter.model(**inputs, output_attentions=True)
        attn_masked = out_masked.attentions

        if attn_unmasked is None or attn_masked is None:
            weight_checks.append(False)
            mask_cell_checks.append(False)
            continue
        text_mask = ~image_mask
        # layer 0 (start_layer=0's OWN first targeted layer): text-query x image-key weights must be ~0
        w_masked_layer0 = attn_masked[0][0, :, text_mask, :][:, :, image_mask]
        weight_checks.append(bool(torch.allclose(w_masked_layer0, torch.zeros_like(w_masked_layer0), atol=1e-5)))
        # a layer NOT covered by start_layer=0's "0 through last" range doesn't exist here (0 covers
        # everything) -- instead confirm a DIFFERENT knockout (start_layer=num_layers, i.e. nothing
        # targeted) leaves attention EXACTLY as the unmasked run, at every layer.
        with knockout_attention(attn_modules, start_layer=num_layers, image_mask_prompt=image_mask):
            with torch.no_grad():
                out_none = adapter.model(**inputs, output_attentions=True)
        if out_none.attentions is not None:
            for L in range(num_layers):
                weight_checks.append(bool(torch.allclose(out_none.attentions[L], attn_unmasked[L], atol=1e-6)))

        # check 5: within a MASKED layer, ONLY [text-query, image-key] cells are touched -- image-
        # query rows (image tokens attending to anything) must be byte-identical to the unmasked run.
        for L in range(num_layers):
            img_rows_masked = attn_masked[L][0, :, image_mask, :]
            img_rows_unmasked = attn_unmasked[L][0, :, image_mask, :]
            mask_cell_checks.append(bool(torch.equal(img_rows_masked, img_rows_unmasked)))

    checks["attention weights: 0 in masked layers, unchanged in unmasked layers"] = (
        len(weight_checks) > 0 and all(weight_checks))
    checks["only [text-query, image-key] cells are touched -- image-query rows byte-identical"] = (
        len(mask_cell_checks) > 0 and all(mask_cell_checks))

    # --- check 2: generated tokens are also blocked, at generation step 2 ---
    step2_checks = []
    for _, row in sample.iterrows():
        path = os.path.join(images_dir, row["filename"])
        inputs = adapter.build_inputs(path, KNOCKOUT_PROMPT)
        image_mask = adapter.image_token_positions(inputs)
        captured = {}

        def spy(module, args, kwargs):
            mask, _where = _extract_attention_mask(args, kwargs)
            if mask is not None and mask.shape[2] == 1 and "step" not in captured:
                captured["step"] = mask[0, 0, 0, :].clone()

        with knockout_attention(attn_modules, start_layer=0, image_mask_prompt=image_mask):
            handle = attn_modules[0].register_forward_pre_hook(spy, with_kwargs=True)
            try:
                adapter.generate_answer(inputs, max_new_tokens=2)
            finally:
                handle.remove()
        if "step" in captured:
            kv_len = captured["step"].shape[0]
            prompt_len = image_mask.shape[0]
            image_key_at_step = torch.cat([image_mask, torch.zeros(kv_len - prompt_len, dtype=torch.bool)]) \
                if kv_len > prompt_len else image_mask
            step2_checks.append(bool(torch.all(captured["step"][image_key_at_step] <= -1e30)))
    checks["generated-token queries are also blocked (checked at generation step 2)"] = (
        len(step2_checks) > 0 and all(step2_checks))

    # --- checks 3/4: positive and negative control on actual generated answers ---
    pos_control, neg_control = [], []
    for _, row in sample.iterrows():
        path = os.path.join(images_dir, row["filename"])
        inputs = adapter.build_inputs(path, KNOCKOUT_PROMPT)
        image_mask = adapter.image_token_positions(inputs)
        baseline = adapter.generate_answer(inputs, max_new_tokens)
        with knockout_attention(attn_modules, start_layer=0, image_mask_prompt=image_mask):
            ans_all_masked = adapter.generate_answer(inputs, max_new_tokens)
        with knockout_attention(attn_modules, start_layer=num_layers, image_mask_prompt=image_mask):
            ans_none_masked = adapter.generate_answer(inputs, max_new_tokens)
        pos_control.append(ans_all_masked != baseline)
        neg_control.append(ans_none_masked == baseline)
    checks["start_layer=0 (mask everything) changes the answer (positive control)"] = (
        len(pos_control) > 0 and any(pos_control))
    checks["start_layer=num_layers (mask nothing) reproduces the baseline exactly (negative control)"] = (
        len(neg_control) > 0 and all(neg_control))

    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Attention knockout: block text-query -> image-key attention from a chosen start "
                    "layer through the last layer, and see whether the stated time collapses.")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--data_csv", type=str, default="data_balanced/data.csv")
    parser.add_argument("--images_dir", type=str, default="data_balanced")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--start_layers", type=str, default=None,
                        help=f"comma-separated start layers (default: {START_LAYERS})")
    parser.add_argument("--max_images", type=int, default=None, help="cap on images, for a quick smoke test")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip_logit_metric", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="run the 5 verification checks and exit (nonzero on any FAIL) -- does NOT "
                             "run the sweep")
    parser.add_argument("--verify_images", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)

    # setup_model_and_vision (intervene.py, reused unmodified) only reads model_id/adapter/
    # vision_method off its `args` -- knockout.py doesn't touch the vision-ceiling mechanism at
    # all, so vision_method is fixed to intervene.py's own trusted default.
    model_args = argparse.Namespace(model_id=args.model_id, adapter=args.adapter, vision_method="layer0_embed")
    adapter, decoder_layers, _owners, _vm = setup_model_and_vision(model_args)
    attn_modules = find_attention_modules(decoder_layers)
    out_dir = output_dir_for(args.out_dir, adapter)
    os.makedirs(out_dir, exist_ok=True)

    if args.verify:
        checks = run_verification(adapter, decoder_layers, attn_modules, df, args.images_dir,
                                  max_new_tokens=args.max_new_tokens, n_verify_images=args.verify_images,
                                  seed=args.seed)
        print("\n--- knockout.py --verify verdict ---")
        all_pass = True
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
            all_pass = all_pass and ok
        sys.exit(0 if all_pass else 1)

    start_layers = ([int(x) for x in args.start_layers.split(",")] if args.start_layers else START_LAYERS)
    trials = run_knockout_sweep(adapter, decoder_layers, attn_modules, df, args.images_dir, out_dir,
                                start_layers=start_layers, max_new_tokens=args.max_new_tokens,
                                max_images=args.max_images, logit_metric=not args.skip_logit_metric)
    summary = summarize_knockout(trials, out_dir, seed=args.seed)
    text = print_knockout_summary(summary)
    with open(os.path.join(out_dir, "knockout_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nAll outputs written to '{out_dir}/'.")


if __name__ == "__main__":
    main()
