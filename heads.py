"""
heads.py -- HEAD-LEVEL analysis: which attention heads carry the stated
minute from image-token positions into the answer, and do 3B and 7B differ?

BACKGROUND (see README.md's "Causal intervention (Step 3)" and
intervene.py's module docstring): Experiment A established WHERE the
minute is causally read out (image-token positions, a mid-relative-depth
window, replicated on fresh pairs for both 3B and 7B with no detectable
cross-model difference in WHERE). This script asks WHICH components inside
that window do it -- patching one attention head's contribution at a time,
instead of the whole residual stream.

THE MECHANISM (new -- read this before trusting anything downstream): every
attention module in this project's supported model families (confirmed
directly from this environment's installed transformers source for
Qwen2.5-VL, and the same convention holds for Gemma3/Llama-family
attention via the shared ALL_ATTENTION_FUNCTIONS refactor -- see
find_attention_modules) computes each head's output, concatenates all
heads along the last dimension, THEN applies one linear layer `o_proj` to
mix them back into the residual stream:
    attn_output = attn_output.reshape(bsz, seq, num_heads * head_dim)
    attn_output = self.o_proj(attn_output)
Head h's contribution occupies columns [h*head_dim : (h+1)*head_dim] of
that concatenated tensor, in head order. So "patch one head's output" means:
register a forward-PRE-hook on `o_proj` (NOT a new mechanism -- the same
kind of hook `patched()`/`register_layer_patch` already use, just at a
different point in the forward pass) that replaces ONLY that column slice,
at the image-token positions, with the corresponding slice from a cached
source (clock B)'s own forward pass at the same layer. Everything else
(shape-mismatch guard for incremental decode steps, patch-source caching
via cheap baseline reuse, pair building/exclusion, the transfer-to-baseline
metric, pair-level permutation tests/bootstrap CIs) is REUSED from
intervene.py / analyze_replication.py, not reimplemented -- this file adds
only the head-granularity patch/capture mechanics and the
sweep/verify/reporting code that's genuinely new.

RUN --verify BEFORE TRUSTING ANY RESULT FROM THIS FILE. This is a NEW
patching mechanism with zero real-weight testing (this environment has no
GPU) -- the exact failure mode intervene.py's own history warns about
(get_image_features looked fine by every superficial check for two rounds
before --verify caught it doing nothing) applies here with equal force, and
there is no reason to expect a first attempt at head-level hooking is
immune to the same class of bug.

That risk was realized during development, against a fake model, before
any real weights were touched: every call site of head_patched() was
passing find_attention_modules()'s return value (the attention module
itself) straight through, instead of that module's `.o_proj` child --
patching the PRE-attention residual stream instead of the per-head
post-attention tensor. Against a fake model built to reproduce this
project's own round-3 Qwen bug shape (o_proj fires but its output is
discarded and recomputed independently from the unpatched value),
--verify's downstream-propagation check should have failed and instead
reported PASS -- because patching the wrong (upstream) tensor bypassed the
"discarded" path entirely rather than exercising it. Fixed by patching
`attn_module.o_proj` everywhere (matching what capture_head_inputs already
did correctly); re-verified against the same fake broken model, which now
correctly reports FAIL. Left here as the concrete answer to "why does this
file insist on --verify" for the next person (or model) who's tempted to
skip it.

ATTENTION-PATTERN INSPECTION CAVEAT: there is no verified pixel-to-patch-
token mapping in this codebase (see adapters.py's InternVLAdapter
docstring for the same caveat in a different context). This script reports
ONLY the fraction of a head's attention mass on image-token positions vs.
text positions -- never a claim about attending "to the minute hand" or any
other specific image region.
"""

import argparse
import contextlib
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adapters import get_adapter, output_dir_for, relative_depth
from eval_behavior import PROMPT, SEED, parse_time_answer
from intervene import (MAX_NEW_TOKENS, MIN_GAP_MINUTES, _baseline_fields, build_pairs,
                        compute_transfer_columns, find_same_minute_partner, forward_hidden_states,
                        load_excluded_pairs, make_matched_noise, relative_l2_diff, resolve_layers_cli,
                        run_baseline, setup_model_and_vision)
from analyze_replication import bootstrap_ci, bootstrap_diff_ci, permutation_test

OUT_DIR = "heads_output"
N_PAIRS = 20   # small default -- heads x layers x pairs explodes fast, see module docstring's cost section


# ---------------------------------------------------------------------------
# Head-level patch/capture mechanics -- the one genuinely new piece here
# ---------------------------------------------------------------------------

def find_attention_modules(decoder_layers):
    """For each decoder layer, find its attention submodule -- identified
    by REFLECTION (has a callable `o_proj` attribute), not a hardcoded
    attribute name like "self_attn" -- same philosophy as adapters.py's
    find_decoder_layers_generic, and confirmed against the REAL installed
    transformers source for Qwen2.5-VL (Qwen2_5_VLAttention.o_proj); the
    same `o_proj`-after-concatenated-heads convention is shared by every
    Llama-family attention module in transformers' ALL_ATTENTION_FUNCTIONS
    refactor (Gemma3, Qwen2, Llama, ...), so this works unmodified for
    Gemma3Adapter/InternVLAdapter's language backbones too -- but see the
    module docstring: run --verify per model before trusting that this
    generalization actually holds for a model that hasn't been checked.

    Returns a list, index i = decoder_layers[i]'s own attention module (so
    index i corresponds to hidden_states[i+1] in output_hidden_states'
    convention -- the layer whose OUTPUT that hidden state is). Raises if
    any decoder layer doesn't have EXACTLY one such submodule -- silently
    guessing wrong here would corrupt every downstream result.
    """
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


def head_geometry(attn_module):
    """(num_heads, head_dim) for one attention module, read directly off
    its OWN attributes (set at __init__ time, e.g.
    `Qwen2_5_VLAttention.num_heads`/`.head_dim`) rather than re-derived from
    a config object whose nesting differs per model family -- more robust,
    and cross-checked against `o_proj.in_features` so a mismatch (this
    module doesn't actually match the assumed concatenated-heads-then-
    o_proj convention) raises immediately instead of silently patching the
    wrong columns."""
    num_heads = attn_module.num_heads
    head_dim = attn_module.head_dim
    in_features = attn_module.o_proj.in_features
    if num_heads * head_dim != in_features:
        raise AttributeError(
            f"num_heads ({num_heads}) * head_dim ({head_dim}) = {num_heads * head_dim} != "
            f"o_proj.in_features ({in_features}) -- this attention module doesn't match the "
            "concatenated-heads-then-o_proj convention find_attention_modules assumes. Do not "
            "proceed: patching by head_dim-sized column slices would corrupt the wrong data."
        )
    return num_heads, head_dim


def _head_col_slice(head_idx, head_dim):
    """head_idx=None means ALL heads (the --verify extreme test); otherwise
    the column range for that one head's contribution to the concatenated
    pre-o_proj tensor."""
    if head_idx is None:
        return slice(None)
    return slice(head_idx * head_dim, (head_idx + 1) * head_dim)


def _extract_o_proj_input(args, kwargs):
    """o_proj is called positionally in every real attention module this
    project has checked (`self.o_proj(attn_output)`), but handle the
    keyword form too -- same defensive-but-cheap pattern as intervene.py's
    _extract_hidden_states for the analogous decoder-layer hook."""
    if len(args) > 0 and torch.is_tensor(args[0]):
        return args[0], "arg0"
    if "input" in kwargs and torch.is_tensor(kwargs["input"]):
        return kwargs["input"], "kwarg"
    raise RuntimeError("Could not locate o_proj's input tensor (checked args[0] and kwargs['input']).")


def _replace_o_proj_input(args, kwargs, where, new_inp):
    if where == "arg0":
        return (new_inp,) + tuple(args[1:]), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["input"] = new_inp
    return args, new_kwargs


@contextlib.contextmanager
def head_patched(o_proj_module, mask, values, head_idx, head_dim, mode="replace"):
    """Context manager: register a patch hook on `o_proj_module` for the
    duration of a `with` block, ALWAYS removed afterward (even if the
    generate() call inside raises) -- same contract as intervene.py's
    `patched`. `values`: (seq, num_heads*head_dim) CPU/GPU tensor (the
    SOURCE's cached pre-o_proj activations at this layer, batch dim already
    dropped -- see capture_head_inputs)."""
    def hook(module, args, kwargs):
        inp, where = _extract_o_proj_input(args, kwargs)
        new_inp = inp.clone()
        col = _head_col_slice(head_idx, head_dim)
        if inp.shape[1] == mask.shape[0]:   # prefill-shaped call only -- see head_patched's docstring;
                                             # a single-token incremental-decode call has no "image
                                             # token" structure, and the prefill pass already baked in
                                             # whatever effect this patch has via the KV cache
            v = values[:, col].to(dtype=inp.dtype, device=inp.device)
            if mode == "replace":
                new_inp[0, mask, col] = v[mask]
            else:
                new_inp[0, mask, col] = new_inp[0, mask, col] + v[mask]
        return _replace_o_proj_input(args, kwargs, where, new_inp)

    handle = o_proj_module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def capture_head_inputs(attn_modules_by_layer, capture_holder):
    """Captures o_proj's INPUT tensor (pre-projection, per-head-concatenated,
    (seq, num_heads*head_dim), batch dim dropped, CPU float32) for each of
    `attn_modules_by_layer` ({layer: attn_module} -- ATTENTION modules, as
    returned by find_attention_modules; the hook is registered on each
    module's `.o_proj` child, NOT on the attention module itself, which
    would instead see the attention module's own INPUT -- the pre-attention
    hidden states -- and silently capture the wrong tensor) into
    `capture_holder[layer]`, during whatever forward/generate call(s) happen
    inside this `with` block. Only captures multi-token (prefill-shaped)
    calls -- a single-token incremental-decode-step call would otherwise
    silently corrupt the cache with a 1-token slice. Designed to WRAP an
    existing call (e.g. `run_baseline(...)`, unmodified) rather than
    duplicate its logic: `run_baseline` itself does one plain forward pass
    (captured here) and then one generate() call (whose OWN prefill
    forward pass recomputes the identical activations under eval-mode
    greedy decoding -- redundant but numerically identical, so re-capturing
    it is harmless; only its LATER single-token decode steps are actually
    guarded against)."""
    handles = []

    def make_hook(layer):
        def hook(module, args, kwargs):
            inp, _where = _extract_o_proj_input(args, kwargs)
            if inp.shape[1] > 1:
                capture_holder[layer] = inp[0].detach().float().cpu().clone()
            return None   # capture only -- never modifies anything

        return hook

    for layer, attn_module in attn_modules_by_layer.items():
        handles.append(attn_module.o_proj.register_forward_pre_hook(make_hook(layer), with_kwargs=True))
    try:
        yield capture_holder
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# The sweep: head-level patching, reusing intervene.py's pair-building,
# baseline caching, controls, and transfer metric UNCHANGED (see module
# docstring -- everything below is orchestration, not new mechanics).
# ---------------------------------------------------------------------------

def run_experiment_heads(adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
                          df, images_dir, out_dir, layers, n_pairs=N_PAIRS, max_pairs=None,
                          min_gap=MIN_GAP_MINUTES, max_heads=None, max_new_tokens=MAX_NEW_TOKENS,
                          seed=SEED, exclude_pairs=None):
    """Same paired design as Experiment A (build_pairs, same-minute partner,
    matched-norm noise) -- but patches ONE attention head's o_proj-input
    slice at the image-token positions, instead of the whole residual
    stream, swept over `layers` x every head (or the first `max_heads` of
    them). Output schema matches Experiment A's (`_baseline_fields`, so
    `compute_transfer_columns` works unmodified) plus `head` alongside
    `layer`."""
    rng = np.random.RandomState(seed)
    pairs = build_pairs(df, n_pairs=n_pairs, min_gap=min_gap, seed=seed, exclude_pairs=exclude_pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if len(pairs) == 0:
        raise ValueError("No valid (A, B) pairs found -- check --min_gap against this dataset's spread of hours/minutes.")

    num_layers = len(decoder_layers)
    attn_by_layer = {L: attn_modules[L] for L in layers}

    num_heads, head_dim = head_geometry(attn_modules[layers[0]])
    for L in layers:
        nh, hdim = head_geometry(attn_modules[L])
        if (nh, hdim) != (num_heads, head_dim):
            raise ValueError(f"Layer {L} has head geometry ({nh},{hdim}), different from layer {layers[0]}'s "
                             f"({num_heads},{head_dim}) -- this sweep assumes uniform head geometry across "
                             "swept layers (true for every architecture this project supports; a mismatch "
                             "means something unexpected about this model, not a case to silently paper over).")
    heads_to_sweep = list(range(num_heads)) if max_heads is None else list(range(min(num_heads, max_heads)))

    already_used = {f for pair in pairs for f in (pair[0]["filename"], pair[1]["filename"])}
    n_total_trials = len(pairs) * len(layers) * len(heads_to_sweep) * 3   # real_b, same_minute, noise
    print(f"heads.py: {len(pairs)} pair(s) x {len(layers)} layer(s) x {len(heads_to_sweep)} head(s) "
          f"x 3 conditions = {n_total_trials} generate() calls total (num_heads={num_heads}, "
          f"head_dim={head_dim}).")

    rows = []
    t_start = time.perf_counter()
    n_timed = 0

    for pair_idx, (a, b) in enumerate(tqdm(pairs, desc="heads.py pairs")):
        a_path = os.path.join(images_dir, a["filename"])
        b_path = os.path.join(images_dir, b["filename"])

        head_cache_a, head_cache_b, head_cache_same = {}, {}, {}
        with capture_head_inputs(attn_by_layer, head_cache_a):
            base_a = run_baseline(adapter, image_features_owners, vision_method, a_path,
                                   max_new_tokens=max_new_tokens)
        with capture_head_inputs(attn_by_layer, head_cache_b):
            base_b = run_baseline(adapter, image_features_owners, vision_method, b_path,
                                   max_new_tokens=max_new_tokens)

        same_row = find_same_minute_partner(df, a, exclude=already_used, rng=rng)
        base_same = None
        if same_row is not None:
            same_path = os.path.join(images_dir, same_row["filename"])
            with capture_head_inputs(attn_by_layer, head_cache_same):
                base_same = run_baseline(adapter, image_features_owners, vision_method, same_path,
                                          max_new_tokens=max_new_tokens)

        if base_a["seq_len"] != base_b["seq_len"] or (base_same is not None and base_a["seq_len"] != base_same["seq_len"]):
            print(f"WARNING: sequence-length mismatch for pair ({a['filename']}, {b['filename']}) -- skipping.")
            continue

        image_mask = base_a["image_mask"]

        for layer in layers:
            o_proj = attn_modules[layer].o_proj
            a_cache = head_cache_a.get(layer)
            b_cache = head_cache_b.get(layer)
            same_cache = head_cache_same.get(layer) if base_same is not None else None
            if a_cache is None or b_cache is None:
                print(f"WARNING: layer {layer} was never captured for pair {pair_idx} (prefill length "
                      "mismatch or o_proj never called?) -- skipping this layer for this pair.")
                continue

            for head_idx in heads_to_sweep:
                col = slice(head_idx * head_dim, (head_idx + 1) * head_dim)

                with head_patched(o_proj, image_mask, b_cache, head_idx, head_dim, mode="replace"):
                    ans = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                pred_hour, pred_minute, ok = parse_time_answer(ans)
                rows.append({
                    "seed": seed, "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"],
                    "condition": "real_b", "layer": layer, "head": head_idx, "num_layers": num_layers,
                    "relative_depth": relative_depth(layer, num_layers),
                    "a_true_minute": int(a["minute"]), "b_true_minute": int(b["minute"]),
                    **_baseline_fields("a", base_a), **_baseline_fields("b", base_b),
                    "patched_hour": pred_hour if ok else None, "patched_minute": pred_minute if ok else None,
                    "patched_answer": ans, "answer_changed": (ans != base_a["raw_answer"]),
                })

                if base_same is not None and same_cache is not None:
                    with head_patched(o_proj, image_mask, same_cache, head_idx, head_dim, mode="replace"):
                        ans_s = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                    pred_hour_s, pred_minute_s, ok_s = parse_time_answer(ans_s)
                    rows.append({
                        "seed": seed, "pair": pair_idx, "a_file": a["filename"], "b_file": same_row["filename"],
                        "condition": "same_minute", "layer": layer, "head": head_idx, "num_layers": num_layers,
                        "relative_depth": relative_depth(layer, num_layers),
                        "a_true_minute": int(a["minute"]), "b_true_minute": int(same_row["minute"]),
                        **_baseline_fields("a", base_a), **_baseline_fields("b", base_same),
                        "patched_hour": pred_hour_s if ok_s else None,
                        "patched_minute": pred_minute_s if ok_s else None,
                        "patched_answer": ans_s, "answer_changed": (ans_s != base_a["raw_answer"]),
                    })

                # matched-norm noise, on JUST this head's slice of A's OWN cached
                # activations (same convention as Experiment A's full-layer noise
                # control -- see intervene.py's make_matched_noise) -- reuses B's
                # baseline as the "transfer target" for apples-to-apples comparison
                # with real_b, same as Experiment A's noise condition does.
                noise_full = a_cache.clone()
                noise_full[:, col] = make_matched_noise(a_cache[:, col], rng)
                with head_patched(o_proj, image_mask, noise_full, head_idx, head_dim, mode="replace"):
                    ans_n = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                pred_hour_n, pred_minute_n, ok_n = parse_time_answer(ans_n)
                rows.append({
                    "seed": seed, "pair": pair_idx, "a_file": a["filename"], "b_file": None,
                    "condition": "noise", "layer": layer, "head": head_idx, "num_layers": num_layers,
                    "relative_depth": relative_depth(layer, num_layers),
                    "a_true_minute": int(a["minute"]), "b_true_minute": int(b["minute"]),
                    **_baseline_fields("a", base_a), **_baseline_fields("b", base_b),
                    "patched_hour": pred_hour_n if ok_n else None,
                    "patched_minute": pred_minute_n if ok_n else None,
                    "patched_answer": ans_n, "answer_changed": (ans_n != base_a["raw_answer"]),
                })

                n_timed += 1
                if pair_idx == 0 and n_timed == 6:
                    elapsed = time.perf_counter() - t_start
                    per_trial = elapsed / n_timed
                    remaining = n_total_trials - n_timed
                    print(f"\n[time estimate] ~{per_trial:.2f}s/trial measured from the first pair -> "
                          f"~{per_trial * n_total_trials / 60:.1f} min total "
                          f"(~{per_trial * remaining / 60:.1f} min remaining)\n")

    trials_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    trials_df.to_csv(os.path.join(out_dir, "heads_trials.csv"), index=False)
    print(f"\nheads.py: {len(trials_df)} trials over {len(pairs)} pairs written to '{out_dir}/heads_trials.csv'.")
    return trials_df


# ---------------------------------------------------------------------------
# Summary: layer x head heatmap, top-heads table, cross-model comparison --
# pair-level statistics via analyze_replication.py, REUSED not reimplemented.
# ---------------------------------------------------------------------------

def summarize_heads(trials_df, out_dir, n_permutations=2000):
    """Per (layer, head): to_B rate for real_b/same_minute/noise, restricted
    to pairs where A's and the target's stated minutes differ (reusing
    intervene.py's compute_transfer_columns -- the SAME restriction
    Experiment A and analyze_replication.py use), plus a pair-level
    permutation test of real_b vs. noise (analyze_replication.permutation_test,
    reused not reimplemented) with a Bonferroni correction across however
    many (layer, head) cells were actually tested -- sweeping this many
    cells at once and reading each at the raw alpha=0.05 level would produce
    false positives by construction."""
    df = compute_transfer_columns(trials_df)
    restricted = df[df["minute_baselines_differ"] == 1.0]

    cells = restricted[["layer", "head"]].drop_duplicates()
    n_cells = len(cells)
    rows = []
    for _, cell in cells.iterrows():
        layer, head = int(cell["layer"]), int(cell["head"])
        sub = restricted[(restricted["layer"] == layer) & (restricted["head"] == head)]
        real_b = sub[sub["condition"] == "real_b"].set_index("pair")["minute_transfer"].dropna()
        same_minute = sub[sub["condition"] == "same_minute"].set_index("pair")["minute_transfer"].dropna()
        noise = sub[sub["condition"] == "noise"].set_index("pair")["minute_transfer"].dropna()

        p_value = float("nan")
        if len(real_b) >= 2 and len(noise) >= 2:
            _, p_value = permutation_test(real_b.to_numpy(), noise.to_numpy(),
                                           n_permutations=n_permutations, seed=0)
        p_bonf = min(1.0, p_value * n_cells) if not np.isnan(p_value) else float("nan")

        rows.append({
            "layer": layer, "head": head, "relative_depth": float(sub["relative_depth"].iloc[0]),
            "n_pairs_usable": int(len(real_b)),
            "to_b_real_b": float(real_b.mean()) if len(real_b) else float("nan"),
            "to_b_same_minute": float(same_minute.mean()) if len(same_minute) else float("nan"),
            "to_b_noise": float(noise.mean()) if len(noise) else float("nan"),
            "diff_vs_noise": (float(real_b.mean()) - float(noise.mean())) if len(real_b) and len(noise) else float("nan"),
            "p_value": p_value, "p_value_bonferroni": p_bonf,
        })

    summary = pd.DataFrame(rows).sort_values("diff_vs_noise", ascending=False).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "heads_summary.csv"), index=False)
    return summary


def plot_heads_heatmap(summary_df, out_dir, title_suffix=""):
    """Layer x head heatmap of to_B(real_b) - to_B(noise) -- the primary
    output requested: which heads carry the minute, at a glance."""
    layers = sorted(summary_df["layer"].unique())
    heads = sorted(summary_df["head"].unique())
    grid = np.full((len(layers), len(heads)), np.nan)
    layer_pos = {l: i for i, l in enumerate(layers)}
    head_pos = {h: i for i, h in enumerate(heads)}
    for _, r in summary_df.iterrows():
        grid[layer_pos[r["layer"]], head_pos[r["head"]]] = r["diff_vs_noise"]

    fig, ax = plt.subplots(figsize=(max(6, len(heads) * 0.5), max(4, len(layers) * 0.4)))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels(heads)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"to_B(real_b) - to_B(noise){title_suffix}")
    fig.colorbar(im, ax=ax, label="diff vs. noise")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heads_heatmap.png"), dpi=120)
    plt.close(fig)


def print_heads_summary(summary_df, top_n=10):
    lines = ["=== HEAD-LEVEL SUMMARY ===", ""]
    n_cells = len(summary_df)
    n_sig = int((summary_df["p_value_bonferroni"] < 0.05).sum())
    lines.append(f"{n_cells} (layer, head) cell(s) tested; {n_sig} significant after Bonferroni "
                 "correction (alpha=0.05) -- read the RAW p_value column with that correction in mind, "
                 "not at face value, given how many cells were tested at once.")
    lines.append("")
    top = summary_df.sort_values("diff_vs_noise", ascending=False).head(top_n)
    header = (f"{'layer':>6}{'head':>6}{'rel_depth':>11}{'to_B(real_b)':>14}{'to_B(same_min)':>16}"
              f"{'to_B(noise)':>13}{'diff':>9}{'n':>5}{'p':>9}{'p_bonf':>9}{'sig':>5}")
    lines.append(header)
    for _, r in top.iterrows():
        sig = "*" if (not np.isnan(r["p_value_bonferroni"]) and r["p_value_bonferroni"] < 0.05) else ""
        lines.append(
            f"{int(r['layer']):>6}{int(r['head']):>6}{r['relative_depth']:>11.2f}"
            f"{r['to_b_real_b']:>14.1%}{r['to_b_same_minute']:>16.1%}{r['to_b_noise']:>13.1%}"
            f"{r['diff_vs_noise']:>+9.1%}{int(r['n_pairs_usable']):>5}{r['p_value']:>9.4f}"
            f"{r['p_value_bonferroni']:>9.4f}{sig:>5}"
        )
    lines.append("")
    positive = summary_df[summary_df["diff_vs_noise"] > 0]["diff_vs_noise"].sort_values(ascending=False)
    total = positive.sum()
    lines.append("Concentration -- fraction of the total positive (diff_vs_noise) effect mass carried by "
                 "the top-K heads (one head dominating vs. spread across many looks very different here):")
    for k in (1, 3, 5, 10):
        frac = positive.head(k).sum() / total if total > 0 else float("nan")
        lines.append(f"  top {k:>2} head(s): {frac:.1%} of total positive effect (n={min(k, len(positive))} "
                     f"of {len(positive)} heads with a positive effect)")
    text = "\n".join(lines)
    print("\n" + text)
    return text


def compare_heads_across_models(name_a, summary_a, name_b, summary_b, sig_threshold=0.05):
    """The key cross-model report requested: how many heads carry the
    minute in each model, how concentrated the effect is, and at what
    relative depth the top heads sit -- side by side."""
    lines = ["=== CROSS-MODEL HEAD COMPARISON ===", ""]
    for name, summary in ((name_a, summary_a), (name_b, summary_b)):
        n_sig = int((summary["p_value_bonferroni"] < sig_threshold).sum())
        positive = summary[summary["diff_vs_noise"] > 0]["diff_vs_noise"].sort_values(ascending=False)
        total = positive.sum()
        top1_frac = (positive.head(1).sum() / total) if total > 0 else float("nan")
        top3_frac = (positive.head(3).sum() / total) if total > 0 else float("nan")
        top_row = summary.sort_values("diff_vs_noise", ascending=False).iloc[0] if len(summary) else None
        lines.append(f"{name}:")
        lines.append(f"  {n_sig}/{len(summary)} (layer, head) cells significant (Bonferroni, "
                     f"alpha={sig_threshold})")
        lines.append(f"  concentration: top 1 head = {top1_frac:.1%} of positive effect, "
                     f"top 3 heads = {top3_frac:.1%}")
        if top_row is not None:
            lines.append(f"  top head: layer {int(top_row['layer'])} head {int(top_row['head'])} "
                         f"(relative_depth={top_row['relative_depth']:.2f}), "
                         f"diff_vs_noise={top_row['diff_vs_noise']:+.1%}")
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    return text


# ---------------------------------------------------------------------------
# --verify: prove head-level patching actually lands, before trusting a null
# ---------------------------------------------------------------------------

def verify_head_patch(adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
                       pairs, images_dir, layers, max_heads_to_verify, max_new_tokens):
    """For a few (A, B) pairs and a few (layer, head) combos: confirm (a)
    the patched head's o_proj-INPUT slice at the image-token positions
    actually changes to B's cached value, and (b) that change propagates to
    a downstream layer AND the final layer's hidden states -- same spirit
    as intervene.py's verify_decoder_patch, at head instead of full-layer
    granularity."""
    records = []
    lines = [f"--- Head-level patch verification (vision method: {vision_method}) ---"]
    attn_by_layer = {L: attn_modules[L] for L in layers}
    num_heads, head_dim = head_geometry(attn_modules[layers[0]])
    heads_to_check = list(range(min(num_heads, max_heads_to_verify)))
    lines.append(f"Checking {len(pairs)} pair(s) x layers {layers} x heads {heads_to_check} "
                 f"(of {num_heads} total heads/layer).")

    for pair_idx, (a, b) in enumerate(pairs):
        a_path = os.path.join(images_dir, a["filename"])
        b_path = os.path.join(images_dir, b["filename"])
        head_cache_a, head_cache_b = {}, {}
        with capture_head_inputs(attn_by_layer, head_cache_a):
            base_a = run_baseline(adapter, image_features_owners, vision_method, a_path,
                                   max_new_tokens=max_new_tokens)
        with capture_head_inputs(attn_by_layer, head_cache_b):
            base_b = run_baseline(adapter, image_features_owners, vision_method, b_path,
                                   max_new_tokens=max_new_tokens)
        image_mask = base_a["image_mask"]
        hs_unpatched = base_a["hidden_states"]
        final_idx = len(hs_unpatched) - 1

        for layer in layers:
            o_proj = attn_modules[layer].o_proj
            for head_idx in heads_to_check:
                col = slice(head_idx * head_dim, (head_idx + 1) * head_dim)

                spy = {}

                def _spy(module, args, kwargs):
                    inp, _where = _extract_o_proj_input(args, kwargs)
                    # .float().cpu() here, matching capture_head_inputs's own capture hook --
                    # under device_map="auto" this layer's o_proj can live on a different cuda
                    # device than image_mask (always CPU, see adapters.py) or head_cache_a/b
                    # (always CPU, see capture_head_inputs), so comparing this raw would crash
                    # with a cross-device error the moment relative_l2_diff subtracts them.
                    spy["inp"] = inp[0].detach().float().cpu().clone()

                with head_patched(o_proj, image_mask, head_cache_b[layer], head_idx, head_dim, mode="replace"):
                    handle = o_proj.register_forward_pre_hook(_spy, with_kwargs=True)
                    try:
                        forward_hidden_states(adapter.model, base_a["inputs"])
                    finally:
                        handle.remove()
                actual_input = spy["inp"]
                diff_at_head = relative_l2_diff(head_cache_a[layer][image_mask][:, col],
                                                 actual_input[image_mask][:, col])

                with head_patched(o_proj, image_mask, head_cache_b[layer], head_idx, head_dim, mode="replace"):
                    hs_patched = forward_hidden_states(adapter.model, base_a["inputs"])
                downstream_idx = min(layer + 1, final_idx)
                diff_downstream = relative_l2_diff(hs_unpatched[downstream_idx][image_mask],
                                                    hs_patched[downstream_idx][image_mask])
                diff_final = relative_l2_diff(hs_unpatched[final_idx][image_mask], hs_patched[final_idx][image_mask])

                records.append({
                    "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"],
                    "layer": layer, "head": head_idx, "diff_at_head": diff_at_head,
                    "downstream_layer": downstream_idx, "diff_downstream": diff_downstream,
                    "diff_final": diff_final,
                })
                lines.append(
                    f"pair {pair_idx} layer {layer} head {head_idx}: diff-at-head={diff_at_head:.4f}, "
                    f"diff-at-layer-{downstream_idx}={diff_downstream:.4f}, diff-at-final-layer={diff_final:.4f}"
                )

    extreme_records = []
    if pairs:
        a0, b0 = pairs[0]
        a_path = os.path.join(images_dir, a0["filename"])
        head_cache_a0 = {}
        layer0 = layers[0]
        with capture_head_inputs({layer0: attn_modules[layer0]}, head_cache_a0):
            base_a0 = run_baseline(adapter, image_features_owners, vision_method, a_path,
                                    max_new_tokens=max_new_tokens)
        zero_vals = torch.zeros_like(head_cache_a0[layer0])
        with head_patched(attn_modules[layer0].o_proj, base_a0["image_mask"], zero_vals, None, head_dim, mode="replace"):
            ans_zero = adapter.generate_answer(base_a0["inputs"], max_new_tokens)
        changed = (ans_zero != base_a0["raw_answer"])
        extreme_records.append({"layer": layer0, "baseline_answer": base_a0["raw_answer"],
                                "extreme_answer": ans_zero, "extreme_answer_changed": changed})
        lines.append(f"\nEXTREME (ALL heads zeroed at layer {layer0}, image-token positions): "
                     f"baseline={base_a0['raw_answer']!r} -> {ans_zero!r} (changed={changed})")

    return lines, records, extreme_records


def build_head_verification_verdict(records, extreme_records):
    lines = ["--- Verdict ---"]
    problems = []
    df = pd.DataFrame(records)
    if len(df) == 0:
        problems.append("No head-patch verification data was collected at all.")
    else:
        if (df["diff_at_head"] < 1e-6).all():
            problems.append("Every tested head patch produced a BIT-IDENTICAL o_proj input at the "
                            "target head's own columns -- the patch hook is not writing anything.")
        if (df["diff_final"] < 1e-6).all():
            problems.append("Every tested head patch produced a BIT-IDENTICAL final-layer hidden state -- "
                            "patches are not propagating to the output at all, regardless of layer or head.")
    if not extreme_records or not any(r.get("extreme_answer_changed") for r in extreme_records):
        problems.append("The EXTREME test (ALL heads zeroed at one layer, image-token positions) did NOT "
                        "change the answer. If even total removal of every head's image-token contribution "
                        "at a layer doesn't move the answer, the hook is not wired into generation -- any "
                        "null result from the real sweep is unverified until this is fixed.")

    if problems:
        lines.append("FAIL -- do not trust any head-level sweep result yet:")
        for p in problems:
            lines.append(f"  - {p}")
    else:
        lines.append("PASS: head-level patches write genuinely different o_proj inputs at the target head's")
        lines.append("own columns, that propagate to the final layer, and the extreme all-heads-zeroed test")
        lines.append("DOES change the answer -- the mechanism is demonstrably wired into the forward path.")
    return lines


def run_verification(adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
                      df, images_dir, out_dir, n_verify_pairs=2, verify_layers=None, max_heads_to_verify=2,
                      min_gap=MIN_GAP_MINUTES, max_new_tokens=MAX_NEW_TOKENS, seed=SEED):
    if verify_layers is None:
        num_layers = len(decoder_layers)
        verify_layers = sorted(set([num_layers // 3, 2 * num_layers // 3]))
    rng = np.random.RandomState(seed)
    pairs = build_pairs(df, n_pairs=n_verify_pairs, min_gap=min_gap, seed=seed)
    if len(pairs) == 0:
        raise ValueError("No (A, B) pairs available for --verify -- check --min_gap against this dataset.")

    print(f"\n{'=' * 70}\nHEADS.PY VERIFICATION: confirming head-level patching actually lands, "
          f"before trusting any null result\n{'=' * 70}")
    print(f"model: {adapter.short_name} ({type(adapter).__name__})")
    print(f"Using {len(pairs)} pair(s), layers {verify_layers}, up to {max_heads_to_verify} head(s)/layer.")

    lines, records, extreme_records = verify_head_patch(
        adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
        pairs, images_dir, verify_layers, max_heads_to_verify, max_new_tokens)
    verdict_lines = build_head_verification_verdict(records, extreme_records)

    all_lines = ["=== HEADS.PY VERIFICATION REPORT ===", ""] + lines + [""] + verdict_lines
    text = "\n".join(all_lines)
    print("\n" + text)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "heads_verification_report.txt"), "w") as f:
        f.write(text + "\n")
    pd.DataFrame(records).to_csv(os.path.join(out_dir, "heads_verification_records.csv"), index=False)
    pd.DataFrame(extreme_records).to_csv(os.path.join(out_dir, "heads_verification_extreme.csv"), index=False)
    print(f"\nVerification report saved to '{out_dir}/heads_verification_report.txt'.")
    return text, records, extreme_records


# ---------------------------------------------------------------------------
# Attention pattern inspection -- BEST EFFORT (see module docstring on the
# unverified pixel-to-patch-token mapping caveat: image-vs-text fraction
# ONLY, never a spatial claim).
# ---------------------------------------------------------------------------

def inspect_attention_pattern(adapter, attn_module, layer, head_idx, image_path, max_new_tokens=MAX_NEW_TOKENS):
    """Forces eager attention (sdpa/flash-attention do not return attention
    weights at all -- see module docstring) for the duration of ONE forward
    pass, then reports what fraction of the FINAL prompt token's ("the
    answer position", the one that starts generating the reply) attention,
    for `head_idx` at `layer`, lands on image-token positions vs. text
    positions. Returns None (not a fabricated number) if attention weights
    still aren't available after forcing eager -- some model/implementation
    combinations may not support this at all, and reporting nothing is
    better than reporting a wrong number silently."""
    attn_config = getattr(attn_module, "config", None)
    orig_impl = getattr(attn_config, "_attn_implementation", None) if attn_config is not None else None
    if orig_impl is not None:
        attn_module.config._attn_implementation = "eager"
    try:
        inputs = adapter.build_inputs(image_path, PROMPT)
        with torch.no_grad():
            outputs = adapter.model(**inputs, output_hidden_states=False, output_attentions=True)
    finally:
        if orig_impl is not None:
            attn_module.config._attn_implementation = orig_impl

    attentions = getattr(outputs, "attentions", None)
    if attentions is None or layer >= len(attentions) or attentions[layer] is None:
        return None
    weights = attentions[layer][0, head_idx, -1, :].detach().float().cpu()
    image_mask = adapter.image_token_positions(inputs)
    return {
        "frac_image": float(weights[image_mask].sum()),
        "frac_text": float(weights[~image_mask].sum()),
    }


def inspect_top_heads(adapter, attn_modules, summary_df, df, images_dir, top_n=5, n_images=5, seed=SEED):
    """Runs inspect_attention_pattern for the top `top_n` heads (by
    diff_vs_noise) over `n_images` sample images, averaging the image/text
    attention fraction. Returns a DataFrame, one row per (layer, head) with
    the averaged fractions -- or an empty DataFrame with a printed note if
    attention weights aren't available at all for this model (see
    inspect_attention_pattern)."""
    rng = np.random.RandomState(seed)
    sample = df.sample(n=min(n_images, len(df)), random_state=seed)
    top = summary_df.sort_values("diff_vs_noise", ascending=False).head(top_n)

    rows = []
    n_unavailable = 0
    for _, r in top.iterrows():
        layer, head = int(r["layer"]), int(r["head"])
        fracs_image, fracs_text = [], []
        for _, row in sample.iterrows():
            path = os.path.join(images_dir, row["filename"])
            result = inspect_attention_pattern(adapter, attn_modules[layer], layer, head, path)
            if result is None:
                n_unavailable += 1
                continue
            fracs_image.append(result["frac_image"])
            fracs_text.append(result["frac_text"])
        rows.append({
            "layer": layer, "head": head, "diff_vs_noise": r["diff_vs_noise"],
            "n_images_usable": len(fracs_image),
            "mean_frac_image": float(np.mean(fracs_image)) if fracs_image else float("nan"),
            "mean_frac_text": float(np.mean(fracs_text)) if fracs_text else float("nan"),
        })
    if n_unavailable > 0:
        print(f"NOTE: attention weights were unavailable for {n_unavailable} (head, image) combination(s) "
              "-- see inspect_attention_pattern's docstring (some model/attn-implementation combinations "
              "may not support output_attentions=True even after forcing eager). Reported fractions above "
              "are averaged only over the combinations where they WERE available.")
    return pd.DataFrame(rows)


def print_attention_inspection(inspect_df):
    lines = ["=== ATTENTION PATTERN INSPECTION (image-vs-text mass only -- see module docstring's caveat", "on why this is NOT a claim about attending to the minute hand specifically) ===", ""]
    if len(inspect_df) == 0 or inspect_df["n_images_usable"].sum() == 0:
        lines.append("No attention weights were available for any inspected head -- this model/attention "
                     "implementation combination may not support output_attentions=True even after forcing "
                     "eager (see inspect_attention_pattern). Nothing to report.")
    else:
        header = f"{'layer':>6}{'head':>6}{'diff_vs_noise':>15}{'frac_image':>12}{'frac_text':>11}{'n_images':>10}"
        lines.append(header)
        for _, r in inspect_df.iterrows():
            lines.append(f"{int(r['layer']):>6}{int(r['head']):>6}{r['diff_vs_noise']:>+15.1%}"
                         f"{r['mean_frac_image']:>12.1%}{r['mean_frac_text']:>11.1%}{int(r['n_images_usable']):>10}")
    text = "\n".join(lines)
    print("\n" + text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Head-level analysis: which attention heads carry the stated minute from image "
                    "tokens into the answer, patching one head's o_proj-input slice at a time.")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Hugging Face model id. Default: the resolved adapter's own default "
                             "(Qwen2.5-VL-3B if neither --model_id nor --adapter is given).")
    parser.add_argument("--adapter", type=str, default=None,
                        help="which model-family adapter to use (see adapters.py) -- inferred from "
                             "--model_id if omitted.")
    parser.add_argument("--data_csv", type=str, default="data_balanced/data.csv")
    parser.add_argument("--images_dir", type=str, default="data_balanced")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                        help="results are written to <out_dir>/<model_short_name>/ (see adapters.output_dir_for)")
    parser.add_argument("--layers", type=str, required=True,
                        help="REQUIRED (no safe default -- heads x layers x pairs explodes fast, see module "
                             "docstring's cost section): comma-separated absolute layers, or 'rel:f1,f2,...' "
                             "for relative depth (resolved once the model's layer count is known). Suggested: "
                             "the readout window found by Experiment A for THIS model, e.g. "
                             "'rel:0.55,0.60,...,0.80' or the model-specific absolute layers from README.md's "
                             "\"Causal intervention\" section (3B: ~20-28, 7B: ~15-22).")
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS,
                        help=f"number of (A, B) pairs (default {N_PAIRS} -- kept small, see cost section)")
    parser.add_argument("--max_pairs", type=int, default=None, help="cap on pairs, for a quick smoke test")
    parser.add_argument("--max_heads", type=int, default=None,
                        help="cap on heads swept PER LAYER (first --max_heads of them), for a quick smoke test")
    parser.add_argument("--min_gap", type=int, default=MIN_GAP_MINUTES)
    parser.add_argument("--exclude_pairs_from", type=str, default=None,
                        help="path to a prior run's *_trials.csv -- see intervene.py's --exclude_pairs_from "
                             "(same function, reused directly)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--vision_method", type=str, default="layer0_embed", choices=["layer0_embed"],
                        help="head-level patching doesn't touch the vision-ceiling mechanism at all -- this "
                             "is here only because run_baseline() takes it; layer0_embed is intervene.py's "
                             "own trusted default and the only one exercised by this script.")
    parser.add_argument("--verify", action="store_true",
                        help="before the sweep, verify head-level patching actually lands (see module "
                             "docstring) -- ALWAYS run this at least once for a new model before trusting "
                             "any sweep result from it.")
    parser.add_argument("--verify_pairs", type=int, default=2)
    parser.add_argument("--verify_layers", type=str, default=None,
                        help="comma-separated or 'rel:...' layers to verify (default: two layers spanning "
                             "--layers' range)")
    parser.add_argument("--max_heads_verify", type=int, default=2,
                        help="how many heads per verify layer to check (kept small -- this is a sanity "
                             "check, not the full sweep)")
    parser.add_argument("--no_sweep", action="store_true",
                        help="skip the full sweep -- combine with --verify to just check the mechanism")
    parser.add_argument("--inspect_attention", action="store_true",
                        help="after the sweep, run BEST-EFFORT attention-pattern inspection on the top "
                             "heads (image-vs-text attention mass only -- see module docstring's caveat)")
    parser.add_argument("--top_n_inspect", type=int, default=5)
    parser.add_argument("--n_images_inspect", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    adapter, decoder_layers, image_features_owners, vision_method = setup_model_and_vision(args)
    num_layers = len(decoder_layers)
    attn_modules = find_attention_modules(decoder_layers)
    out_dir = output_dir_for(args.out_dir, adapter)
    os.makedirs(out_dir, exist_ok=True)

    layers = resolve_layers_cli(args.layers, num_layers)
    verify_layers = resolve_layers_cli(args.verify_layers, num_layers) if args.verify_layers else None

    exclude_pairs = load_excluded_pairs(args.exclude_pairs_from) if args.exclude_pairs_from else None

    if args.verify:
        run_verification(adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
                         df, args.images_dir, out_dir, n_verify_pairs=args.verify_pairs,
                         verify_layers=verify_layers, max_heads_to_verify=args.max_heads_verify,
                         min_gap=args.min_gap, max_new_tokens=args.max_new_tokens, seed=args.seed)

    if args.no_sweep:
        print("\n--no_sweep: skipping the full sweep.")
        return

    trials = run_experiment_heads(
        adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
        df, args.images_dir, out_dir, layers, n_pairs=args.n_pairs, max_pairs=args.max_pairs,
        min_gap=args.min_gap, max_heads=args.max_heads, max_new_tokens=args.max_new_tokens,
        seed=args.seed, exclude_pairs=exclude_pairs)

    summary = summarize_heads(trials, out_dir)
    plot_heads_heatmap(summary, out_dir, title_suffix=f" ({adapter.short_name})")
    text = print_heads_summary(summary)
    with open(os.path.join(out_dir, "heads_summary.txt"), "w") as f:
        f.write(text + "\n")

    if args.inspect_attention:
        inspect_df = inspect_top_heads(adapter, attn_modules, summary, df, args.images_dir,
                                       top_n=args.top_n_inspect, n_images=args.n_images_inspect, seed=args.seed)
        inspect_df.to_csv(os.path.join(out_dir, "heads_attention_inspection.csv"), index=False)
        text_inspect = print_attention_inspection(inspect_df)
        with open(os.path.join(out_dir, "heads_attention_inspection.txt"), "w") as f:
            f.write(text_inspect + "\n")

    print(f"\nAll outputs written to '{out_dir}/'.")


if __name__ == "__main__":
    main()
