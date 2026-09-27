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

UNDERPOWERED-CELL CAVEAT (found on a real Kaggle run, not hypothetical):
every summary cell is restricted to pairs where A's and the target's
STATED minutes differ (compute_transfer_columns' minute_baselines_differ),
and a `--max_pairs 10` run once had only 2 of those 10 pairs actually
qualify -- every cell printed a clean-looking 0.0%/p=1.0000, which reads
like "no effect" but was really "next to no data". `estimate_usable_pairs`
now predicts and warns about this BEFORE the sweep runs, `summarize_heads`
flags any cell below MIN_N_FOR_RELIABLE_CELL as `underpowered` (excluded
from ranking/concentration/cross-model comparison/attention-inspection
selection, though its raw numbers stay in the CSV), and `power_note_text`
states the actual pairs-per-cell a run's (layer, head) count needs at
Bonferroni-corrected significance -- read it before sizing --n_pairs, not
after a wasted sweep.

THE SAME RUN ALSO SURFACED A DEEPER AMBIGUITY: every INDIVIDUAL head gave
EXACTLY 0.0% for real_b, same_minute, AND noise alike -- a perfect zero
across all three conditions that --verify (checks tensors propagate to
hidden states, not that a patch can move the ANSWER) cannot resolve
between "needs many heads together" (real finding) and "the head-patching
path is broken" (a bug). `run_all_heads_control` (--all_heads, and run
automatically at the top of every sweep) patches ALL heads at once as the
strongest test this mechanism can run, distinguishing the two cases by
whether that maximal patch can move the decoded answer at all. And since
the power note above makes a full per-head grid unaffordable on a Kaggle
T4, `make_head_selectors` (--head_groups / --head_range) supports patching
CONTIGUOUS GROUPS of heads at once instead of one at a time -- far fewer
cells (gentler Bonferroni correction, larger per-test effect) for a coarse
first pass, with --head_range for a narrower, individually-powered
follow-up on whichever group showed an effect.

A REAL KAGGLE RUN OF run_all_heads_control THEN CAME BACK 0% AT LAYER 22 ON
3B, vs. intervene.py's ~87% for a full residual-stream patch at the same
position -- prompting a deeper look that found ONE real, confirmed bug and
motivated two independent strengthenings, none of which are hypothetical:

  1. A REAL OFF-BY-ONE (find_attention_modules/attn_module_for_layer): this
     file's "--layers L" was silently patching what intervene.py calls
     layer L+1 -- attn_modules[i] is decoder_layers[i]'s OWN attention,
     whose output is hidden_states[i+1], but every call site indexed
     `attn_modules[layer]` directly using intervene.py's hidden_states-index
     convention (0=embeddings, 1..num_layers=decoder block outputs).
     Confirmed empirically (not just by re-reading docstrings): patching
     "layer 2" via intervene.py's patched() and via this file's (old)
     attn_modules[2] on an IDENTICAL fake model changed DIFFERENT
     hidden_states indices (2 vs. 3). Fixed via attn_module_for_layer,
     which also now raises a clear error for layer=0 (embeddings -- this
     file has no attention module to patch there) instead of silently
     wrapping to the last layer. This alone probably does not explain a
     full 87%->0% collapse (layer 22 shifted to 23 is still well inside
     the 20-28 readout window) but is a real, independently-worth-fixing
     bug regardless.

  2. --verify's OWN check was a near-tautology: it compared head_patched's
     pre-hook's return value to ITSELF (via a second pre-hook chained
     after it), which only proves the hook fired, not that its effect
     survived into o_proj's real computation -- a pre-hook that mutates a
     detached copy without returning it, or gets silently overridden,
     would look identical to a working one. verify_head_patch now ALSO
     checks, via _capture_o_proj_forward (a genuine forward_HOOK, a
     different mechanism entirely, registered independently of
     head_patched): (a) does o_proj's ACTUALLY-consumed input (guaranteed
     real by PyTorch's own hook semantics, confirmed directly against
     accelerate's source for device_map="auto" -- its AlignDevicesHook
     monkey-patches .forward but only moves tensors' device/dtype in
     pre_forward, never alters values) match what we intended to write,
     and (b) does o_proj's ACTUAL output match a from-scratch
     recomputation using its own weight/bias applied to that intended
     input. Both were near-zero (correct) for the working AND the broken
     fake-model verification cases alike -- for the broken case, this
     correctly shows o_proj computes right but gets discarded downstream
     (diff_final still catches that), not that the hook itself is inert.

  3. run_all_heads_control now asserts, HARD (raises, does not just print),
     that the ANSWER STRING changes from baseline on most (>=50%) real_b
     trials when ALL heads are patched -- a more basic, harder-to-fake
     signal than to_B, since to_B could in principle read 0% even from a
     genuinely working mechanism (the answer moves, just not to exactly
     B's minute). A mechanism where the answer never moves at all fails
     this immediately, before any to_B number is even computed or could
     be misread as a measurement.

None of this rules out that patching ONE layer's attention contribution
alone (even all heads) is just a smaller, different intervention than
intervene.py's whole-residual-stream patch (which also overwrites the
incoming residual and this layer's MLP contribution) -- a real, legitimate
"needs more than attention-at-one-layer" finding remains possible and would
show up as: --verify passes, the new independent checks above are all
near-zero, AND the answer-changed assertion passes, but to_B still comes
back near-0%. That combination is real signal, not evidence of a bug.
"""

import argparse
import contextlib
import math
import os
import time
from statistics import NormalDist

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
                        load_excluded_pairs, make_matched_noise, recompute_baselines_for_files,
                        relative_l2_diff, resolve_layers_cli, run_baseline, setup_model_and_vision)
from analyze_replication import bootstrap_ci, bootstrap_diff_ci, permutation_test

OUT_DIR = "heads_output"
N_PAIRS = 20   # small default -- heads x layers x pairs explodes fast, see module docstring's cost section

# Statistical-power constants (see estimate_usable_pairs / power_note_text below): a coarse
# sweep with too few USABLE pairs (A's and B's STATED minutes differ -- see
# compute_transfer_columns) reads as a false "null" (0% everywhere, p=1.0) rather than what it
# actually is -- an underpowered/empty measurement. This bit a real run: --max_pairs 10 produced
# only 2 usable pairs, and every cell printed 0%/p=1.0 as if it were a real result.
MIN_USABLE_PAIRS_WARN = 30      # req #1: warn loudly before the sweep if fewer than this many pairs are predicted usable
MIN_N_FOR_RELIABLE_CELL = 20    # req #2: below this per-cell n, mark UNDERPOWERED instead of printing raw rates/p-values
POWER_DETECT_DELTA = 0.20       # req #3: the effect size ("a 20-point difference") the power note sizes for
POWER_ALPHA = 0.05              # two-sided, before Bonferroni correction across cells
POWER_TARGET = 0.80             # standard 80% power convention

# POSITIVE CONTROL (see run_all_heads_control): a handful of pairs is enough to tell "the
# head-patching path is broken" (~0% even with ALL heads patched) from "it works, the effect just
# needs many heads together" (a rate roughly in the ballpark of intervene.py's own whole-layer
# result) -- this doesn't need anywhere near MIN_USABLE_PAIRS_WARN pairs, since it's a sanity check,
# not a statistically powered claim.
ALL_HEADS_CONTROL_PAIRS = 8
# A more basic, harder-to-fake gate than to_B: with ALL heads patched, the ANSWER STRING should
# differ from A's own unpatched baseline on MOST real_b trials, regardless of whether it lands on
# B's specific minute. to_B could in principle read 0% even from a genuinely working mechanism (the
# answer moves, just not to exactly B's minute); an answer that doesn't even change from baseline on
# most trials is a much stronger, more basic signal that the patch isn't reaching generation at all
# -- see run_all_heads_control's hard assertion below.
MIN_ANSWER_CHANGE_FRAC = 0.5
# intervene.py's OWN established finding (Experiment A, full residual-stream patch at image-token
# positions, NOT this file's o_proj-only patch -- see run_all_heads_control's docstring for why
# these are different-sized interventions) -- printed as context for interpreting the control's
# number, never as a hardcoded pass/fail threshold (it's specific to 3B's layer 22).
INTERVENE_3B_LAYER22_TO_B_REFERENCE = 0.87


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


def attn_module_for_layer(attn_modules, layer):
    """Maps a `--layers`-style layer index -- intervene.py's convention:
    0 = embeddings, 1..num_layers = the OUTPUT of decoder blocks
    0..num_layers-1, the SAME indexing `output_hidden_states=True` uses --
    to the attention module whose forward pass PRODUCES that hidden state:
    attn_modules[layer - 1] (find_attention_modules' OWN indexing has
    attn_modules[i] = decoder_layers[i]'s attention, whose output is
    hidden_states[i+1] -- so hidden_states[layer] is attn_modules[layer-1]'s
    output).

    THIS WAS A REAL, CONFIRMED BUG, not a hypothetical one: every call site
    in this file used to index `attn_modules[layer]` directly, off by one
    from every OTHER layer-indexed thing in this project (intervene.py's
    `register_layer_patch`, `relative_depth`, `resolve_layers_cli`) --
    confirmed empirically by patching "layer 2" both ways on an identical
    fake model and comparing which hidden_states index actually changed:
    intervene.py's patched(layer=2) changed hidden_states[2] first;
    heads.py's (old) attn_modules[2] changed hidden_states[3] first. A run
    asking for `--layers 22` was silently patching what intervene.py calls
    layer 23. Layer 0 (embeddings) has no attention module to patch in this
    file -- there is nothing upstream of the embedding layer for an
    o_proj-hook to touch (intervene.py's Experiment A can patch layer 0
    because it hooks the INPUT to decoder_layers[0], a different
    mechanism) -- raises a clear error rather than silently wrapping to
    attn_modules[-1] (the LAST layer)."""
    if layer < 1 or layer > len(attn_modules):
        raise ValueError(
            f"heads.py can only patch layers 1..{len(attn_modules)} (layer 0 -- the embeddings -- has no "
            "attention module to patch; that's intervene.py's Experiment A territory), got "
            f"layer={layer}."
        )
    return attn_modules[layer - 1]


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
    """head_idx may be:
      - None: ALL heads (--verify's extreme test, and the --all_heads
        positive control).
      - an int: that one head's column range.
      - a (start, end) tuple: a CONTIGUOUS group of heads [start, end) --
        see --head_groups / make_head_selectors.
    Returns the column range of the concatenated pre-o_proj tensor."""
    if head_idx is None:
        return slice(None)
    if isinstance(head_idx, tuple):
        start, end = head_idx
        return slice(start * head_dim, end * head_dim)
    return slice(head_idx * head_dim, (head_idx + 1) * head_dim)


def make_head_selectors(num_heads, head_groups=None, head_range=None, max_heads=None):
    """The list of "head selectors" to sweep -- normally individual head
    indices (int), or `head_groups` CONTIGUOUS (start, end) ranges when
    --head_groups is given. Motivation (see module docstring): the power
    note showed a full per-head grid (e.g. 16 heads x 9 layers = 144 cells)
    needs ~245 usable pairs PER CELL for Bonferroni-corrected significance
    at 80% power -- unaffordable on a Kaggle T4. Splitting each layer's
    heads into N contiguous groups cuts the cell count (and hence the
    Bonferroni penalty) by roughly num_heads/N, and each group's patch is a
    much LARGER intervention (more likely to show a detectable effect if
    the minute needs several heads together).

    Two-stage flow: run coarse with --head_groups to find a promising
    region cheaply, then re-run with --head_range restricted to that
    region (optionally with --head_groups again for an intermediate zoom,
    or without it to sweep that region's heads individually).

    `head_range=(lo, hi)` restricts the universe of heads considered BEFORE
    grouping/max_heads -- e.g. --head_range 8,12 (no --head_groups) sweeps
    heads 8,9,10,11 individually; --head_range 8,12 --head_groups 2 splits
    just that 4-head range into 2 groups of 2."""
    lo, hi = head_range if head_range is not None else (0, num_heads)
    if not (0 <= lo < hi <= num_heads):
        raise ValueError(f"--head_range ({lo},{hi}) is out of bounds for {num_heads} heads/layer.")
    universe = list(range(lo, hi))

    if head_groups is not None:
        if head_groups < 1:
            raise ValueError(f"--head_groups must be >= 1, got {head_groups}.")
        n = len(universe)
        bounds = [lo + int(round(b)) for b in np.linspace(0, n, head_groups + 1)]
        selectors = [(bounds[i], bounds[i + 1]) for i in range(head_groups) if bounds[i] < bounds[i + 1]]
    else:
        selectors = universe

    if max_heads is not None:
        selectors = selectors[:max_heads]
    return selectors


def _head_label(selector):
    """Normalizes a head selector (int for a single head, (start, end)
    tuple for a --head_groups range) into (head, head_start, head_end,
    n_heads_patched) for the trials/summary schema. `head` is always a
    plain int (the selection's own start index) so it stays sortable and
    groupable exactly like the single-head case did before --head_groups
    existed."""
    if isinstance(selector, tuple):
        start, end = selector
        return start, start, end, end - start
    return selector, selector, selector + 1, 1


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
# Power/usable-pairs diagnostics -- see the module constants above. Every
# summary cell is restricted to pairs where A's and the target's STATED
# (model-predicted, not ground-truth) minutes differ (compute_transfer_
# columns' minute_baselines_differ), so a run where most pairs don't meet
# that bar silently reads as a null result (0%, p=1.0) instead of what it
# actually is: too little data to say anything.
# ---------------------------------------------------------------------------

def estimate_usable_pairs(adapter, image_features_owners, vision_method, images_dir, pairs,
                           max_new_tokens=MAX_NEW_TOKENS):
    """Cheaply PREDICTS how many of `pairs` will end up usable for the to_B
    metric, BEFORE spending the full sweep's GPU time: reuses intervene.py's
    recompute_baselines_for_files (one plain generate() call per UNIQUE
    image, no patching -- much cheaper than the sweep) to get each image's
    own STATED minute, then counts pairs where A's and B's stated minutes
    differ -- the exact restriction summarize_heads applies afterward.
    Since nothing upstream of this baseline call differs between this
    pre-check and the sweep's own per-pair run_baseline call (same model,
    same prompt, greedy decoding), this is an exact predictor, not a rough
    guess. Prints the estimate and a loud warning if it falls below
    MIN_USABLE_PAIRS_WARN, since that's exactly the failure mode that
    motivated this function: a run that came back all-0%/p=1.0 not because
    nothing transfers, but because only 2 of 10 pairs were even eligible.
    Returns the predicted usable count."""
    unique_files = sorted({f for pair in pairs for f in (pair[0]["filename"], pair[1]["filename"])})
    baselines_df = recompute_baselines_for_files(adapter, image_features_owners, vision_method,
                                                  images_dir, unique_files, max_new_tokens=max_new_tokens)
    minute_by_file = baselines_df.set_index("filename")["baseline_minute"]

    n_usable = 0
    for a, b in pairs:
        am, bm = minute_by_file.get(a["filename"]), minute_by_file.get(b["filename"])
        if am is not None and bm is not None and not pd.isna(am) and not pd.isna(bm) and am != bm:
            n_usable += 1

    frac = n_usable / len(pairs) if pairs else 0.0
    print(f"\nPre-sweep check: {n_usable}/{len(pairs)} pair(s) ({frac:.0%}) predicted usable for the to_B "
          f"metric (A's and B's STATED baseline minutes differ -- {len(unique_files)} unique image(s), "
          "one cheap unpatched generate() call each).")
    if n_usable < MIN_USABLE_PAIRS_WARN:
        print(
            "\n" + "!" * 70 +
            f"\nWARNING: only {n_usable} pair(s) are predicted usable -- well below the "
            f"~{MIN_USABLE_PAIRS_WARN} this project treats as a floor for a permutation-test/Bonferroni "
            "summary to mean anything (see the power note below/in the summary for exactly how many "
            "are actually needed to detect a real effect). Every cell built from this few pairs will "
            "print as UNDERPOWERED rather than a trustworthy 0.0%/p=1.0000 -- but you are about to "
            "spend the FULL sweep's GPU time computing it regardless. Raise --n_pairs (the pair-building "
            "pool, not just --max_pairs, which only TRUNCATES that pool) before committing to a real "
            "run, or proceed knowingly if this is just a smoke test.\n" + "!" * 70 + "\n"
        )
    return n_usable


def min_n_per_cell_for_power(n_cells, delta=POWER_DETECT_DELTA, alpha=POWER_ALPHA, power=POWER_TARGET):
    """Minimum n (usable pairs) PER (layer, head) CELL needed to detect a
    `delta` difference between the real_b and noise to_B rates, at
    Bonferroni-corrected alpha across `n_cells` simultaneous cells, with
    `power` probability of detecting it if it's real -- the standard
    two-proportion z-test sample-size formula:
        n = (z_(alpha/2) + z_power)^2 * (p1(1-p1) + p2(1-p2)) / delta^2
    using the CONSERVATIVE p=0.5 for both rates (the maximum-variance case),
    since the true rates aren't known before running -- this project's
    replicated finding puts noise around 5-25% and real_b around 60-80%,
    which needs somewhat FEWER pairs than this; treat this as a safe upper
    bound for sizing --n_pairs, not the exact number required."""
    alpha_corrected = alpha / max(n_cells, 1)
    z_alpha2 = NormalDist().inv_cdf(1 - alpha_corrected / 2)
    z_power = NormalDist().inv_cdf(power)
    variance_term = 2 * 0.5 * (1 - 0.5)   # p(1-p) + p(1-p) at the conservative p=0.5
    n = ((z_alpha2 + z_power) ** 2) * variance_term / (delta ** 2)
    return alpha_corrected, math.ceil(n)


def power_note_text(n_cells, delta=POWER_DETECT_DELTA, alpha=POWER_ALPHA, power=POWER_TARGET):
    """Formats the power note requested alongside the summary: how many
    usable pairs PER CELL this sweep's (layer, head) cell count actually
    needs to reliably detect a real effect, given Bonferroni correction --
    printed both BEFORE the sweep (so --n_pairs can be sized against it
    without spending GPU hours first) and again in the final summary."""
    alpha_corrected, n_needed = min_n_per_cell_for_power(n_cells, delta=delta, alpha=alpha, power=power)
    return (
        f"Power note: {n_cells} (layer, head) cell(s) tested at once -> Bonferroni-corrected "
        f"alpha={alpha_corrected:.2e}. Detecting a {delta:.0%}-point real_b-vs-noise difference at "
        f"{power:.0%} power needs at least ~{n_needed} usable pairs PER CELL (conservative worst-case "
        "estimate assuming a 50/50 rate; this project's actual replicated rates -- noise ~5-25%, "
        "real_b ~60-80% -- would need somewhat fewer). Narrowing --layers/--max_heads for a targeted "
        "confirmation pass on a specific head (n_cells=1) drops this requirement sharply (e.g. ~99 "
        "pairs at the same alpha/power/delta) -- a coarse full-grid sweep and a narrow confirmation "
        "pass have very different pair budgets."
    )


def _compute_pair_baselines(adapter, image_features_owners, vision_method, images_dir, attn_by_layer,
                             df, a, b, already_used, rng, max_new_tokens):
    """Shared by run_experiment_heads and run_all_heads_control: cache each
    image's pre-o_proj activations at every layer in `attn_by_layer` while
    computing its (otherwise unmodified) run_baseline -- one plain forward
    pass plus one generate() call per image, exactly as intervene.py's own
    Experiment A does it, just also captured via capture_head_inputs.
    Returns None if the pair's sequence lengths don't match (caller should
    skip this pair), else (base_a, base_b, base_same, same_row,
    head_cache_a, head_cache_b, head_cache_same) -- base_same/same_row/
    head_cache_same are None if no same-minute partner is available for
    `a`."""
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
        return None
    return base_a, base_b, base_same, same_row, head_cache_a, head_cache_b, head_cache_same


def run_all_heads_control(adapter, image_features_owners, vision_method, attn_modules, df, images_dir,
                           layers, out_dir=None, n_pairs=ALL_HEADS_CONTROL_PAIRS, min_gap=MIN_GAP_MINUTES,
                           max_new_tokens=MAX_NEW_TOKENS, seed=SEED, exclude_pairs=None, pairs=None):
    """POSITIVE CONTROL: patch ALL heads at once (head_idx=None -- the WHOLE
    per-head-concatenated o_proj-input tensor, not one head's slice) at
    image-token positions, at each of `layers`, on a small number of pairs
    -- cheap enough to run before every real sweep (see run_experiment_heads,
    which does exactly that automatically), and standalone via --all_heads.

    WHY THIS EXISTS (found on a real run, not hypothetical): every
    INDIVIDUAL head gave EXACTLY 0.0% to_B for real_b, same_minute, AND
    noise alike. A perfect zero across all three conditions is ambiguous
    between two very different explanations:
      (a) the minute genuinely needs many heads acting together, so no
          SINGLE head's patch moves the answer -- a real finding.
      (b) the head-patching path is broken somewhere between the o_proj
          hook and generation, so NOTHING patched through it ever reaches
          the answer -- a bug, not a finding.
    --verify can't distinguish these: it only confirms tensors differ and
    propagate to hidden STATES, not that a maximal patch can move the
    ANSWER in the expected direction. Patching ALL heads at once is the
    strongest test this mechanism can run.

    Context for interpreting the number (NOT a hardcoded pass/fail
    threshold -- see INTERVENE_3B_LAYER22_TO_B_REFERENCE): intervene.py's
    own Experiment A patches the WHOLE residual stream at image-token
    positions -- a materially LARGER intervention than even all-heads-at-
    once here, which only replaces this ONE layer's attention contribution,
    not the incoming residual from earlier layers -- and found ~87% to_B
    transfer at 3B's layer 22. If all-heads-at-once here comes back
    reasonably close to that ballpark, the o_proj-hook path demonstrably
    reaches generation and case (a) is live. If it ALSO comes back near
    0%, that's case (b): the mechanism itself isn't reaching generation,
    and that must be found and fixed before any single-head or grouped
    result can be trusted."""
    if pairs is None:
        pairs = build_pairs(df, n_pairs=n_pairs, min_gap=min_gap, seed=seed, exclude_pairs=exclude_pairs)
    pairs = pairs[:n_pairs]
    lines = ["=== POSITIVE CONTROL: ALL heads patched at once (image-token positions) ==="]
    if len(pairs) == 0:
        lines.append("No pairs available -- skipped.")
        text = "\n".join(lines)
        print("\n" + text)
        return pd.DataFrame(), text

    attn_by_layer = {L: attn_module_for_layer(attn_modules, L) for L in layers}
    _, head_dim = head_geometry(attn_module_for_layer(attn_modules, layers[0]))
    already_used = {f for pair in pairs for f in (pair[0]["filename"], pair[1]["filename"])}
    rng = np.random.RandomState(seed)

    rows = []
    for pair_idx, (a, b) in enumerate(pairs):
        result = _compute_pair_baselines(adapter, image_features_owners, vision_method, images_dir,
                                         attn_by_layer, df, a, b, already_used, rng, max_new_tokens)
        if result is None:
            continue
        base_a, base_b, base_same, same_row, head_cache_a, head_cache_b, head_cache_same = result
        image_mask = base_a["image_mask"]

        for layer in layers:
            o_proj = attn_module_for_layer(attn_modules, layer).o_proj
            a_cache, b_cache = head_cache_a.get(layer), head_cache_b.get(layer)
            same_cache = head_cache_same.get(layer) if base_same is not None else None
            if a_cache is None or b_cache is None:
                continue

            def _trial(condition, values, other_file, other_base):
                with head_patched(o_proj, image_mask, values, None, head_dim, mode="replace"):
                    ans = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                pred_hour, pred_minute, ok = parse_time_answer(ans)
                rows.append({
                    "pair": pair_idx, "a_file": a["filename"], "b_file": other_file,
                    "condition": condition, "layer": layer,
                    **_baseline_fields("a", base_a), **_baseline_fields("b", other_base),
                    "patched_hour": pred_hour if ok else None, "patched_minute": pred_minute if ok else None,
                    "patched_answer": ans, "answer_changed": (ans != base_a["raw_answer"]),
                })

            _trial("real_b", b_cache, b["filename"], base_b)
            if same_cache is not None:
                _trial("same_minute", same_cache, same_row["filename"], base_same)
            noise_full = make_matched_noise(a_cache, rng)
            _trial("noise", noise_full, b["filename"], base_b)

    control_df = pd.DataFrame(rows)
    if len(control_df) == 0:
        lines.append("No usable trials (every pair had a sequence-length mismatch) -- skipped.")
        text = "\n".join(lines)
        print("\n" + text)
        return control_df, text

    # HARD, loud check -- more basic than to_B and impossible to quietly misread as a real 0%
    # measurement: does the ANSWER STRING even change from baseline on MOST all-heads-patched
    # real_b trials, regardless of whether it lands on B's specific minute? to_B could read 0% from
    # a genuinely working mechanism (the answer moves, just not to exactly B's minute) -- this
    # can't. If it fails, raise immediately rather than let the (necessarily 0%) to_B rate below be
    # read as a measurement of anything.
    real_b_all = control_df[control_df["condition"] == "real_b"]
    if len(real_b_all) > 0:
        frac_changed = float(real_b_all["answer_changed"].mean())
        if frac_changed < MIN_ANSWER_CHANGE_FRAC:
            raise RuntimeError(
                f"POSITIVE CONTROL FAILED LOUDLY: with ALL heads patched (image-token positions) using "
                f"B's own cached activations, the generated answer changed from A's unpatched baseline "
                f"in only {frac_changed:.0%} of {len(real_b_all)} real_b trial(s) across layers {layers} "
                f"-- below the {MIN_ANSWER_CHANGE_FRAC:.0%} floor for 'most trials'. This is a MORE BASIC "
                "signal than the to_B rate: a working mechanism should move the answer on most trials "
                "even before asking whether it lands on B's SPECIFIC minute, so a to_B rate computed "
                "from these trials would not be a measurement of anything -- it would just inherit this "
                "same failure. Do not proceed to interpreting to_B numbers; find why the answer isn't "
                "moving first (re-run --verify, and see run_all_heads_control's docstring for the "
                "interpretation guide once this passes)."
            )
        lines.append(f"Answer-changed check: {frac_changed:.0%} of {len(real_b_all)} real_b trial(s) "
                     f"had the answer differ from A's baseline at all (floor: {MIN_ANSWER_CHANGE_FRAC:.0%}) "
                     "-- passed, so the to_B rates below are at least based on trials where SOMETHING moved.")

    with_transfer = compute_transfer_columns(control_df)
    restricted = with_transfer[with_transfer["minute_baselines_differ"] == 1.0]
    any_signal = False
    any_real_data = False   # at least one layer had a nonzero-n real_b rate -- distinguishes
                             # "0% from actual trials" (investigate) from "0% because there was no
                             # data at all" (the SAME ambiguity this whole feature targets -- a
                             # perfect 0% from n=0 is not evidence of anything, broken or working)
    for layer in layers:
        sub = restricted[restricted["layer"] == layer]
        rates = {}
        for cond in ("real_b", "same_minute", "noise"):
            vals = sub[sub["condition"] == cond]["minute_transfer"].dropna()
            rates[cond] = (float(vals.mean()) if len(vals) else float("nan"), len(vals))
        (r_rate, r_n), (s_rate, s_n), (n_rate, n_n) = rates["real_b"], rates["same_minute"], rates["noise"]
        if r_n > 0:
            any_real_data = True
            if not np.isnan(r_rate) and r_rate > 0:
                any_signal = True
        r_text = f"{r_rate:.1%}" if r_n else "n/a"
        s_text = f"{s_rate:.1%}" if s_n else "n/a"
        n_text = f"{n_rate:.1%}" if n_n else "n/a"
        lines.append(f"  layer {layer}: to_B(real_b)={r_text} (n={r_n})  to_B(same_minute)={s_text} (n={s_n})  "
                     f"to_B(noise)={n_text} (n={n_n})")

    lines.append("")
    if not any_real_data:
        lines.append(f"-> NO layer had any pair where A's and B's stated minutes differed (n=0 "
                     f"everywhere, out of {len(pairs)} pair(s) tried) -- this control is INCONCLUSIVE, "
                     "not evidence of anything, broken or working. Raise n_pairs for the control (see "
                     "--all_heads_pairs) or try different pairs; a 0% built from zero trials is exactly "
                     "the same trap this feature exists to catch, just one level up.")
    elif any_signal:
        lines.append(f"-> at least one layer shows a nonzero real_b rate (from actual trials, not an "
                     f"empty measurement): the o_proj-hook path DOES reach generation. Compare against "
                     f"intervene.py's own whole-residual-stream reference "
                     f"({INTERVENE_3B_LAYER22_TO_B_REFERENCE:.0%} at 3B's layer 22, a LARGER intervention "
                     "than this one -- see this function's docstring) to judge whether the head path is "
                     "roughly as effective or clearly weaker; either way, a nonzero rate here means an "
                     "individual-head null result is a real 'needs many heads together' finding, not "
                     "broken plumbing.")
    else:
        lines.append("-> every layer with actual data shows a 0% real_b rate even with ALL heads patched "
                     "at once -- this is the OTHER case: the head-patching path itself is likely not "
                     "reaching generation. Do NOT trust any single-head or --head_groups result until "
                     "this is found and fixed (start by re-running --verify, then checking whether "
                     "o_proj's OWN output -- not just its input -- actually feeds the rest of the "
                     "layer, the same round-3 bug shape this project has hit before).")
    text = "\n".join(lines)
    print("\n" + text)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        control_df.to_csv(os.path.join(out_dir, "heads_positive_control_trials.csv"), index=False)
        with open(os.path.join(out_dir, "heads_positive_control.txt"), "w") as f:
            f.write(text + "\n")
    return control_df, text


# ---------------------------------------------------------------------------
# The sweep: head-level patching, reusing intervene.py's pair-building,
# baseline caching, controls, and transfer metric UNCHANGED (see module
# docstring -- everything below is orchestration, not new mechanics).
# ---------------------------------------------------------------------------

def run_experiment_heads(adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
                          df, images_dir, out_dir, layers, n_pairs=N_PAIRS, max_pairs=None,
                          min_gap=MIN_GAP_MINUTES, max_heads=None, max_new_tokens=MAX_NEW_TOKENS,
                          seed=SEED, exclude_pairs=None, head_groups=None, head_range=None):
    """Same paired design as Experiment A (build_pairs, same-minute partner,
    matched-norm noise) -- but patches one HEAD SELECTOR (a single head by
    default, or a contiguous GROUP of heads if `head_groups` is given -- see
    make_head_selectors) at a time, at the image-token positions, instead of
    the whole residual stream, swept over `layers` x every selector (or the
    first `max_heads` of them). Output schema matches Experiment A's
    (`_baseline_fields`, so `compute_transfer_columns` works unmodified)
    plus `layer`/`head`/`head_start`/`head_end`/`n_heads_patched`.

    Runs the POSITIVE CONTROL (run_all_heads_control) automatically first,
    on a handful of pairs -- see that function's docstring for why a sweep
    showing 0% everywhere is ambiguous without it."""
    rng = np.random.RandomState(seed)
    pairs = build_pairs(df, n_pairs=n_pairs, min_gap=min_gap, seed=seed, exclude_pairs=exclude_pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if len(pairs) == 0:
        raise ValueError("No valid (A, B) pairs found -- check --min_gap against this dataset's spread of hours/minutes.")

    num_layers = len(decoder_layers)
    attn_by_layer = {L: attn_module_for_layer(attn_modules, L) for L in layers}

    num_heads, head_dim = head_geometry(attn_module_for_layer(attn_modules, layers[0]))
    for L in layers:
        nh, hdim = head_geometry(attn_module_for_layer(attn_modules, L))
        if (nh, hdim) != (num_heads, head_dim):
            raise ValueError(f"Layer {L} has head geometry ({nh},{hdim}), different from layer {layers[0]}'s "
                             f"({num_heads},{head_dim}) -- this sweep assumes uniform head geometry across "
                             "swept layers (true for every architecture this project supports; a mismatch "
                             "means something unexpected about this model, not a case to silently paper over).")
    heads_to_sweep = make_head_selectors(num_heads, head_groups=head_groups, head_range=head_range,
                                         max_heads=max_heads)
    n_cells = len(layers) * len(heads_to_sweep)

    already_used = {f for pair in pairs for f in (pair[0]["filename"], pair[1]["filename"])}
    n_total_trials = len(pairs) * len(layers) * len(heads_to_sweep) * 3   # real_b, same_minute, noise
    unit = "group(s)" if head_groups is not None else "head(s)"
    print(f"heads.py: {len(pairs)} pair(s) x {len(layers)} layer(s) x {len(heads_to_sweep)} {unit} "
          f"x 3 conditions = {n_total_trials} generate() calls total (num_heads={num_heads}, "
          f"head_dim={head_dim}).")
    print(power_note_text(n_cells))
    estimate_usable_pairs(adapter, image_features_owners, vision_method, images_dir, pairs,
                          max_new_tokens=max_new_tokens)

    print("\n" + "=" * 70)
    print("POSITIVE CONTROL (automatic, before the full sweep -- see run_all_heads_control's docstring)")
    print("=" * 70)
    run_all_heads_control(adapter, image_features_owners, vision_method, attn_modules, df, images_dir,
                          layers, out_dir=out_dir, n_pairs=ALL_HEADS_CONTROL_PAIRS, min_gap=min_gap,
                          max_new_tokens=max_new_tokens, seed=seed, pairs=pairs)

    rows = []
    t_start = time.perf_counter()
    n_timed = 0

    for pair_idx, (a, b) in enumerate(tqdm(pairs, desc="heads.py pairs")):
        result = _compute_pair_baselines(adapter, image_features_owners, vision_method, images_dir,
                                         attn_by_layer, df, a, b, already_used, rng, max_new_tokens)
        if result is None:
            print(f"WARNING: sequence-length mismatch for pair ({a['filename']}, {b['filename']}) -- skipping.")
            continue
        base_a, base_b, base_same, same_row, head_cache_a, head_cache_b, head_cache_same = result
        image_mask = base_a["image_mask"]

        for layer in layers:
            o_proj = attn_module_for_layer(attn_modules, layer).o_proj
            a_cache = head_cache_a.get(layer)
            b_cache = head_cache_b.get(layer)
            same_cache = head_cache_same.get(layer) if base_same is not None else None
            if a_cache is None or b_cache is None:
                print(f"WARNING: layer {layer} was never captured for pair {pair_idx} (prefill length "
                      "mismatch or o_proj never called?) -- skipping this layer for this pair.")
                continue

            for selector in heads_to_sweep:
                head, head_start, head_end, n_heads_patched = _head_label(selector)
                col = _head_col_slice(selector, head_dim)
                head_cols = {"head": head, "head_start": head_start, "head_end": head_end,
                            "n_heads_patched": n_heads_patched}

                with head_patched(o_proj, image_mask, b_cache, selector, head_dim, mode="replace"):
                    ans = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                pred_hour, pred_minute, ok = parse_time_answer(ans)
                rows.append({
                    "seed": seed, "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"],
                    "condition": "real_b", "layer": layer, **head_cols, "num_layers": num_layers,
                    "relative_depth": relative_depth(layer, num_layers),
                    "a_true_minute": int(a["minute"]), "b_true_minute": int(b["minute"]),
                    **_baseline_fields("a", base_a), **_baseline_fields("b", base_b),
                    "patched_hour": pred_hour if ok else None, "patched_minute": pred_minute if ok else None,
                    "patched_answer": ans, "answer_changed": (ans != base_a["raw_answer"]),
                })

                if base_same is not None and same_cache is not None:
                    with head_patched(o_proj, image_mask, same_cache, selector, head_dim, mode="replace"):
                        ans_s = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                    pred_hour_s, pred_minute_s, ok_s = parse_time_answer(ans_s)
                    rows.append({
                        "seed": seed, "pair": pair_idx, "a_file": a["filename"], "b_file": same_row["filename"],
                        "condition": "same_minute", "layer": layer, **head_cols, "num_layers": num_layers,
                        "relative_depth": relative_depth(layer, num_layers),
                        "a_true_minute": int(a["minute"]), "b_true_minute": int(same_row["minute"]),
                        **_baseline_fields("a", base_a), **_baseline_fields("b", base_same),
                        "patched_hour": pred_hour_s if ok_s else None,
                        "patched_minute": pred_minute_s if ok_s else None,
                        "patched_answer": ans_s, "answer_changed": (ans_s != base_a["raw_answer"]),
                    })

                # matched-norm noise, on JUST this selection's slice of A's OWN cached
                # activations (same convention as Experiment A's full-layer noise
                # control -- see intervene.py's make_matched_noise) -- reuses B's
                # baseline as the "transfer target" for apples-to-apples comparison
                # with real_b, same as Experiment A's noise condition does.
                noise_full = a_cache.clone()
                noise_full[:, col] = make_matched_noise(a_cache[:, col], rng)
                with head_patched(o_proj, image_mask, noise_full, selector, head_dim, mode="replace"):
                    ans_n = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                pred_hour_n, pred_minute_n, ok_n = parse_time_answer(ans_n)
                rows.append({
                    "seed": seed, "pair": pair_idx, "a_file": a["filename"], "b_file": None,
                    "condition": "noise", "layer": layer, **head_cols, "num_layers": num_layers,
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
    false positives by construction. Every cell also gets an `underpowered`
    flag (n_pairs_usable < MIN_N_FOR_RELIABLE_CELL): the raw rate/p-value
    columns are still computed and saved (nothing is hidden from the CSV),
    but print_heads_summary/compare_heads_across_models must NOT treat an
    underpowered cell's numbers as a measurement -- a cell with n=2 showing
    0%/p=1.0 is not evidence of no effect, it's too little data to say
    anything (see the module constants and estimate_usable_pairs above)."""
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

        n_usable = int(len(real_b))
        rows.append({
            "layer": layer, "head": head,
            "head_start": int(sub["head_start"].iloc[0]), "head_end": int(sub["head_end"].iloc[0]),
            "n_heads_patched": int(sub["n_heads_patched"].iloc[0]),
            "relative_depth": float(sub["relative_depth"].iloc[0]),
            "n_pairs_usable": n_usable,
            "underpowered": n_usable < MIN_N_FOR_RELIABLE_CELL,
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


def _head_tick_label(row):
    """'8' for a single head, '8-11' for a --head_groups range [8, 12)."""
    if int(row["n_heads_patched"]) == 1:
        return str(int(row["head_start"]))
    return f"{int(row['head_start'])}-{int(row['head_end']) - 1}"


def plot_heads_heatmap(summary_df, out_dir, title_suffix=""):
    """Layer x head (or layer x head-group) heatmap of to_B(real_b) -
    to_B(noise) -- the primary output requested: which heads carry the
    minute, at a glance. X-axis ticks show a range ("8-11") instead of a
    single number when this summary came from a --head_groups sweep."""
    layers = sorted(summary_df["layer"].unique())
    heads = sorted(summary_df["head"].unique())
    head_labels = {h: _head_tick_label(summary_df[summary_df["head"] == h].iloc[0]) for h in heads}
    grid = np.full((len(layers), len(heads)), np.nan)
    layer_pos = {l: i for i, l in enumerate(layers)}
    head_pos = {h: i for i, h in enumerate(heads)}
    for _, r in summary_df.iterrows():
        grid[layer_pos[r["layer"]], head_pos[r["head"]]] = r["diff_vs_noise"]

    fig, ax = plt.subplots(figsize=(max(6, len(heads) * 0.5), max(4, len(layers) * 0.4)))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([head_labels[h] for h in heads])
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
    """Prints the ranked head table, restricted to RELIABLE cells
    (n_pairs_usable >= MIN_N_FOR_RELIABLE_CELL) -- an underpowered cell's
    0.0%/p=1.0000 is not a measurement, it's too little data, and including
    it in a "top heads" ranking or a concentration calculation would treat
    noise as if it were a finding. Underpowered cells are still saved (with
    their raw, unreliable values) to heads_summary.csv by summarize_heads --
    only the printed report and its derived stats exclude them."""
    lines = ["=== HEAD-LEVEL SUMMARY ===", ""]
    n_cells = len(summary_df)
    reliable = summary_df[~summary_df["underpowered"]]
    n_underpowered = int(summary_df["underpowered"].sum())
    n_sig = int((reliable["p_value_bonferroni"] < 0.05).sum())
    lines.append(f"{n_cells} (layer, head) cell(s) tested; {n_sig} significant after Bonferroni "
                 "correction (alpha=0.05) -- read the RAW p_value column with that correction in mind, "
                 "not at face value, given how many cells were tested at once.")
    if n_underpowered > 0:
        lines.append(f"{n_underpowered}/{n_cells} cell(s) are UNDERPOWERED (n_pairs_usable < "
                     f"{MIN_N_FOR_RELIABLE_CELL}) and are EXCLUDED from the ranked table and concentration "
                     "block below -- their raw (unreliable) values are still in heads_summary.csv, but a "
                     "0.0%/p=1.0000 from a couple of pairs is not evidence of 'no effect', it's too little "
                     "data to say anything.")
    lines.append("")
    lines.append(power_note_text(n_cells))
    lines.append("")

    if len(reliable) == 0:
        lines.append(f"ALL {n_cells} cell(s) are UNDERPOWERED (n_pairs_usable < {MIN_N_FOR_RELIABLE_CELL} "
                     "everywhere) -- this run cannot report reliable head-level results, full stop. Re-run "
                     "with a larger --n_pairs (see the power note above, and estimate_usable_pairs' "
                     "pre-sweep check) before drawing any conclusion from this data.")
        text = "\n".join(lines)
        print("\n" + text)
        return text

    top = reliable.sort_values("diff_vs_noise", ascending=False).head(top_n)
    header = (f"{'layer':>6}{'head':>10}{'rel_depth':>11}{'to_B(real_b)':>14}{'to_B(same_min)':>16}"
              f"{'to_B(noise)':>13}{'diff':>9}{'n':>5}{'p':>9}{'p_bonf':>9}{'sig':>5}")
    lines.append(header)
    for _, r in top.iterrows():
        sig = "*" if (not np.isnan(r["p_value_bonferroni"]) and r["p_value_bonferroni"] < 0.05) else ""
        lines.append(
            f"{int(r['layer']):>6}{_head_tick_label(r):>10}{r['relative_depth']:>11.2f}"
            f"{r['to_b_real_b']:>14.1%}{r['to_b_same_minute']:>16.1%}{r['to_b_noise']:>13.1%}"
            f"{r['diff_vs_noise']:>+9.1%}{int(r['n_pairs_usable']):>5}{r['p_value']:>9.4f}"
            f"{r['p_value_bonferroni']:>9.4f}{sig:>5}"
        )
    lines.append("")
    positive = reliable[reliable["diff_vs_noise"] > 0]["diff_vs_noise"].sort_values(ascending=False)
    total = positive.sum()
    lines.append("Concentration -- fraction of the total positive (diff_vs_noise) effect mass carried by "
                 "the top-K heads, among RELIABLE cells only (one head dominating vs. spread across many "
                 "looks very different here):")
    for k in (1, 3, 5, 10):
        if total > 0:
            frac = positive.head(k).sum() / total
            lines.append(f"  top {k:>2} head(s): {frac:.1%} of total positive effect (n={min(k, len(positive))} "
                         f"of {len(positive)} heads with a positive effect)")
        else:
            lines.append(f"  top {k:>2} head(s): insufficient data (no reliable cell showed a positive "
                         "effect vs. noise)")
    text = "\n".join(lines)
    print("\n" + text)
    return text


def compare_heads_across_models(name_a, summary_a, name_b, summary_b, sig_threshold=0.05):
    """The key cross-model report requested: how many heads carry the
    minute in each model, how concentrated the effect is, and at what
    relative depth the top heads sit -- side by side. Restricted to
    RELIABLE cells (n_pairs_usable >= MIN_N_FOR_RELIABLE_CELL) for the same
    reason print_heads_summary is -- an underpowered cell's numbers aren't
    a measurement, and letting them into a cross-model "top head" or
    concentration comparison would compare noise, not heads."""
    lines = ["=== CROSS-MODEL HEAD COMPARISON ===", ""]
    for name, summary in ((name_a, summary_a), (name_b, summary_b)):
        reliable = summary[~summary["underpowered"]]
        n_underpowered = int(summary["underpowered"].sum())
        lines.append(f"{name}:")
        if n_underpowered > 0:
            lines.append(f"  {n_underpowered}/{len(summary)} cell(s) UNDERPOWERED (n_pairs_usable < "
                         f"{MIN_N_FOR_RELIABLE_CELL}) -- excluded below.")
        if len(reliable) == 0:
            lines.append("  ALL cells UNDERPOWERED -- insufficient data for this model; re-run with a "
                         "larger --n_pairs before comparing.")
            lines.append("")
            continue
        n_sig = int((reliable["p_value_bonferroni"] < sig_threshold).sum())
        positive = reliable[reliable["diff_vs_noise"] > 0]["diff_vs_noise"].sort_values(ascending=False)
        total = positive.sum()
        top1_text = f"{(positive.head(1).sum() / total):.1%}" if total > 0 else "insufficient data"
        top3_text = f"{(positive.head(3).sum() / total):.1%}" if total > 0 else "insufficient data"
        top_row = reliable.sort_values("diff_vs_noise", ascending=False).iloc[0]
        lines.append(f"  {n_sig}/{len(reliable)} reliable (layer, head) cell(s) significant (Bonferroni, "
                     f"alpha={sig_threshold})")
        lines.append(f"  concentration: top 1 head = {top1_text} of positive effect, "
                     f"top 3 heads = {top3_text}")
        lines.append(f"  top head: layer {int(top_row['layer'])} head {_head_tick_label(top_row)} "
                     f"(relative_depth={top_row['relative_depth']:.2f}), "
                     f"diff_vs_noise={top_row['diff_vs_noise']:+.1%}")
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    return text


# ---------------------------------------------------------------------------
# --verify: prove head-level patching actually lands, before trusting a null
# ---------------------------------------------------------------------------

def _capture_o_proj_forward(o_proj):
    """Registers a genuine forward_HOOK (module, input, output) on o_proj --
    NOT a forward_pre_hook, and NOT head_patched's own hook. PyTorch fills
    `input`/`output` in with the args ACTUALLY used to call o_proj.forward()
    and what it ACTUALLY returned, regardless of anything else registered
    on the module (confirmed directly against accelerate's source: even
    though `device_map="auto"` monkey-patches `.forward` itself rather than
    using PyTorch's hook registry, `_call_impl` still resolves
    forward_pre_hooks -- including head_patched's -- into the final args
    BEFORE calling that (possibly wrapped) `.forward`, and a forward_hook's
    `output` reflects the true end-to-end return value of that whole call).

    This exists because a prior verification design compared head_patched's
    OWN pre-hook's return value to itself (via a second pre-hook chained
    after it) -- which only proves the hook FIRED, not that its effect
    survived into o_proj's real computation. A pre-hook that mutates a
    detached/cloned copy without returning it, or that gets silently
    overridden by something else, would look identical to a working one
    under that old check. This is a genuinely independent instrument: a
    different hook mechanism, reading what PyTorch guarantees is real.

    Only fires for prefill-shaped (multi-token) calls, matching
    capture_head_inputs' own guard against corrupting the capture with a
    single-token incremental-decode step. Returns (holder, handle) --
    caller must handle.remove()."""
    holder = {}

    def hook(module, input, output):
        inp = input[0] if len(input) > 0 else None
        if inp is not None and inp.shape[1] > 1:
            holder["input"] = inp[0].detach().float().cpu().clone()
            out_t = output if torch.is_tensor(output) else (output[0] if isinstance(output, tuple) else None)
            if out_t is not None:
                holder["output"] = out_t[0].detach().float().cpu().clone()

    handle = o_proj.register_forward_hook(hook)
    return holder, handle


def verify_head_patch(adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
                       pairs, images_dir, layers, max_heads_to_verify, max_new_tokens):
    """For a few (A, B) pairs and a few (layer, head) combos: confirm (a)
    the patched head's o_proj-INPUT slice at the image-token positions
    actually changes to B's cached value -- verified via a genuinely
    INDEPENDENT forward_hook (_capture_o_proj_forward), not by re-reading
    head_patched's own pre-hook return value, which would be a tautology;
    (b) o_proj's ACTUAL returned output matches what it SHOULD be given
    that patched input (recomputed from scratch via o_proj's own weight/
    bias, bypassing this file's entire hook chain) -- catches a pre-hook
    that fires but is silently discarded downstream, even if (a) somehow
    didn't; and (c) that change propagates to a downstream layer AND the
    final layer's hidden states -- same spirit as intervene.py's
    verify_decoder_patch, at head instead of full-layer granularity."""
    records = []
    lines = [f"--- Head-level patch verification (vision method: {vision_method}) ---"]
    attn_by_layer = {L: attn_module_for_layer(attn_modules, L) for L in layers}
    num_heads, head_dim = head_geometry(attn_module_for_layer(attn_modules, layers[0]))
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
            o_proj = attn_module_for_layer(attn_modules, layer).o_proj
            for head_idx in heads_to_check:
                col = slice(head_idx * head_dim, (head_idx + 1) * head_dim)

                holder, handle = _capture_o_proj_forward(o_proj)
                try:
                    with head_patched(o_proj, image_mask, head_cache_b[layer], head_idx, head_dim, mode="replace"):
                        hs_patched = forward_hidden_states(adapter.model, base_a["inputs"])
                finally:
                    handle.remove()

                actual_input = holder.get("input")
                actual_output = holder.get("output")
                if actual_input is None:
                    raise RuntimeError(
                        f"o_proj's forward() never fired for a prefill-shaped call at layer {layer} head "
                        f"{head_idx} -- the patch context manager isn't even reaching o_proj's forward at "
                        "all (a more basic failure than a patch being silently discarded downstream). "
                        "Check that attn_module_for_layer/find_attention_modules resolved the RIGHT module "
                        "and that it's actually invoked during generation."
                    )

                # Old check (kept): did the ACTUALLY-consumed input change AT ALL from A's own
                # baseline? Large = the hook fired. This alone is NOT sufficient -- it would also
                # pass if the hook wrote garbage instead of B's value, or even if it wrote the
                # RIGHT columns but the wrong VALUES.
                diff_vs_a_baseline = relative_l2_diff(head_cache_a[layer][image_mask][:, col],
                                                       actual_input[image_mask][:, col])

                # NEW, genuinely independent check #1: does the input o_proj ACTUALLY consumed
                # (captured via a real forward_hook, not head_patched's own pre-hook chain -- see
                # _capture_o_proj_forward's docstring for why that distinction matters) match what
                # we INTENDED to write (B's cached value)? Near-zero = correct. This is the check
                # that catches "the hook fired on SOMETHING but not the right value" -- comparing
                # our own pre-hook's return value to itself, as the old verification effectively
                # did, can never catch that class of bug.
                diff_vs_intended = relative_l2_diff(head_cache_b[layer][image_mask][:, col],
                                                     actual_input[image_mask][:, col])

                # NEW, genuinely independent check #2 (stronger still): recompute what o_proj's
                # OUTPUT should be from scratch -- using its OWN weight/bias, applied to the
                # INTENDED patched input -- bypassing head_patched, capture_head_inputs, and this
                # file's entire hook chain. Compare against what o_proj's forward call ACTUALLY
                # returned (also from the independent forward_hook). If a pre-hook silently failed
                # to take effect (mutated a detached copy without returning it, returned None, or
                # was overridden by something registered after it), or if o_proj's real output is
                # for any OTHER reason ignored/recomputed downstream (this project's own round-3
                # bug shape), this is the check that catches it even if diff_vs_intended somehow
                # didn't. Skipped (NaN) for a non-floating-point o_proj.weight (e.g. a quantized
                # linear where a from-scratch F.linear recomputation wouldn't be valid) -- printed
                # as "n/a", not silently treated as passing.
                diff_output_vs_expected = float("nan")
                if (actual_output is not None and hasattr(o_proj, "weight")
                        and o_proj.weight.dtype.is_floating_point):
                    expected_full_input = head_cache_a[layer].clone()
                    expected_full_input[image_mask, col] = head_cache_b[layer][image_mask, col]
                    weight = o_proj.weight.detach().float().cpu()
                    bias = o_proj.bias.detach().float().cpu() if o_proj.bias is not None else None
                    expected_output = torch.nn.functional.linear(expected_full_input, weight, bias)
                    diff_output_vs_expected = relative_l2_diff(expected_output[image_mask], actual_output[image_mask])

                downstream_idx = min(layer + 1, final_idx)
                diff_downstream = relative_l2_diff(hs_unpatched[downstream_idx][image_mask],
                                                    hs_patched[downstream_idx][image_mask])
                diff_final = relative_l2_diff(hs_unpatched[final_idx][image_mask], hs_patched[final_idx][image_mask])

                records.append({
                    "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"],
                    "layer": layer, "head": head_idx,
                    "diff_at_head": diff_vs_a_baseline, "diff_vs_intended": diff_vs_intended,
                    "diff_output_vs_expected": diff_output_vs_expected,
                    "downstream_layer": downstream_idx, "diff_downstream": diff_downstream,
                    "diff_final": diff_final,
                })
                out_text = f"{diff_output_vs_expected:.4f}" if not np.isnan(diff_output_vs_expected) else "n/a"
                lines.append(
                    f"pair {pair_idx} layer {layer} head {head_idx}: diff-vs-A-baseline={diff_vs_a_baseline:.4f}, "
                    f"diff-vs-intended={diff_vs_intended:.4f} (independent, want ~0), "
                    f"diff-output-vs-expected={out_text} (independent, want ~0), "
                    f"diff-at-layer-{downstream_idx}={diff_downstream:.4f}, diff-at-final-layer={diff_final:.4f}"
                )

    extreme_records = []
    if pairs:
        a0, b0 = pairs[0]
        a_path = os.path.join(images_dir, a0["filename"])
        head_cache_a0 = {}
        layer0 = layers[0]
        with capture_head_inputs({layer0: attn_module_for_layer(attn_modules, layer0)}, head_cache_a0):
            base_a0 = run_baseline(adapter, image_features_owners, vision_method, a_path,
                                    max_new_tokens=max_new_tokens)
        zero_vals = torch.zeros_like(head_cache_a0[layer0])
        with head_patched(attn_module_for_layer(attn_modules, layer0).o_proj, base_a0["image_mask"], zero_vals, None, head_dim, mode="replace"):
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
        if "diff_vs_intended" in df.columns and (df["diff_vs_intended"] > 0.05).all():
            problems.append(
                "Every tested head patch's ACTUALLY-consumed o_proj input (captured via an "
                "INDEPENDENT forward_hook, not head_patched's own pre-hook -- see "
                "_capture_o_proj_forward) does NOT match what we intended to write. The hook is "
                "firing on something, but not writing the intended value -- comparing our own "
                "pre-hook's return value to itself, as an earlier version of this check "
                "effectively did, cannot catch this class of bug."
            )
        valid_output_checks = (df["diff_output_vs_expected"].dropna()
                               if "diff_output_vs_expected" in df.columns else pd.Series(dtype=float))
        if len(valid_output_checks) > 0 and (valid_output_checks > 0.05).all():
            problems.append(
                "Every tested head patch's o_proj OUTPUT does not match a from-scratch "
                "recomputation using o_proj's OWN weight applied to the intended patched input -- "
                "o_proj's real computed output is being ignored or overridden somewhere "
                "downstream, even though its INPUT may look correct (this project's own round-3 "
                "bug shape: a pre-hook can fire and even write the right value, but if the "
                "downstream code recomputes from an unpatched reference instead of using o_proj's "
                "actual return value, none of it reaches the answer)."
            )
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
    diff_vs_noise, RELIABLE cells only -- an underpowered cell's diff isn't
    a measurement, and picking it as a "top head" to spend attention-
    inspection effort on would chase noise) over `n_images` sample images,
    averaging the image/text attention fraction. Returns a DataFrame, one
    row per (layer, head) with the averaged fractions -- or an empty
    DataFrame with a printed note if attention weights aren't available at
    all for this model (see inspect_attention_pattern)."""
    rng = np.random.RandomState(seed)
    sample = df.sample(n=min(n_images, len(df)), random_state=seed)
    reliable = summary_df[~summary_df["underpowered"]] if "underpowered" in summary_df.columns else summary_df
    if "n_heads_patched" in reliable.columns:
        n_group_cells = int((reliable["n_heads_patched"] > 1).sum())
        reliable = reliable[reliable["n_heads_patched"] == 1]
        if n_group_cells > 0:
            print(f"NOTE: {n_group_cells} reliable cell(s) came from a --head_groups sweep (a group's "
                  "attention pattern doesn't reduce to a single head's without a design decision this "
                  "project hasn't made) -- excluded from attention inspection. Re-run with --head_range "
                  "on a promising group (no --head_groups) to inspect its individual heads.")
    if len(reliable) == 0:
        print(f"NOTE: all {len(summary_df)} cell(s) are underpowered or group cells -- skipping "
              "attention inspection (picking a 'top head' from unreliable diffs would chase noise). "
              "Re-run with a larger --n_pairs and/or --head_range on individual heads first.")
        return pd.DataFrame(columns=["layer", "head", "diff_vs_noise", "n_images_usable",
                                     "mean_frac_image", "mean_frac_text"])
    top = reliable.sort_values("diff_vs_noise", ascending=False).head(top_n)

    rows = []
    n_unavailable = 0
    for _, r in top.iterrows():
        layer, head = int(r["layer"]), int(r["head"])
        fracs_image, fracs_text = [], []
        for _, row in sample.iterrows():
            path = os.path.join(images_dir, row["filename"])
            result = inspect_attention_pattern(adapter, attn_module_for_layer(attn_modules, layer), layer, head, path)
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
                        help="cap on heads/groups swept PER LAYER (first --max_heads of them), for a quick "
                             "smoke test")
    parser.add_argument("--head_groups", type=int, default=None,
                        help="split each swept layer's heads into N contiguous groups and patch a whole "
                             "group at once, instead of one head at a time -- far fewer cells (gentler "
                             "Bonferroni correction, much larger per-test effect); see the power note and "
                             "module docstring. Two-stage flow: run coarse with --head_groups, then re-run "
                             "narrower with --head_range on whichever group showed an effect (optionally "
                             "with --head_groups again for an intermediate zoom).")
    parser.add_argument("--head_range", type=str, default=None,
                        help="'START,END' (half-open, e.g. '8,12' = heads 8,9,10,11) -- restricts the heads "
                             "considered BEFORE --head_groups/--max_heads, for drilling into a region found "
                             "interesting by a coarser --head_groups pass. Default: all heads.")
    parser.add_argument("--all_heads", action="store_true",
                        help="POSITIVE CONTROL ONLY: patch ALL heads at once (image-token positions) at "
                             "each of --layers, on a handful of pairs, then exit WITHOUT running the full "
                             "sweep -- see run_all_heads_control's docstring. This ALSO runs automatically "
                             "(cheaply) at the top of every normal sweep, so this flag is only needed for a "
                             "quick standalone check before committing to --layers/--head_groups choices.")
    parser.add_argument("--all_heads_pairs", type=int, default=ALL_HEADS_CONTROL_PAIRS,
                        help=f"pairs used by the positive control, standalone or embedded (default "
                             f"{ALL_HEADS_CONTROL_PAIRS} -- a handful is enough to tell 'broken plumbing' "
                             "(~0%%) from 'works' (near intervene.py's own whole-layer ballpark))")
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
    head_range = None
    if args.head_range:
        lo_s, hi_s = args.head_range.split(",")
        head_range = (int(lo_s), int(hi_s))

    exclude_pairs = load_excluded_pairs(args.exclude_pairs_from) if args.exclude_pairs_from else None

    if args.verify:
        run_verification(adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
                         df, args.images_dir, out_dir, n_verify_pairs=args.verify_pairs,
                         verify_layers=verify_layers, max_heads_to_verify=args.max_heads_verify,
                         min_gap=args.min_gap, max_new_tokens=args.max_new_tokens, seed=args.seed)

    if args.all_heads:
        run_all_heads_control(adapter, image_features_owners, vision_method, attn_modules, df,
                              args.images_dir, layers, out_dir=out_dir, n_pairs=args.all_heads_pairs,
                              min_gap=args.min_gap, max_new_tokens=args.max_new_tokens, seed=args.seed,
                              exclude_pairs=exclude_pairs)
        print("\n--all_heads: positive-control-only run, skipping the full sweep.")
        return

    if args.no_sweep:
        print("\n--no_sweep: skipping the full sweep.")
        return

    trials = run_experiment_heads(
        adapter, decoder_layers, attn_modules, image_features_owners, vision_method,
        df, args.images_dir, out_dir, layers, n_pairs=args.n_pairs, max_pairs=args.max_pairs,
        min_gap=args.min_gap, max_heads=args.max_heads, max_new_tokens=args.max_new_tokens,
        seed=args.seed, exclude_pairs=exclude_pairs, head_groups=args.head_groups, head_range=head_range)

    summary = summarize_heads(trials, out_dir)
    plot_heads_heatmap(summary, out_dir, title_suffix=f" ({adapter.short_name})")
    text = print_heads_summary(summary)
    with open(os.path.join(out_dir, "heads_summary.txt"), "w") as f:
        f.write(text + "\n")

    if args.inspect_attention:
        if args.head_groups is not None:
            print("\nNOTE: --inspect_attention is skipped with --head_groups -- a group's attention "
                  "pattern doesn't reduce to a single head's without a design decision this project "
                  "hasn't made. Re-run with --head_range on a promising group (no --head_groups) to "
                  "inspect its individual heads' attention.")
        else:
            inspect_df = inspect_top_heads(adapter, attn_modules, summary, df, args.images_dir,
                                           top_n=args.top_n_inspect, n_images=args.n_images_inspect, seed=args.seed)
            inspect_df.to_csv(os.path.join(out_dir, "heads_attention_inspection.csv"), index=False)
            text_inspect = print_attention_inspection(inspect_df)
            with open(os.path.join(out_dir, "heads_attention_inspection.txt"), "w") as f:
                f.write(text_inspect + "\n")

    print(f"\nAll outputs written to '{out_dir}/'.")


if __name__ == "__main__":
    main()
