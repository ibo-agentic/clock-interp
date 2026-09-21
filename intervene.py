"""
intervene.py -- STEP 3: causal interventions on the clock-reading failure.

Background (see README.md and probe_output/): the vision encoder already
encodes the minute-hand angle at R^2=0.96, and a linear probe recovers it
from the LLM's hidden states at every layer, R^2=0.86-0.94, all the way to
the final layer -- even on images the model itself answers wrong. So the
open question is NOT "where is the angle computed" (it's there, early, and
stays there). It is: IS THE ANGLE THAT IS PRESENT EVER ACTUALLY USED to
produce the answer? This script tests that causally, two ways:

  EXPERIMENT A -- ACTIVATION PATCHING BETWEEN CLOCKS
    Cache clock B's hidden states, then run clock A (same hour, a clearly
    different minute) while substituting B's hidden states in at a chosen
    layer and set of token positions. If A's stated minute moves toward B's,
    that layer/position combination is causally used. Swept over every
    layer, three position sets, and (as a ceiling condition) the vision
    encoder's own output. Includes same-minute and matched-norm-noise
    controls.

  EXPERIMENT B -- STEERING ALONG THE PROBE DIRECTION
    Add a scaled version of a fitted probe direction to the residual stream
    at the IMAGE-TOKEN positions (uniformly, at every one of them -- see
    "WHERE THE MINUTE ACTUALLY LIVES" below for why, not the final token),
    and see whether that pushes the stated minute toward a target -- and,
    independently, whether it moves the PROBE's own readout of the angle
    (so we can tell "steering failed to move the representation" apart from
    "the representation moved but the output ignored it").

WHERE THE MINUTE ACTUALLY LIVES (from the n=60 Experiment A transfer
analysis -- this overturned an earlier "causally inert" reading of the same
data and is why Experiment B steers image tokens, not the final token):
restricted to pairs where A's and B's own STATED minutes differ (n=36),
patching clock B's hidden states into A's IMAGE-TOKEN positions makes A's
stated minute become B's own stated minute 100% of the time at layers
0/8/16, 92% at layer 21, 61% at layer 24, and 0% at layer 36. Patching the
FINAL TOKEN transfers the minute 0% of the time at EVERY layer. So the
stated minute is read out of the image-token positions somewhere in a
window around layers 16-24; after that window it lives elsewhere (not at
the final token, and not readable via this patch by layer 36). The
layer-36 (0% transfer) result is a KV-cache mechanics artifact, not
evidence layer 36 doesn't matter: during generation the patch is only ever
applied during the single prefill forward pass, and this transformers
version's decoder stack applies attention/KV-caching *inside* each block
from that block's UNPATCHED input -- patching the LAST block's OUTPUT
changes the residual stream at the image-token positions, but nothing
downstream of the last block re-attends to those positions (there's no
next block to do so), so it never reaches the logits used to pick ANY
generated token, first or later. See `print_per_layer_transfer_table_a`.

IMPORTANT FRAMING: if patching or steering does NOT move the stated answer
anywhere in the sweep, that null result IS the finding (the model has the
information and doesn't use it) -- it is reported as such, not treated as a
bug to chase. But per --verify (below), that must be demonstrated, not
assumed -- this took TWO rounds to get right:

  Round 1: a --verify run caught the vision-encoder intervention doing
  EXACTLY nothing (bit-identical downstream hidden states, no extreme test
  moved the answer) while looking superficially plausible (the captured
  tensor itself did differ). Root cause: this transformers version merges
  `self.get_image_features(pixel_values, ...).pooler_output` into
  `inputs_embeds`, produced by an EXTRA step downstream of the vision
  tower's own forward output -- hooking the tower directly (the original
  approach, and what `probe.py`'s Step 2 extraction still does) captures a
  real tensor that simply isn't the one that reaches the LLM.

  Round 2: patching `get_image_features` fixed that, but --verify STILL
  failed -- the wrapper never fired at all. Root cause: `get_image_features`
  is defined on more than one object in the model's hierarchy (the
  top-level `Qwen2_5_VLForConditionalGeneration` AND the inner
  `Qwen2_5_VLModel` it wraps), and the call that actually matters
  (`self.get_image_features(...)` inside `Qwen2_5_VLModel.forward`) goes
  through the INNER object -- a separate Python instance from the outer
  wrapper, with independently-resolved attributes. Patching only the outer
  one (found via a `hasattr` check that happened to match first) had zero
  effect on the inner one's own attribute resolution.

  The current fix (see `find_all_image_features_owners`,
  `image_features_patched`) patches EVERY object in the hierarchy that
  defines `get_image_features`, binding each wrapper explicitly with
  `types.MethodType`, and tracks which one (if any) actually fires. If NONE
  fire, `determine_vision_interception_method` falls back to patching
  `hidden_states[0]` (the fully merged embeddings) at the image-token
  positions directly -- reusing the already-verified decoder-layer-0 patch
  mechanism, which targets the exact tensor the decoder stack consumes
  without needing to know where (or whether) get_image_features is
  involved at all. `run_baseline` hard-fails (raises) if the primary method
  doesn't fire during a required capture, rather than silently recording a
  missing one -- silence is how round 2's bug went unnoticed for one
  --verify iteration. Step 2's "vision_encoder" probing results were NOT
  run through either fix and should be treated as probing the vision
  tower's raw output, not necessarily the exact tensor the LLM consumes --
  see README.md for the caveat.

Model loading, chat-input construction, generation, decoder-layer finding,
and image-token-id finding are ALL now behind adapters.py's ModelAdapter
interface (see adapters.py's module docstring) -- one class per model
family (Qwen2.5-VL, Gemma 3, InternVL3), so replicating these experiments
on a different model means writing an adapter, not editing this file.
Reuses probe.py's `load_probe_direction`/`apply_saved_probe` for Experiment
B's direction (representation-agnostic, unaffected by which model produced
the underlying activations). The vision-ceiling interception mechanics
(`image_features_patched`, `layer0_embed`) are model-AGNOSTIC by design
(see adapters.py) -- NOT reused from probe.py's vision-tower hook, for the
reason above.

Usage:
    # 1. Make sure Experiment B has directions to load: one per layer in its
    #    steering window, fit on MEAN-POOLED IMAGE-TOKEN activations (not
    #    hidden_last/final-token -- see "WHERE THE MINUTE ACTUALLY LIVES"
    #    above for why Experiment B steers image tokens now):
    python probe.py --stage direction --direction_representation hidden_meanpool \
        --direction_hand minute --direction_layers 14,16,18,20,21,22,24

    # 2. Smoke-test on a couple of pairs and a handful of layers first:
    python intervene.py --max_pairs 1 --layers 0,1,21,36

    # 3. Then the full sweep. For just a finer Experiment A layer sweep in
    #    the readout window (no need to also re-run B):
    python intervene.py
    python intervene.py --experiment a --layers 14,16,18,19,20,21,22,23,24,26
    python intervene.py --experiment a --layers readout_window   # same list, shorthand

Must run on a Kaggle T4 (16GB): one image at a time, no batching, and every
saved array is float32 (never float16 -- see probe.py's module docstring for
why that bit us before).
"""

import argparse
import contextlib
import os
import time
import types

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adapters import (find_all_image_features_owners_generic, find_decoder_layers_generic, get_adapter,
                       output_dir_for, relative_depth, resolve_layers_arg)
from eval_behavior import PROMPT, SEED, parse_time_answer
from probe import apply_saved_probe, direction_path_for_layer, load_probe_direction

DATA_CSV = "data_balanced/data.csv"
IMAGES_DIR = "data_balanced"
DIRECTION_DIR = "probe_output/probe_results"
DIRECTION_REPRESENTATION = "hidden_meanpool"   # Experiment B steers image tokens -> fit on THEIR mean-pooled rep
DIRECTION_HAND = "minute"
OUT_DIR = "intervene_output"

# The n=60 Experiment A transfer analysis (see module docstring) found the
# stated minute is causally read out of image-token positions somewhere in
# layers 16-24 -- these are the two sweep windows that follow from that:
READOUT_WINDOW_LAYERS_A = [14, 16, 18, 19, 20, 21, 22, 23, 24, 26]   # Experiment A: --layers readout_window
READOUT_WINDOW_LAYERS_B = [14, 16, 18, 20, 21, 22, 24]               # Experiment B: default --steer_layers

# The number of decoder layers found on prior runs of this model -- used
# ONLY as a fallback to flag the final-layer KV-cache mechanics artifact
# (see module docstring) when --analyze_only hasn't loaded the model to
# check the real count directly. main() always passes the real, freshly
# measured count instead of relying on this.
NUM_DECODER_LAYERS_HINT = 36

N_PAIRS = 5              # default trial count -- see module docstring for the time budget this implies
MIN_GAP_MINUTES = 15     # how far apart A and B's minutes must be (circular)
MAX_NEW_TOKENS = 16
TARGET_OFFSET_MINUTES = 30   # Experiment B's steering target: (true_minute + this) % 60
DEFAULT_ALPHAS = [-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2]


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def circular_dist_minutes(m1, m2):
    """Distance between two minute values (0-59) on a 60-minute circle."""
    d = abs(m1 - m2) % 60
    return min(d, 60 - d)


def score_shift(baseline_minute, patched_minute, source_minute):
    """How far a patched/steered minute moved toward `source_minute`,
    relative to the baseline (pre-intervention) minute. Returns
    (moved_toward, shift_score), both None if either minute is unavailable
    (a parse failure) or there was nothing to move toward (source ==
    baseline exactly).
      moved_toward: bool -- did circular distance to source_minute decrease?
      shift_score:  (d_before - d_after) / d_before, using circular
        distance. 0 = no movement, 1 = landed exactly on source_minute,
        negative = moved away, >1 = overshot past it. Deliberately NOT
        clipped, so the raw number is directly interpretable.
    """
    if baseline_minute is None or patched_minute is None:
        return None, None
    d_before = circular_dist_minutes(baseline_minute, source_minute)
    if d_before == 0:
        return None, None
    d_after = circular_dist_minutes(patched_minute, source_minute)
    return d_after < d_before, (d_before - d_after) / d_before


def angle_deg_to_minute(angle_deg):
    """Convert a hand angle (degrees, clockwise from 12) back to a minute
    value (0-59), using the same convention as clocks.py::hand_angles
    (minute_angle = minute * 6 degrees)."""
    return (angle_deg / 6.0) % 60


def relative_l2_diff(x, y):
    """||x - y|| / ||x||, as a plain float -- used by --verify to report a
    magnitude, not just a boolean, for "did this intervention actually
    change anything". NaN if x is ~0 (nothing to take a ratio against)."""
    x = torch.as_tensor(x).reshape(-1).float()
    y = torch.as_tensor(y).reshape(-1).float()
    denom = x.norm().item()
    if denom < 1e-8:
        return float("nan")
    return (x - y).norm().item() / denom


# ---------------------------------------------------------------------------
# Finding the decoder layer stack, and the patch-hook machinery
# ---------------------------------------------------------------------------

# find_decoder_layers: this used to be defined here, Qwen-specific-looking
# but actually never was (it scans by CLASS NAME PATTERN, not a hardcoded
# attribute path) -- moved to adapters.py as find_decoder_layers_generic so
# every adapter (Qwen, Gemma3, InternVL) shares the exact same scanner
# instead of three near-duplicates. Kept as a local alias since intervene.py
# (and its tests) call it by this name throughout.
find_decoder_layers = find_decoder_layers_generic


def _extract_hidden_states(args, kwargs):
    """Find the hidden_states tensor among a decoder layer's incoming
    positional args or keyword args (this varies by transformers version /
    calling convention). Returns (tensor, where) so the caller can put a
    modified tensor back in the same place."""
    if len(args) > 0 and torch.is_tensor(args[0]):
        return args[0], "arg0"
    if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
        return kwargs["hidden_states"], "kwarg"
    raise RuntimeError("Could not locate hidden_states in a decoder layer call "
                        "(checked args[0] and kwargs['hidden_states']).")


def _replace_hidden_states(args, kwargs, where, new_hs):
    if where == "arg0":
        return (new_hs,) + tuple(args[1:]), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["hidden_states"] = new_hs
    return args, new_kwargs


def _extract_output_hidden_states(output):
    """Same idea, but for what a decoder layer RETURNS -- a bare tensor, a
    tuple whose first element is hidden_states, or (in newer transformers)
    a dataclass/ModelOutput with a `.hidden_states` attribute. This is the
    same version-robustness issue that bit probe.py's vision hook."""
    if torch.is_tensor(output):
        return output, "tensor"
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0], "tuple"
    if hasattr(output, "hidden_states") and torch.is_tensor(output.hidden_states):
        return output.hidden_states, "attr:hidden_states"
    raise RuntimeError(f"Unrecognized decoder layer output type: {type(output)}")


def _replace_output_hidden_states(output, where, new_hs):
    if where == "tensor":
        return new_hs
    if where == "tuple":
        return (new_hs,) + tuple(output[1:])
    attr = where.split(":", 1)[1]
    setattr(output, attr, new_hs)
    return output


def _apply_replace_patch(hs, mask, values):
    """REPLACE hs at the masked positions with `values` (Experiment A).
    hs: (1, seq, H). mask: (seq,) bool. values: (seq, H) -- the FULL
    per-position source tensor (e.g. clock B's entire hidden state at this
    layer); only the rows at masked positions are actually read and
    written. If hs's sequence length doesn't match the mask (a
    single-token incremental decode step during generation, which has no
    "image token"/"final token" structure to speak of -- see module
    docstring on prefill vs. decode), returns hs unchanged: the patch was
    already "baked in" via the KV cache during the one prefill pass where
    the lengths did match."""
    if hs.shape[1] != mask.shape[0]:
        return hs
    new_hs = hs.clone()
    new_hs[0, mask, :] = values[mask, :].to(dtype=hs.dtype, device=hs.device)
    return new_hs


def _apply_add_patch(hs, mask, delta):
    """ADD `delta` to hs at the masked positions (Experiment B steering).
    Same shape-mismatch guard as `_apply_replace_patch`."""
    if hs.shape[1] != mask.shape[0]:
        return hs
    new_hs = hs.clone()
    new_hs[0, mask, :] = new_hs[0, mask, :] + delta.to(dtype=hs.dtype, device=hs.device)
    return new_hs


def register_layer_patch(decoder_layers, layer, mask, values, mode="replace"):
    """Register a patch hook at `layer` (0 = embeddings / the input to the
    first decoder block, 1..num_layers = the output of decoder blocks
    0..num_layers-1 -- the SAME indexing as `output_hidden_states=True`
    uses, which is what probe.py's per-layer results are keyed on).
    mode='replace' overwrites the masked positions with `values`; mode='add'
    adds `values` to them (steering). Returns the hook handle -- the caller
    must remove it (see the `patched` context manager below)."""
    apply_fn = _apply_replace_patch if mode == "replace" else _apply_add_patch

    if layer == 0:
        def hook(module, args, kwargs):
            hs, where = _extract_hidden_states(args, kwargs)
            return _replace_hidden_states(args, kwargs, where, apply_fn(hs, mask, values))
        return decoder_layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    else:
        def hook(module, args, kwargs, output):
            hs, where = _extract_output_hidden_states(output)
            return _replace_output_hidden_states(output, where, apply_fn(hs, mask, values))
        return decoder_layers[layer - 1].register_forward_hook(hook, with_kwargs=True)


@contextlib.contextmanager
def patched(decoder_layers, layer, mask, values, mode="replace"):
    """Context manager: register the patch hook, yield, always remove it
    afterward (even if the generate() call inside raises)."""
    handle = register_layer_patch(decoder_layers, layer, mask, values, mode=mode)
    try:
        yield
    finally:
        handle.remove()


# find_all_image_features_owners: moved to adapters.py as
# find_all_image_features_owners_generic(model, method_name) -- unchanged
# logic, just parameterized on the method name (Qwen/Gemma3 both use
# "get_image_features"; InternVL's analogous method is "extract_feature",
# see adapters.py) instead of hardcoding "get_image_features". A prior
# version of this fix picked ONE owner (top-level model, else model.model)
# and monkey-patched only that. A --verify run proved that was the wrong
# one: `Qwen2_5_VLForConditionalGeneration` (top-level) DOES define/inherit
# `get_image_features`, but the call that actually matters happens inside
# `Qwen2_5_VLModel.forward` (model.model) as `self.get_image_features(...)`
# -- a COMPLETELY SEPARATE Python object with its own, independently-resolved
# attributes. Patching the outer object has zero effect on the inner one.
# Since we can't be sure in advance which object's method is the one that
# actually executes for a given transformers version, we patch ALL of them
# (see `image_features_patched`) and rely on --verify's hard-failure check
# to prove at least one patch fired for real.
def find_all_image_features_owners(model, method_name="get_image_features"):
    owners = find_all_image_features_owners_generic(model, method_name)
    if not owners:
        raise AttributeError(f"Could not find any object with a `{method_name}` method "
                              "anywhere in this model's hierarchy.")
    print(f"Found {method_name}() on: {[name for name, _ in owners]}")
    return owners


def _extract_image_features_tensor(output):
    """Find the tensor that actually gets merged into inputs_embeds from
    whatever get_image_features() returned. Checked in this priority order
    because a --verify run showed `.pooler_output` (when present) is what
    the merge step actually reads -- NOT `.last_hidden_state`, even though
    a typical vision-model output carries both."""
    if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
        return output.pooler_output, "attr:pooler_output"
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state, "attr:last_hidden_state"
    if torch.is_tensor(output):
        return output, "tensor"
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0], "tuple"
    return None, None


def _replace_image_features_tensor(output, where, new_tensor):
    if where == "attr:pooler_output":
        output.pooler_output = new_tensor
        return output
    if where == "attr:last_hidden_state":
        output.last_hidden_state = new_tensor
        return output
    if where == "tensor":
        return new_tensor
    if where == "tuple":
        return (new_tensor,) + tuple(output[1:])
    raise RuntimeError(f"Unrecognized get_image_features() return type/location tag: {where}")


@contextlib.contextmanager
def image_features_patched(owners, replacement=None, capture_holder=None):
    """Monkey-patch `get_image_features` on EVERY object in `owners` (see
    `find_all_image_features_owners`) for the duration of a `with` block,
    binding each wrapper explicitly with `types.MethodType` so it shadows
    the class method exactly the way a real bound method would -- since we
    don't know in advance which owner's method is the one that actually
    executes, we intercept all of them and let whichever one fires do the
    work.

    If `replacement` is given, the firing call returns `replacement` in
    place of the real image features. If `capture_holder` is given (and
    `replacement` is None), the REAL return value is captured into
    `capture_holder["out"]` without altering behavior -- used for baseline
    caching (`run_baseline`).

    Always writes into `capture_holder` (creating a private one if none was
    given) two bookkeeping keys the caller can check afterward:
      - "fired": True if ANY patched owner's method actually ran.
      - "fired_by": which owner's name did so (from `owners`).
    `run_baseline` uses these to hard-fail rather than silently record a
    missing capture -- silent non-firing is exactly how the original bug
    went unnoticed for one --verify iteration.
    """
    state = capture_holder if capture_holder is not None else {}
    state["fired"] = False
    state["fired_by"] = None
    originals = []

    for name, owner in owners:
        original_bound = getattr(owner, "get_image_features")
        originals.append((owner, original_bound))

        def make_wrapped(original_bound, owner_name):
            def wrapped(self, *args, **kwargs):
                real_output = original_bound(*args, **kwargs)
                real_tensor, where = _extract_image_features_tensor(real_output)

                state["fired"] = True
                state["fired_by"] = owner_name

                if capture_holder is not None:
                    capture_holder["out"] = real_tensor.detach() if real_tensor is not None else None

                if replacement is None or real_tensor is None:
                    return real_output
                new_tensor = replacement.to(dtype=real_tensor.dtype, device=real_tensor.device)
                return _replace_image_features_tensor(real_output, where, new_tensor)
            return wrapped

        owner.get_image_features = types.MethodType(make_wrapped(original_bound, name), owner)

    try:
        yield state
    finally:
        for owner, original_bound in originals:
            owner.get_image_features = original_bound


# ---------------------------------------------------------------------------
# Running the model: baseline caching + patched generation
# ---------------------------------------------------------------------------

# build_inputs / generate_answer: these used to be free functions here,
# hardcoded to Qwen's chat-template + processor + prompt-length-trimming
# conventions. They're now `adapter.build_inputs(image, prompt)` /
# `adapter.generate_answer(inputs, max_new_tokens)` (see adapters.py) --
# every call site below takes an `adapter` and calls through it instead.
# This is NOT just a rename: InternVL's generate() returns ONLY the new
# tokens (no prompt-length trimming needed, unlike Qwen/Gemma3) -- baking
# that convention into a free function here would have been silently wrong
# for it. See adapters.py's module docstring for the full reasoning.


@torch.no_grad()
def forward_hidden_states(model, inputs):
    """One plain forward pass (no generation), returning the full per-layer
    hidden_states tuple (CPU float32, one (seq, H) tensor per layer). Used
    only by --verify to inspect what a patch actually produced, downstream
    of wherever it was applied -- the normal experiments never need this
    directly (run_baseline already captures it once per image)."""
    outputs = model(**inputs, output_hidden_states=True)
    hs = tuple(h[0].float().cpu() for h in outputs.hidden_states)
    del outputs
    return hs


@torch.no_grad()
def run_baseline(adapter, image_features_owners, vision_method, image_path,
                  prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS, require_vision_capture=False):
    """Run one image with NO intervention. Returns everything later trials
    need from it:
      - inputs:        the tokenized+image-processed input dict (kept on
                        the model's device), reused for every subsequent
                        generate() call on THIS image (baseline + patched).
      - hidden_states:  tuple of (seq, H) CPU float32 tensors, one per
                        layer (0=embeddings..num_layers=final) -- used as a
                        patch SOURCE (e.g. this is clock B) and, for the
                        RECEIVER image, as the norm reference for the
                        noise control.
      - image_mask:     (seq,) CPU bool tensor, which positions are image tokens.
      - vision_output:  (n_image_tokens, H) CPU float32 tensor, the image
                        representation the LLM actually reads for this
                        image -- captured differently depending on
                        `vision_method` (see `determine_vision_interception_method`):
                        for "get_image_features", whatever
                        get_image_features() actually returns; for
                        "layer0_embed", hidden_states[0] at the
                        image-token positions (the fully merged embeddings,
                        read directly off `hidden_states` below at zero
                        extra cost). Used for the vision-encoder ceiling
                        condition (see `apply_vision_replacement`).
      - raw_answer / pred_hour / pred_minute / parse_success: this image's
                        own (unpatched) generated answer.

    Always does both a plain forward pass (for hidden_states) and a
    generate() call (for the answer), even when only used as a patch
    source and the answer is discarded -- a bit of redundant compute, but
    one simple function reused everywhere beats three near-identical ones.

    If `vision_method == "get_image_features"` and `require_vision_capture`
    is True, raises immediately when NO patched owner's method fires during
    the forward pass, rather than silently returning `vision_output=None`.
    Silent non-firing is exactly how the original bug (patching the wrong
    object) went unnoticed for one --verify iteration -- callers that
    specifically need a working capture (--verify itself) must ask for this;
    ordinary experiment runs default to permissive (a broken vision capture
    there just means the vision-ceiling condition gets skipped, not that
    1000+ unrelated decoder-patch trials should crash).
    """
    inputs = adapter.build_inputs(image_path, prompt)   # already on-device (see adapters.py)

    if vision_method == "get_image_features":
        holder = {}
        with image_features_patched(image_features_owners, capture_holder=holder):
            outputs = adapter.model(**inputs, output_hidden_states=True)
        if require_vision_capture and not holder.get("fired"):
            raise RuntimeError(
                f"get_image_features() never fired for '{image_path}' -- patched owners: "
                f"{[name for name, _ in image_features_owners]}. A probe earlier in this run found "
                "this method DOES work, so this is unexpected non-determinism; do not trust any "
                "vision-ceiling result until this is understood."
            )
        vision_out = holder.get("out")
        vision_output = vision_out.float().cpu() if vision_out is not None else None
    else:
        outputs = adapter.model(**inputs, output_hidden_states=True)
        vision_output = None  # filled in below once hidden_states[0] exists

    hidden_states = tuple(h[0].float().cpu() for h in outputs.hidden_states)
    input_ids = inputs["input_ids"][0]
    image_mask = adapter.image_token_positions(inputs)
    del outputs

    if vision_method == "layer0_embed":
        vision_output = hidden_states[0][image_mask].clone()

    raw_answer = adapter.generate_answer(inputs, max_new_tokens)
    pred_hour, pred_minute, ok = parse_time_answer(raw_answer)

    return {
        "inputs": inputs, "hidden_states": hidden_states, "image_mask": image_mask,
        "seq_len": int(input_ids.shape[0]), "vision_output": vision_output,
        "raw_answer": raw_answer, "pred_hour": pred_hour, "pred_minute": pred_minute,
        "parse_success": ok,
    }


def determine_vision_interception_method(requested="auto"):
    """Decide which mechanism intercepts the image representation the LLM
    reads for THIS model. Model-agnostic: this is a pure decision based on
    `requested` (`--vision_method`) -- it doesn't probe the model at all
    (three real-model --verify rounds already showed a "does it fire" probe
    isn't sufficient evidence something actually propagates; see below).

      - "layer0_embed": patch hidden_states[0] (the fully merged
        embeddings) at the image-token positions, right before the decoder
        stack runs. This is a plain PyTorch forward-hook on a decoder
        layer -- the SAME mechanism the decoder-patch sweep already uses
        and --verify has repeatedly confirmed propagates correctly (diffs
        0.38 -> 0.50 through to the final layer). It targets the exact
        tensor the LLM consumes regardless of where, or whether,
        get_image_features is involved at all.

      - "get_image_features": monkey-patch get_image_features() (see
        `find_all_image_features_owners` / `image_features_patched`).
        Returned immediately, without re-probing here -- run --verify to
        confirm it actually propagates before trusting it. History on this
        project: round 1 patched the wrong tensor field, round 2 patched
        the wrong owner object, round 3 patched the right owner AND field
        (confirmed firing, confirmed the captured tensor differed) and
        --verify STILL found layer-0/1 hidden states bit-identical -- the
        merge step evidently reads a different reference than the one we
        mutate. Three failed verification rounds on this transformers
        version is a real result, not a debugging dead end; further
        chasing it was deliberately stopped in favor of "layer0_embed".

      - "auto" (default): resolves to "layer0_embed", for the reason
        above. Kept as a distinct name (rather than just changing
        "layer0_embed"'s default) so a future run can explicitly ask for
        "get_image_features" without it silently being what "auto" means.
    """
    if requested == "layer0_embed":
        print("Vision interception: using layer0_embed (the verified decoder-layer-0 patch mechanism).")
        return "layer0_embed"
    if requested == "get_image_features":
        print("Vision interception: using get_image_features, as explicitly requested via --vision_method "
              "-- NOT re-verified here. This method has failed --verify on three separate prior rounds on "
              "this transformers version (wrong field; wrong owner; right owner+field but the merge step "
              "still didn't read the mutated value) -- run --verify before trusting any result from it.")
        return "get_image_features"

    # "auto": prefer layer0_embed outright. get_image_features is not
    # re-probed here (see docstring) -- three real-model verification
    # rounds already showed a "fires" check isn't sufficient evidence it
    # actually propagates, and this project stopped chasing why.
    print("Vision interception (auto): defaulting to layer0_embed -- get_image_features has failed "
          "--verify on three prior rounds on this transformers version (see module docstring); pass "
          "--vision_method get_image_features to try it again explicitly.")
    return "layer0_embed"


def vision_replacement_from(vision_method, source_base):
    """Pull the correctly-shaped replacement tensor for `apply_vision_replacement`
    out of a `run_baseline(...)` result -- shape/meaning depends on
    `vision_method`: for "get_image_features", (n_image_tokens, H); for
    "layer0_embed", the FULL (seq_len, H) hidden_states[0] (only its
    image-token rows are actually used, matching `patched()`'s contract)."""
    if vision_method == "get_image_features":
        return source_base["vision_output"]
    return source_base["hidden_states"][0]


def zeroed_vision_replacement(vision_method, base):
    """A same-shape all-zero replacement, for --verify's extreme zero test."""
    if vision_method == "get_image_features":
        return torch.zeros_like(base["vision_output"])
    zeroed = base["hidden_states"][0].clone()
    zeroed[base["image_mask"]] = 0.0
    return zeroed


@contextlib.contextmanager
def apply_vision_replacement(vision_method, decoder_layers, image_features_owners, image_mask, replacement):
    """Replace the image representation the LLM reads, for one
    forward/generate call, using whichever interception method
    `determine_vision_interception_method` found actually works for this
    model. `replacement` must already be shaped for the active method (see
    `vision_replacement_from` / `zeroed_vision_replacement`)."""
    if vision_method == "get_image_features":
        with image_features_patched(image_features_owners, replacement=replacement):
            yield
    else:
        with patched(decoder_layers, layer=0, mask=image_mask, values=replacement, mode="replace"):
            yield


def make_random_noise_image(size=512, rng=None):
    """A random RGB noise image, same resolution as the clock renders (so
    it tokenizes to the same number of image patches) -- used by
    --verify's most aggressive vision-encoder sanity check: an image that
    looks nothing like a clock at all."""
    rng = rng if rng is not None else np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def position_masks(image_mask, seq_len):
    """The three position sets Experiment A sweeps over, each a (seq_len,)
    bool tensor."""
    final_token = torch.zeros(seq_len, dtype=torch.bool)
    final_token[-1] = True
    return {
        "image_tokens": image_mask,
        "final_token": final_token,
        "all_positions": torch.ones(seq_len, dtype=torch.bool),
    }


def make_matched_noise(orig_vals, rng):
    """Random Gaussian noise, shape matching `orig_vals` (seq, H), with
    EVERY ROW normalized then rescaled to match that exact position's
    original norm. Used for the noise control: the perturbation at each
    position has the same magnitude a real patch would introduce there, so
    a difference between this and the real-B-patch result isolates "is B's
    specific content what matters" from "does perturbing this position at
    all matter"."""
    noise = torch.from_numpy(rng.standard_normal(orig_vals.shape).astype(np.float32))
    row_norms = orig_vals.norm(dim=-1, keepdim=True)
    noise_norms = noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return noise * (row_norms / noise_norms)


# ---------------------------------------------------------------------------
# Experiment A: pairing
# ---------------------------------------------------------------------------

def build_pairs(df, n_pairs, min_gap=MIN_GAP_MINUTES, seed=SEED):
    """Build up to `n_pairs` (A, B) row-pairs from `df`: same hour, minutes
    at least `min_gap` apart (circular). Sampled without replacement across
    pairs (no image reused) so trials are independent of each other."""
    rng = np.random.RandomState(seed)
    used, pairs = set(), []
    hours = list(df["hour"].unique())
    rng.shuffle(hours)

    for hour in hours:
        if len(pairs) >= n_pairs:
            break
        sub = df[df["hour"] == hour].reset_index(drop=True)
        candidates = []
        for i in range(len(sub)):
            for j in range(len(sub)):
                if i == j:
                    continue
                a, b = sub.iloc[i], sub.iloc[j]
                if a["filename"] in used or b["filename"] in used:
                    continue
                if circular_dist_minutes(a["minute"], b["minute"]) >= min_gap:
                    candidates.append((a, b))
        rng.shuffle(candidates)
        for a, b in candidates:
            if a["filename"] in used or b["filename"] in used:
                continue
            pairs.append((a, b))
            used.add(a["filename"])
            used.add(b["filename"])
            if len(pairs) >= n_pairs:
                break
    return pairs


def find_same_minute_partner(df, a_row, exclude, rng):
    """A DIFFERENT image with the same minute as `a_row` (any hour), for
    the same-minute control -- patching from it should change nothing,
    since the ground-truth minute is unchanged. Returns None if none free."""
    candidates = df[(df["minute"] == a_row["minute"]) & (df["filename"] != a_row["filename"]) &
                     (~df["filename"].isin(exclude))]
    if len(candidates) == 0:
        return None
    return candidates.sample(n=1, random_state=int(rng.randint(0, 2**31 - 1))).iloc[0]


def find_real_image_with_hour_minute(df, hour, minute, rng, exclude_filename=None):
    """A real clock image showing exactly `hour`:`minute` -- used by
    Experiment B to get a concrete "what does the model say about a clock
    that actually shows the target minute" baseline, instead of only
    comparing the steered answer against the abstract true target minute
    (which the model states correctly only ~2-3% of the time even when
    looking straight at it -- see module docstring). SAME HOUR as the
    steered clock is required (not just the target minute): steering is
    only supposed to move the MINUTE representation, so comparing against a
    different-hour image would make exact_transfer/hour_transfer meaningless
    (their hour would differ from A's for a reason that has nothing to do
    with whether steering worked). `data_balanced` only has ~8 images per
    minute value spread randomly across 12 hours, so a same-hour match often
    doesn't exist -- returns None in that case (same graceful-degradation
    pattern as everywhere else in this module: the caller sets that trial's
    transfer columns to NaN rather than falling back to a different-hour
    comparison that would silently mean something different)."""
    candidates = df[(df["hour"] == hour) & (df["minute"] == minute)]
    if exclude_filename is not None:
        candidates = candidates[candidates["filename"] != exclude_filename]
    if len(candidates) == 0:
        return None
    return candidates.sample(n=1, random_state=int(rng.randint(0, 2**31 - 1))).iloc[0]


# ---------------------------------------------------------------------------
# Experiment A: the sweep
# ---------------------------------------------------------------------------

def _baseline_fields(prefix, base):
    """The (hour, minute, raw answer) of a run_baseline(...) result, as CSV
    columns named `{prefix}_baseline_hour` etc. -- saved for EVERY trial row
    (not just once per pair) so `--analyze_only` can recompute transfer
    metrics straight from experiment_a_trials.csv without re-joining
    anything. `prefix` is "a" or "b"; for condition="same_minute" rows, `base`
    is the same-minute PARTNER's baseline (not real B's) -- consistent with
    how the "b_file" column is already repurposed for that condition."""
    return {
        f"{prefix}_baseline_hour": base["pred_hour"],
        f"{prefix}_baseline_minute": base["pred_minute"],
        f"{prefix}_baseline_answer": base["raw_answer"],
    }

def run_experiment_a(adapter, decoder_layers, image_features_owners, vision_method,
                      df, images_dir, out_dir, n_pairs=N_PAIRS, max_pairs=None, min_gap=MIN_GAP_MINUTES,
                      layers=None, max_new_tokens=MAX_NEW_TOKENS, seed=SEED):
    rng = np.random.RandomState(seed)
    pairs = build_pairs(df, n_pairs=n_pairs, min_gap=min_gap, seed=seed)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if len(pairs) == 0:
        raise ValueError("No valid (A, B) pairs found -- check --min_gap against this dataset's spread of hours/minutes.")

    num_layers = len(decoder_layers)
    if layers is None:
        layers = list(range(num_layers + 1))  # 0 (embeddings) .. num_layers (final block output)
    position_sets = ["image_tokens", "final_token", "all_positions"]
    already_used = {f for pair in pairs for f in (pair[0]["filename"], pair[1]["filename"])}

    n_sweep_trials_per_pair = len(layers) * len(position_sets) * 3  # 3 conditions: real_b, same_minute, noise
    n_total_trials = len(pairs) * (n_sweep_trials_per_pair + 1)  # +1 for the vision-encoder ceiling trial
    print(f"Experiment A: {len(pairs)} pair(s) x {len(layers)} layer(s) x {len(position_sets)} position "
          f"set(s) x 3 conditions, plus {len(pairs)} vision-encoder ceiling trial(s) "
          f"= {n_total_trials} generate() calls total.")

    rows = []
    t_start = time.perf_counter()
    n_timed = 0

    for pair_idx, (a, b) in enumerate(tqdm(pairs, desc="Experiment A pairs")):
        a_path = os.path.join(images_dir, a["filename"])
        b_path = os.path.join(images_dir, b["filename"])

        base_a = run_baseline(adapter, image_features_owners, vision_method, a_path,
                               max_new_tokens=max_new_tokens)
        base_b = run_baseline(adapter, image_features_owners, vision_method, b_path,
                               max_new_tokens=max_new_tokens)

        same_row = find_same_minute_partner(df, a, exclude=already_used, rng=rng)
        base_same = None
        if same_row is not None:
            same_path = os.path.join(images_dir, same_row["filename"])
            base_same = run_baseline(adapter, image_features_owners, vision_method, same_path,
                                      max_new_tokens=max_new_tokens)

        if base_a["seq_len"] != base_b["seq_len"] or (base_same is not None and base_a["seq_len"] != base_same["seq_len"]):
            print(f"WARNING: sequence-length mismatch for pair ({a['filename']}, {b['filename']}) -- "
                  "skipping (all clocks are expected to tokenize to the same length; this would mean a "
                  "differently-sized image slipped in).")
            continue

        baseline_a_minute = base_a["pred_minute"] if base_a["parse_success"] else None
        pos_masks = position_masks(base_a["image_mask"], base_a["seq_len"])

        for layer in layers:
            for pos_name, mask in pos_masks.items():
                trial_start = time.perf_counter()

                # --- real B patch: the main condition ---
                with patched(decoder_layers, layer, mask, base_b["hidden_states"][layer], mode="replace"):
                    ans = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                pred_hour, pred_minute, ok = parse_time_answer(ans)
                moved, shift = score_shift(baseline_a_minute, pred_minute if ok else None, int(b["minute"]))
                rows.append({
                    "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"], "condition": "real_b",
                    "layer": layer, "num_layers": num_layers, "relative_depth": relative_depth(layer, num_layers),
                    "position_set": pos_name, "a_true_minute": int(a["minute"]),
                    "b_true_minute": int(b["minute"]), "source_minute": int(b["minute"]),
                    **_baseline_fields("a", base_a), **_baseline_fields("b", base_b),
                    "baseline_minute": baseline_a_minute,
                    "patched_hour": pred_hour if ok else None, "patched_minute": pred_minute if ok else None,
                    "patched_answer": ans,
                    "moved_toward": moved, "shift_score": shift,
                    "answer_changed": (ans != base_a["raw_answer"]),
                })

                # --- control (a): same-minute patch -- should change nothing ---
                if base_same is not None:
                    with patched(decoder_layers, layer, mask, base_same["hidden_states"][layer], mode="replace"):
                        ans_same = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                    pred_hour_s, pred_minute_s, ok_s = parse_time_answer(ans_same)
                    moved_s, shift_s = score_shift(baseline_a_minute, pred_minute_s if ok_s else None, int(a["minute"]))
                    rows.append({
                        "pair": pair_idx, "a_file": a["filename"], "b_file": same_row["filename"],
                        "condition": "same_minute", "layer": layer, "num_layers": num_layers,
                        "relative_depth": relative_depth(layer, num_layers), "position_set": pos_name,
                        "a_true_minute": int(a["minute"]), "b_true_minute": int(same_row["minute"]),
                        "source_minute": int(a["minute"]),
                        **_baseline_fields("a", base_a), **_baseline_fields("b", base_same),
                        "baseline_minute": baseline_a_minute,
                        "patched_hour": pred_hour_s if ok_s else None, "patched_minute": pred_minute_s if ok_s else None,
                        "patched_answer": ans_same,
                        "moved_toward": moved_s, "shift_score": shift_s,
                        "answer_changed": (ans_same != base_a["raw_answer"]),
                    })

                # --- control (b): matched-norm random noise ---
                noise_vals = make_matched_noise(base_a["hidden_states"][layer], rng)
                with patched(decoder_layers, layer, mask, noise_vals, mode="replace"):
                    ans_noise = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                pred_hour_n, pred_minute_n, ok_n = parse_time_answer(ans_noise)
                moved_n, shift_n = score_shift(baseline_a_minute, pred_minute_n if ok_n else None, int(b["minute"]))
                rows.append({
                    "pair": pair_idx, "a_file": a["filename"], "b_file": None, "condition": "noise",
                    "layer": layer, "num_layers": num_layers, "relative_depth": relative_depth(layer, num_layers),
                    "position_set": pos_name, "a_true_minute": int(a["minute"]),
                    "b_true_minute": int(b["minute"]), "source_minute": int(b["minute"]),
                    # the noise control's "transfer target" is B's baseline (same reference as
                    # real_b -- see the module docstring/README for why: this is what lets us ask
                    # "does noise achieve the same apparent transfer as real B, to the SAME target?"
                    **_baseline_fields("a", base_a), **_baseline_fields("b", base_b),
                    "baseline_minute": baseline_a_minute,
                    "patched_hour": pred_hour_n if ok_n else None, "patched_minute": pred_minute_n if ok_n else None,
                    "patched_answer": ans_noise,
                    "moved_toward": moved_n, "shift_score": shift_n,
                    "answer_changed": (ans_noise != base_a["raw_answer"]),
                })

                n_timed += 1
                if pair_idx == 0 and n_timed == 6:  # a handful of trials into the very first pair
                    elapsed = time.perf_counter() - t_start
                    per_trial = elapsed / n_timed
                    remaining = n_total_trials - n_timed
                    print(f"\n[time estimate] ~{per_trial:.2f}s/trial measured from the first pair -> "
                          f"~{per_trial * n_total_trials / 60:.1f} min total "
                          f"(~{per_trial * remaining / 60:.1f} min remaining)\n")

        # --- ceiling condition: replace the whole vision-encoder output ---
        if base_b["vision_output"] is not None:
            replacement = vision_replacement_from(vision_method, base_b)
            with apply_vision_replacement(vision_method, decoder_layers, image_features_owners,
                                           base_a["image_mask"], replacement):
                ans_v = adapter.generate_answer(base_a["inputs"], max_new_tokens)
            pred_hour_v, pred_minute_v, ok_v = parse_time_answer(ans_v)
            moved_v, shift_v = score_shift(baseline_a_minute, pred_minute_v if ok_v else None, int(b["minute"]))
            rows.append({
                "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"], "condition": "real_b",
                "layer": -1, "num_layers": num_layers, "relative_depth": float("nan"),  # not a decoder layer
                "position_set": "vision_encoder", "a_true_minute": int(a["minute"]),
                "b_true_minute": int(b["minute"]), "source_minute": int(b["minute"]),
                **_baseline_fields("a", base_a), **_baseline_fields("b", base_b),
                "baseline_minute": baseline_a_minute,
                "patched_hour": pred_hour_v if ok_v else None, "patched_minute": pred_minute_v if ok_v else None,
                "patched_answer": ans_v,
                "moved_toward": moved_v, "shift_score": shift_v,
                "answer_changed": (ans_v != base_a["raw_answer"]),
            })
        else:
            print(f"WARNING: pair {pair_idx} ({a['filename']}/{b['filename']}): vision capture failed "
                  "despite an earlier probe confirming it works -- skipping this pair's vision-ceiling "
                  "trial (not the rest of the sweep). If this recurs, don't trust the ceiling condition's "
                  "results for this run.")

    trials_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    trials_df.to_csv(os.path.join(out_dir, "experiment_a_trials.csv"), index=False)
    print(f"\nExperiment A: {len(trials_df)} trials over {len(pairs)} pairs written to "
          f"'{out_dir}/experiment_a_trials.csv'.")
    return trials_df


def summarize_experiment_a(trials_df, out_dir):
    """Per (condition, layer, position_set): n trials, % moved toward the
    patch source's minute, mean shift score, % answer changed at all."""
    summary = trials_df.groupby(["condition", "layer", "position_set"]).agg(
        n=("answer_changed", "size"),
        n_parsed=("moved_toward", lambda s: s.notna().sum()),
        pct_moved_toward=("moved_toward", "mean"),
        mean_shift_score=("shift_score", "mean"),
        pct_answer_changed=("answer_changed", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(out_dir, "experiment_a_summary.csv"), index=False)
    return summary


def plot_experiment_a(summary_df, out_dir):
    """One PNG: for each position set, % moved toward B and mean shift
    score across layers, one line per condition (real_b solid, same_minute
    dashed, noise dotted). The vision-encoder ceiling trial (layer=-1) shows
    up as the leftmost point on the same axis."""
    position_sets = ["image_tokens", "final_token", "all_positions"]
    styles = {"real_b": "-", "same_minute": "--", "noise": ":"}
    colors = {"real_b": "tab:blue", "same_minute": "tab:green", "noise": "tab:red"}

    fig, axes = plt.subplots(2, len(position_sets), figsize=(5 * len(position_sets), 8), sharex=True)
    for col, pos_name in enumerate(position_sets):
        ax_moved, ax_shift = axes[0, col], axes[1, col]
        for condition in ("real_b", "same_minute", "noise"):
            s = summary_df[(summary_df["position_set"] == pos_name) & (summary_df["condition"] == condition)]
            s = s.sort_values("layer")
            ax_moved.plot(s["layer"], s["pct_moved_toward"], styles[condition], color=colors[condition],
                          marker="o", markersize=3, label=condition)
            ax_shift.plot(s["layer"], s["mean_shift_score"], styles[condition], color=colors[condition],
                         marker="o", markersize=3, label=condition)
        ax_moved.axhline(0.5, color="lightgray", linewidth=1)
        ax_shift.axhline(0, color="lightgray", linewidth=1)
        ax_moved.set_title(pos_name)
        ax_moved.set_ylabel("% moved toward source minute")
        ax_shift.set_ylabel("mean shift score")
        ax_shift.set_xlabel("layer (-1 = vision encoder)")

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Experiment A: does patching move the stated minute toward the patch source?")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "experiment_a_layers.png"), dpi=120)
    plt.close(fig)


def print_experiment_a_summary(summary_df, trials_df, vision_method):
    """A compact text table: for each position set, the BEST layer's
    real_b result next to the same layer's controls. These are the
    "toward TRUE minute" metrics -- see print_transfer_summary_a for the
    transfer-to-baseline metrics, which are NOT confounded by how rarely
    the model states the true minute even when looking straight at the
    source (read that one as primary; this one is kept for continuity)."""
    import transformers
    lines = ["=== EXPERIMENT A SUMMARY -- TOWARD TRUE MINUTE (activation patching) ===", ""]
    n_pairs = trials_df["pair"].nunique()
    lines.append(f"n pairs: {n_pairs}")
    lines.append(f"transformers=={transformers.__version__}")
    lines.append(f"vision-ceiling mechanism: {vision_method}" +
                 (" (patches hidden_states[0] at image-token positions -- the same mechanism the "
                  "decoder-patch sweep above uses; get_image_features interception was attempted and "
                  "failed --verify on three prior rounds on this transformers version, see README.md)"
                  if vision_method == "layer0_embed" else
                  " (NOT independently re-verified in this run -- run --verify to confirm before trusting "
                  "the vision-ceiling row below)"))
    lines.append("")
    header = f"{'position_set':<14}{'best_layer':>11}{'moved%':>8}{'shift':>8}{'chg%':>7}  ||  " \
             f"{'same-min moved%':>16}{'same-min chg%':>14}  ||  {'noise moved%':>13}{'noise chg%':>11}"
    lines.append(header)

    for pos_name in ("image_tokens", "final_token", "all_positions"):
        real = summary_df[(summary_df["condition"] == "real_b") & (summary_df["position_set"] == pos_name)
                           & (summary_df["layer"] >= 0)]
        if len(real) == 0 or real["pct_moved_toward"].isna().all():
            lines.append(f"{pos_name:<14}  (no usable trials)")
            continue
        best = real.loc[real["pct_moved_toward"].idxmax()]
        layer = int(best["layer"])

        def at_layer(cond):
            row = summary_df[(summary_df["condition"] == cond) & (summary_df["position_set"] == pos_name) &
                              (summary_df["layer"] == layer)]
            return row.iloc[0] if len(row) else None

        same_row = at_layer("same_minute")
        noise_row = at_layer("noise")
        lines.append(
            f"{pos_name:<14}{layer:>11}{best['pct_moved_toward']:>8.1%}{best['mean_shift_score']:>8.2f}"
            f"{best['pct_answer_changed']:>7.1%}  ||  "
            f"{(same_row['pct_moved_toward'] if same_row is not None else float('nan')):>16.1%}"
            f"{(same_row['pct_answer_changed'] if same_row is not None else float('nan')):>14.1%}  ||  "
            f"{(noise_row['pct_moved_toward'] if noise_row is not None else float('nan')):>13.1%}"
            f"{(noise_row['pct_answer_changed'] if noise_row is not None else float('nan')):>11.1%}"
        )

    ceiling = summary_df[(summary_df["condition"] == "real_b") & (summary_df["layer"] == -1)]
    if len(ceiling):
        c = ceiling.iloc[0]
        lines.append("")
        lines.append(f"vision-encoder ceiling [mechanism: {vision_method}] (whole visual representation "
                      f"swapped): moved toward B {c['pct_moved_toward']:.1%}, "
                      f"mean shift {c['mean_shift_score']:.2f}, answer changed {c['pct_answer_changed']:.1%}")

    lines.append("")
    lines.append("Read 'moved%' RELATIVE TO ITS OWN same-min/noise columns, not against an assumed 50%:")
    lines.append("when A and B's minutes are far apart (up to 30, the maximum on a 60-minute circle),")
    lines.append("baseline sits near the worst possible starting point, so almost ANY perturbation --")
    lines.append("including pure noise -- has good odds of looking 'closer' by chance. The noise and")
    lines.append("same-minute columns above already reflect that same geometry, so if real_b's numbers")
    lines.append("stay close to noise's at the same layer/position (not clearly higher), and 'chg%' stays")
    lines.append("low across the WHOLE sweep including the vision-encoder ceiling, THAT null result is")
    lines.append("the finding: the angle information is present but the model's answer doesn't use it.")
    lines.append("")
    lines.append("CAUTION -- these 'moved toward TRUE minute' numbers are further confounded on top of the")
    lines.append("chance-level geometry above: the model states the TRUE minute correctly only ~2-3% of")
    lines.append("the time even when looking straight at the source, so 'moved%'/'chg%' here are capped by")
    lines.append("that baseline failure rate regardless of what the patch actually did. See the")
    lines.append("TRANSFER-TO-BASELINE summary below for the metric that isn't confounded this way --")
    lines.append("compare it, not this table, when judging whether the intervention had a real effect.")

    text = "\n".join(lines)
    print("\n" + text)
    return text


# ---------------------------------------------------------------------------
# Experiment A: transfer-to-baseline metric (fixes the "moved toward TRUE
# minute" confound above -- see module docstring)
# ---------------------------------------------------------------------------
#
# The "toward true minute" metrics compare the PATCHED answer to the patch
# source's TRUE minute. But the model states the true minute correctly only
# ~2-3% of the time even when looking directly at the source (unpatched) --
# so "didn't move toward the true minute" is mostly just that baseline
# failure rate showing up again, not evidence the patch had no effect. The
# vision-ceiling result makes this obvious: swapping the ENTIRE visual
# representation changes the answer ~97% of the time, but only "moves
# toward B's true minute" ~38% of the time -- a huge causal effect, mostly
# invisible to the old metric.
#
# The fix: compare the PATCHED answer to the model's OWN UNPATCHED answer
# about the patch source (its "baseline"), not to the source's true minute.
# This is well-defined regardless of whether that baseline answer is
# correct, so it isn't capped by the ~2-3% true-answer rate.

def compute_transfer_columns(trials_df):
    """Add transfer-to-baseline columns to a copy of `trials_df`:
      - baselines_differ: 1.0/0.0/NaN -- do A's and the patch target's
        UNPATCHED answers already differ? (NaN if either baseline didn't
        parse.) Trials where this is 0.0 (baselines already agree) are
        EXCLUDED from the transfer rates below by the caller, since
        "transfer" would be trivially "successful" there even for a
        completely inert patch.
      - exact_transfer / minute_transfer / hour_transfer: 1.0/0.0/NaN,
        comparing the PATCHED answer to the TARGET's baseline answer
        (b_baseline_*) -- NaN if either didn't parse, or if this trials.csv
        predates saving patched_hour (see `backfill_baselines`), in which
        case exact/hour_transfer are NaN throughout but minute_transfer
        still works (patched_minute has always been saved).
    "Target" means B's baseline for condition in (real_b, noise), and the
    same-minute partner's baseline for condition == same_minute -- matching
    how the `b_file`/`b_baseline_*` columns are already populated per
    condition (see `_baseline_fields`).
    """
    df = trials_df.copy()
    if "patched_hour" not in df.columns:
        df["patched_hour"] = np.nan
    for col in ("a_baseline_hour", "a_baseline_minute", "b_baseline_hour", "b_baseline_minute"):
        if col not in df.columns:
            df[col] = np.nan

    a_valid = df["a_baseline_hour"].notna() & df["a_baseline_minute"].notna()
    b_valid = df["b_baseline_hour"].notna() & df["b_baseline_minute"].notna()
    both_valid = a_valid & b_valid
    same_as_target = (df["a_baseline_hour"] == df["b_baseline_hour"]) & \
                      (df["a_baseline_minute"] == df["b_baseline_minute"])
    df["baselines_differ"] = np.where(both_valid, (~same_as_target).astype(float), np.nan)

    p_valid = df["patched_hour"].notna() & df["patched_minute"].notna()
    usable = p_valid & b_valid
    exact = (df["patched_hour"] == df["b_baseline_hour"]) & (df["patched_minute"] == df["b_baseline_minute"])
    minute_eq = (df["patched_minute"] == df["b_baseline_minute"])
    hour_eq = (df["patched_hour"] == df["b_baseline_hour"])

    df["exact_transfer"] = np.where(usable, exact.astype(float), np.nan)
    df["hour_transfer"] = np.where(usable, hour_eq.astype(float), np.nan)
    # minute_transfer only needs patched_minute + b_baseline_minute (not hour),
    # so it stays computable even for a pre-patched_hour (old-format) CSV.
    minute_usable = df["patched_minute"].notna() & df["b_baseline_minute"].notna()
    df["minute_transfer"] = np.where(minute_usable, minute_eq.astype(float), np.nan)

    # --- per-layer to_B / stay_A / other breakdown ---
    # A stricter, MINUTE-ONLY version of "baselines_differ" above (which
    # requires the full hour+minute tuple to differ): restricts to pairs
    # where A's and the target's STATED MINUTES differ, which is the actual
    # test of whether patching moved the stated minute specifically (an hour
    # mismatch with the same stated minute wouldn't tell us anything about
    # minute transfer either way). Used by summarize_per_layer_transfer_a /
    # print_per_layer_transfer_table_a -- the per-layer curve is the primary
    # result now (see module docstring's "WHERE THE MINUTE ACTUALLY LIVES").
    a_minute_valid = df["a_baseline_minute"].notna()
    b_minute_valid = df["b_baseline_minute"].notna()
    both_minute_valid = a_minute_valid & b_minute_valid
    df["minute_baselines_differ"] = np.where(
        both_minute_valid, (df["a_baseline_minute"] != df["b_baseline_minute"]).astype(float), np.nan)

    stay_a_usable = df["patched_minute"].notna() & a_minute_valid
    stay_a = (df["patched_minute"] == df["a_baseline_minute"])
    df["stay_a"] = np.where(stay_a_usable, stay_a.astype(float), np.nan)

    other_usable = df["patched_minute"].notna() & both_minute_valid
    other_outcome = ~((df["patched_minute"] == df["b_baseline_minute"]) |
                       (df["patched_minute"] == df["a_baseline_minute"]))
    df["other_outcome"] = np.where(other_usable, other_outcome.astype(float), np.nan)

    return df


def baseline_agreement_stats(trials_df):
    """How often A's and the patch target's BASELINE (unpatched) answers
    already agree, per condition -- computed once per (pair, condition),
    since baseline answers don't depend on layer/position_set. High
    agreement means the transfer test below has limited power for that
    condition (most of its pairs get excluded from the "differ" restriction
    -- report this so that's visible, not silently baked into a smaller n)."""
    df = compute_transfer_columns(trials_df.drop_duplicates(subset=["pair", "condition"]))
    df = df[df["baselines_differ"].notna()]
    if len(df) == 0:
        return pd.DataFrame(columns=["n_pairs", "frac_identical"])
    agg = df.groupby("condition").agg(
        n_pairs=("baselines_differ", "size"),
        frac_identical=("baselines_differ", lambda s: (1 - s).mean()),
    )
    return agg


def summarize_transfer_a(trials_df, out_dir):
    """Per (condition, layer, position_set): transfer-to-baseline rates,
    computed ONLY on trials where the baseline answers already differ (see
    compute_transfer_columns) -- otherwise transfer would be trivially
    satisfied regardless of the patch."""
    df = compute_transfer_columns(trials_df)
    restricted = df[df["baselines_differ"] == 1.0]

    summary = restricted.groupby(["condition", "layer", "position_set"]).agg(
        n=("exact_transfer", "size"),
        n_usable=("minute_transfer", lambda s: s.notna().sum()),
        exact_transfer_rate=("exact_transfer", "mean"),
        minute_transfer_rate=("minute_transfer", "mean"),
        hour_transfer_rate=("hour_transfer", "mean"),
    ).reset_index()

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "experiment_a_transfer_summary.csv"), index=False)
    df.to_csv(os.path.join(out_dir, "experiment_a_trials_with_transfer.csv"), index=False)
    return summary


def print_transfer_summary_a(summary_df, agreement_stats, trials_df):
    """A compact text table, mirroring `print_experiment_a_summary`'s
    layout: for each position set, the BEST layer's real_b transfer rate
    next to same_minute/noise at that same layer, plus the vision-ceiling
    row. This is the PRIMARY table for judging whether the intervention had
    a real effect -- see print_experiment_a_summary for why the older
    'toward true minute' table is confounded."""
    lines = ["=== EXPERIMENT A SUMMARY -- TRANSFER TO BASELINE (fixes the true-minute confound) ===", ""]
    lines.append("Compares the PATCHED answer to what the model ITSELF says (unpatched) about the patch")
    lines.append("source/target, NOT to its true minute -- well-defined regardless of whether that answer")
    lines.append("is correct, so it isn't capped by how rarely the model states the truth.")
    lines.append("")
    if "patched_hour" not in trials_df.columns or trials_df["patched_hour"].isna().all():
        lines.append("NOTE: patched_hour is missing from this data (an older run) -- exact_transfer and")
        lines.append("hour_transfer are NaN throughout; only minute_transfer is available. Re-run the full")
        lines.append("sweep to get exact/hour transfer for this run.")
        lines.append("")

    lines.append("Baseline agreement rate per condition (A's and the target's UNPATCHED answers already")
    lines.append("identical -- EXCLUDED from the transfer rates below, since transfer there would be")
    lines.append("trivially 'successful' regardless of what the patch did):")
    for cond in ("real_b", "same_minute", "noise"):
        if cond in agreement_stats.index:
            r = agreement_stats.loc[cond]
            lines.append(f"  {cond:<14}: {r['frac_identical']:.1%} of {int(r['n_pairs'])} pairs")
    lines.append("")

    header = f"{'position_set':<14}{'best_layer':>11}  ||  {'condition':<13}{'exact%':>8}{'minute%':>9}" \
             f"{'hour%':>7}{'n_usable':>10}"
    lines.append(header)

    for pos_name in ("image_tokens", "final_token", "all_positions"):
        real = summary_df[(summary_df["condition"] == "real_b") & (summary_df["position_set"] == pos_name) &
                           (summary_df["layer"] >= 0) & (summary_df["n_usable"] >= 5)]
        if len(real) == 0:
            lines.append(f"{pos_name:<14}  (no usable trials -- fewer than 5 differing-baseline pairs "
                         "at every layer)")
            continue
        best_layer = int(real.loc[real["minute_transfer_rate"].idxmax(), "layer"])

        for i, cond in enumerate(("real_b", "same_minute", "noise")):
            row = summary_df[(summary_df["condition"] == cond) & (summary_df["position_set"] == pos_name) &
                              (summary_df["layer"] == best_layer)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            prefix = f"{pos_name:<14}{best_layer:>11}" if i == 0 else f"{'':<14}{'':>11}"
            lines.append(
                f"{prefix}  ||  {cond:<13}{r['exact_transfer_rate']:>8.1%}{r['minute_transfer_rate']:>9.1%}"
                f"{r['hour_transfer_rate']:>7.1%}{int(r['n_usable']):>10}"
            )

    ceiling = summary_df[(summary_df["condition"] == "real_b") & (summary_df["layer"] == -1)]
    if len(ceiling):
        c = ceiling.iloc[0]
        lines.append("")
        lines.append(f"vision-encoder ceiling: exact_transfer={c['exact_transfer_rate']:.1%}, "
                     f"minute_transfer={c['minute_transfer_rate']:.1%}, hour_transfer={c['hour_transfer_rate']:.1%} "
                     f"(n_usable={int(c['n_usable'])}/{int(c['n'])})")

    lines.append("")
    lines.append("As with the true-minute table, compare real_b's numbers to same-minute/noise AT THE SAME")
    lines.append("layer, not in isolation -- that's the actual test of whether B's specific content matters.")

    text = "\n".join(lines)
    print("\n" + text)
    return text


def summarize_per_layer_transfer_a(trials_df, out_dir):
    """Per (condition, layer, position_set): the to_B / stay_A / other
    breakdown, restricted to pairs where A's and the target's STATED
    MINUTES differ (see `compute_transfer_columns`'s `minute_baselines_differ`)
    -- the per-layer curve this produces (not just a single best layer) is
    the primary result for judging where the stated minute is read out of,
    per the n=60 transfer analysis (see module docstring)."""
    df = compute_transfer_columns(trials_df)
    restricted = df[df["minute_baselines_differ"] == 1.0]

    agg_kwargs = dict(
        n=("minute_transfer", "size"),
        n_usable=("minute_transfer", lambda s: s.notna().sum()),
        to_b_rate=("minute_transfer", "mean"),
        stay_a_rate=("stay_a", "mean"),
        other_rate=("other_outcome", "mean"),
    )
    # relative_depth/num_layers are constant within each (layer, position_set) group
    # (every trial at a given layer has the same depth) -- carried through so
    # compare_models.py can plot the readout curve against relative depth across
    # models with different layer counts, not just this model's own indices.
    if "relative_depth" in restricted.columns:
        agg_kwargs["relative_depth"] = ("relative_depth", "first")
    if "num_layers" in restricted.columns:
        agg_kwargs["num_layers"] = ("num_layers", "first")
    summary = restricted.groupby(["condition", "layer", "position_set"]).agg(**agg_kwargs).reset_index()

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "experiment_a_per_layer_transfer.csv"), index=False)
    return summary


def print_per_layer_transfer_table_a(summary_df, num_layers=None):
    """The per-layer to_B / stay_A / other curve, for EVERY swept layer (not
    just the best one) -- this table IS the primary result of the n=60
    transfer analysis: it's what shows the stated minute being read out of
    image-token positions in a window around layers 16-24, and nowhere at
    the final-token position at any layer. `num_layers`, when known (pass
    `len(decoder_layers)` from a live run; falls back to
    NUM_DECODER_LAYERS_HINT when --analyze_only hasn't loaded the model),
    flags the model's LAST decoder layer with the KV-cache mechanics-artifact
    caveat (see module docstring) -- a 0% transfer rate there is expected and
    uninterpretable as "this layer doesn't matter", not a real result."""
    if num_layers is None:
        num_layers = NUM_DECODER_LAYERS_HINT

    lines = ["=== EXPERIMENT A SUMMARY -- PER-LAYER to_B / stay_A / other (image-token readout) ===", ""]
    lines.append("Restricted to pairs where A's and the target's OWN STATED MINUTES differ (not just where")
    lines.append("the full hour+minute baseline answer differs) -- the direct test of whether patching moved")
    lines.append("the stated MINUTE specifically. to_B: patched minute == target's baseline minute. stay_A:")
    lines.append("patched minute == A's own baseline minute (patch had no visible effect). other: neither --")
    lines.append("the patch changed the answer, but not to either recognizable value. These three sum to")
    lines.append("~100% of n_usable within each row.")
    lines.append("")

    header = f"{'layer':>6}  ||  {'real_b':^28}  ||  {'same_minute to_B%':>18}  ||  {'noise to_B%':>13}"
    lines.append(header)
    subheader = f"{'':>6}  ||  {'to_B%':>8}{'stay_A%':>10}{'other%':>9}{'n_usable':>10}  ||  {'':>18}  ||  {'':>13}"
    lines.append(subheader)

    for pos_name in ("image_tokens", "final_token", "all_positions"):
        pos_df = summary_df[summary_df["position_set"] == pos_name]
        if len(pos_df) == 0:
            continue
        lines.append(f"\n--- position_set = {pos_name} ---")
        layers = sorted(pos_df[pos_df["layer"] >= 0]["layer"].unique())
        for layer in layers:
            real = pos_df[(pos_df["condition"] == "real_b") & (pos_df["layer"] == layer)]
            same = pos_df[(pos_df["condition"] == "same_minute") & (pos_df["layer"] == layer)]
            noise = pos_df[(pos_df["condition"] == "noise") & (pos_df["layer"] == layer)]
            if len(real) == 0:
                continue
            r = real.iloc[0]
            same_str = f"{same.iloc[0]['to_b_rate']:>18.1%}" if len(same) else f"{'--':>18}"
            noise_str = f"{noise.iloc[0]['to_b_rate']:>13.1%}" if len(noise) else f"{'--':>13}"
            marker = " *** FINAL LAYER -- see caveat below ***" if int(layer) == num_layers else ""
            lines.append(
                f"{int(layer):>6}  ||  {r['to_b_rate']:>8.1%}{r['stay_a_rate']:>10.1%}{r['other_rate']:>9.1%}"
                f"{int(r['n_usable']):>10}  ||  {same_str}  ||  {noise_str}{marker}"
            )
        # vision-ceiling row, if present for this position_set's condition grid
        ceiling = summary_df[(summary_df["condition"] == "real_b") & (summary_df["layer"] == -1)]
        if len(ceiling) and pos_name == "image_tokens":
            c = ceiling.iloc[0]
            lines.append(f"{'vision-enc':>6}  ||  {c['to_b_rate']:>8.1%}{c['stay_a_rate']:>10.1%}"
                         f"{c['other_rate']:>9.1%}{int(c['n_usable']):>10}  ||  (n/a)  ||  (n/a)")

    lines.append("")
    lines.append(f"NOTE on the layer-{num_layers} (final decoder layer) row above: patching it is a KV-cache")
    lines.append("mechanics artifact, not a measurement of whether that layer matters. The patch only ever")
    lines.append("applies during the single prefill forward pass; this transformers version computes")
    lines.append("attention/KV-caching for each block from THAT block's UNPATCHED input, and there is no")
    lines.append("block downstream of the last one to re-attend to the patched positions -- so a patch to")
    lines.append("the last layer's image-token positions never reaches the logits for ANY generated token,")
    lines.append("first or later. A near-0% rate there is expected regardless of whether the minute is")
    lines.append("causally read out of image tokens at all -- read the window BELOW it (e.g. 16-24) instead.")

    text = "\n".join(lines)
    print("\n" + text)
    return text


def recompute_baselines_for_files(adapter, image_features_owners, vision_method,
                                   images_dir, filenames, max_new_tokens=MAX_NEW_TOKENS):
    """Re-run ONLY the (cheap) baseline generate() call for each filename in
    `filenames` (deduplicated), with NO patching -- used to backfill full
    baseline answers (hour, minute, raw text) for an --analyze_only
    re-analysis of an experiment_a_trials.csv that predates saving them,
    without redoing the full (slow) patching sweep. Returns a DataFrame:
    filename, baseline_hour, baseline_minute, baseline_answer,
    baseline_parse_success."""
    unique_files = sorted(set(f for f in filenames if isinstance(f, str) and f))
    print(f"Recomputing baselines for {len(unique_files)} unique image(s) ({max_new_tokens} tokens each, "
          "no patching -- much cheaper than the full sweep).")
    rows = []
    for fn in tqdm(unique_files, desc="Recomputing baselines"):
        path = os.path.join(images_dir, fn)
        base = run_baseline(adapter, image_features_owners, vision_method, path,
                             max_new_tokens=max_new_tokens)
        rows.append({
            "filename": fn, "baseline_hour": base["pred_hour"], "baseline_minute": base["pred_minute"],
            "baseline_answer": base["raw_answer"], "baseline_parse_success": base["parse_success"],
        })
    return pd.DataFrame(rows)


def backfill_baselines(trials_df, baselines_df):
    """Merge a `recompute_baselines_for_files(...)` lookup table into
    `trials_df`, filling in a_baseline_*/b_baseline_* wherever they're
    missing or absent entirely (an old-format CSV). Does NOT and CANNOT
    backfill patched_hour/patched_answer -- those require re-running the
    actual PATCHED trial (the full sweep), not just a baseline; prints a
    clear note if they're absent so that limitation isn't silently hidden."""
    df = trials_df.copy()
    lut = baselines_df.set_index("filename")

    def lookup(filenames, field):
        return [lut.at[f, field] if (isinstance(f, str) and f in lut.index) else np.nan for f in filenames]

    if "a_baseline_hour" not in df.columns or df["a_baseline_hour"].isna().all():
        df["a_baseline_hour"] = lookup(df["a_file"], "baseline_hour")
        df["a_baseline_minute"] = lookup(df["a_file"], "baseline_minute")
        df["a_baseline_answer"] = lookup(df["a_file"], "baseline_answer")
        print(f"Backfilled a_baseline_* for {df['a_baseline_hour'].notna().sum()}/{len(df)} rows.")
    if "b_baseline_hour" not in df.columns or df["b_baseline_hour"].isna().all():
        df["b_baseline_hour"] = lookup(df["b_file"], "baseline_hour")
        df["b_baseline_minute"] = lookup(df["b_file"], "baseline_minute")
        df["b_baseline_answer"] = lookup(df["b_file"], "baseline_answer")
        print(f"Backfilled b_baseline_* for {df['b_baseline_hour'].notna().sum()}/{len(df)} rows.")
    if "patched_hour" not in df.columns:
        print("NOTE: this trials.csv predates saving patched_hour/patched_answer -- exact_transfer and "
              "hour_transfer cannot be computed retroactively for it (only minute_transfer can, from the "
              "already-saved patched_minute). Re-run the full sweep to get exact/hour transfer for this run.")
    return df


# ---------------------------------------------------------------------------
# Experiment B: steering along the probe direction, at the IMAGE-TOKEN
# positions, swept over the readout window found by Experiment A's transfer
# analysis (see module docstring's "WHERE THE MINUTE ACTUALLY LIVES") --
# NOT the final token, which that analysis found transfers the minute 0% of
# the time at every layer. This is a rewrite of the earlier final-token
# version: that version's null result wasn't a finding, it was steering the
# wrong position entirely.
# ---------------------------------------------------------------------------
#
# NOT IMPLEMENTED: an option to steer only the image tokens nearest the
# minute hand's tip, rather than uniformly across all of them. This would
# need a reliable pixel-to-patch-token mapping (this model's vision
# patchification grid, folded through matplotlib's default subplot-to-pixel
# layout for clocks.py's renders) that isn't already established anywhere
# in this codebase, and getting it wrong would silently corrupt which
# tokens get steered -- worse than not having the option. Skipped rather
# than guessed at; uniform steering across all image tokens is what's
# implemented below.

def load_directions_for_layers(direction_dir, representation, hand, layers):
    """Load one steering direction per layer in `layers` (each saved by
    `probe.py --stage direction --direction_representation ... --direction_layers ...`,
    fit on MEAN-POOLED IMAGE-TOKEN activations so it matches the position
    Experiment B now steers -- see module docstring). Returns
    {layer: direction_dict}. Raises FileNotFoundError listing every missing
    layer AND the exact probe.py command to fit them, if any are absent --
    a silently partial sweep (steering only some of the requested layers)
    would look identical to a full one in the output; loud failure here
    is cheaper than that ambiguity."""
    directions, missing = {}, []
    for layer in layers:
        path = direction_path_for_layer(direction_dir, representation, hand, layer)
        if os.path.exists(path):
            directions[layer] = load_probe_direction(path)
        else:
            missing.append((layer, path))
    if missing:
        missing_layers_str = ",".join(str(l) for l, _ in missing)
        raise FileNotFoundError(
            "Experiment B: missing steering direction file(s):\n" +
            "\n".join(f"  layer {l}: {p}" for l, p in missing) +
            f"\nFit them with:\n  python probe.py --stage direction --direction_representation "
            f"{representation} --direction_hand {hand} --direction_layers {missing_layers_str}"
        )
    return directions


def run_experiment_b(adapter, decoder_layers, image_features_owners, vision_method,
                      directions, df, images_dir, out_dir, n_trials=N_PAIRS, max_pairs=None,
                      target_offset=TARGET_OFFSET_MINUTES, alphas=None,
                      max_new_tokens=MAX_NEW_TOKENS, seed=SEED):
    """`directions`: {layer: direction_dict}, one per layer to steer at (see
    `load_directions_for_layers`) -- each MUST be fit on mean-pooled
    image-token activations (representation='hidden_meanpool'), since the
    steering delta is added uniformly across every image-token position and
    the probe-readout check below reads it off the mean-pooled vector to
    match (adding the SAME delta to every position shifts their mean by
    exactly that delta -- no extra forward pass needed to check this,
    same trick the old final-token version used with the last-token vector).

    Three steering directions, all swept over the SAME alphas/layers/norms:
      - probe_direction: points toward the TARGET minute (the thing being tested)
      - random_direction: one fixed random unit vector, reused everywhere,
        for an apples-to-apples "does the SPECIFIC direction matter" control
      - own_angle_direction: points toward THIS clock's own TRUE minute --
        since the representation already encodes something close to that
        (R^2 ~0.96, see probe_output/), steering toward it should be close
        to a no-op. If it moves the answer as much as probe_direction does,
        that means the intervention is just generically disruptive at this
        magnitude, not that it's doing anything about the MINUTE
        specifically -- a sanity check `random_direction` alone can't give,
        since a random direction almost certainly points nowhere near
        either angle.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if max_pairs is not None:
        n_trials = min(n_trials, max_pairs)

    layers = sorted(directions.keys())
    any_direction = directions[layers[0]]
    num_layers = len(decoder_layers)
    print(f"Experiment B: steering IMAGE-TOKEN positions at layer(s) {layers} (from "
          f"'{any_direction['representation']}'/'{any_direction['hand']}'; per-layer train R^2: " +
          ", ".join(f"{L}={directions[L]['r2_train']:.3f}" for L in layers) + ").")

    rng = np.random.RandomState(seed)
    trial_images = df.sample(frac=1, random_state=seed).reset_index(drop=True).head(n_trials)

    # One fixed random direction, reused for every trial/layer/alpha, so the
    # probe-direction vs. random-direction comparison is apples to apples.
    n_features_full = any_direction["w_sin_full"].shape[0]
    random_dir = rng.standard_normal(n_features_full)
    random_dir_hat = random_dir / np.linalg.norm(random_dir)

    direction_names = ("probe_direction", "random_direction", "own_angle_direction")
    n_total = len(trial_images) * len(layers) * len(alphas) * len(direction_names)
    print(f"Experiment B: {len(trial_images)} image(s) x {len(layers)} layer(s) x {len(alphas)} alpha(s) x "
          f"{len(direction_names)} directions (probe / random / own-angle) = {n_total} generate() calls, "
          f"plus up to {len(trial_images)} extra baseline calls (one per image, for a real SAME-HOUR "
          "target-minute clock's own unpatched answer).")

    rows = []
    t_start = time.perf_counter()
    n_timed = 0

    for _, row in tqdm(trial_images.iterrows(), total=len(trial_images), desc="Experiment B images"):
        path = os.path.join(images_dir, row["filename"])
        base = run_baseline(adapter, image_features_owners, vision_method, path,
                             max_new_tokens=max_new_tokens)

        true_hour = int(row["hour"])
        true_minute = int(row["minute"])
        target_minute = (true_minute + target_offset) % 60
        target_rad = np.radians(target_minute * 6.0)   # minute -> degrees, matching hand_angles' convention
        true_rad = np.radians(true_minute * 6.0)

        # A REAL clock, SAME HOUR, showing the target minute, and the model's
        # own (unpatched) answer for it -- one extra generate() call per
        # trial image, reused across every layer/alpha/direction below. Same
        # hour is required so exact_transfer/hour_transfer aren't deflated
        # by an hour mismatch that has nothing to do with steering (see
        # find_real_image_with_hour_minute); often has no match in
        # data_balanced (~8 images/minute spread randomly over 12 hours), in
        # which case this trial's transfer columns are NaN rather than
        # silently falling back to a different-hour comparison.
        target_row = find_real_image_with_hour_minute(df, true_hour, target_minute, rng,
                                                        exclude_filename=row["filename"])
        target_base = None
        if target_row is not None:
            target_path = os.path.join(images_dir, target_row["filename"])
            target_base = run_baseline(adapter, image_features_owners, vision_method, target_path,
                                        max_new_tokens=max_new_tokens)

        image_mask = base["image_mask"]
        baseline_minute = base["pred_minute"] if base["parse_success"] else None

        for layer in layers:
            direction = directions[layer]
            w_sin, w_cos = direction["w_sin_full"], direction["w_cos_full"]

            h_image = base["hidden_states"][layer][image_mask].numpy().astype(np.float64)  # (n_img_tok, H)
            per_position_norms = np.linalg.norm(h_image, axis=-1)                           # (n_img_tok,)
            ref_norm = float(per_position_norms.mean())
            h_meanpool_before = h_image.mean(axis=0)                                        # (H,)
            angle_before = apply_saved_probe(direction, h_meanpool_before)
            minute_before_probe = angle_deg_to_minute(angle_before)

            v_target = w_sin * np.sin(target_rad) + w_cos * np.cos(target_rad)
            v_target_hat = v_target / np.linalg.norm(v_target)
            v_own = w_sin * np.sin(true_rad) + w_cos * np.cos(true_rad)
            v_own_hat = v_own / np.linalg.norm(v_own)
            dir_hats = {"probe_direction": v_target_hat, "random_direction": random_dir_hat,
                        "own_angle_direction": v_own_hat}

            for alpha in alphas:
                for dir_name in direction_names:
                    dir_hat = dir_hats[dir_name]
                    # own_angle_direction is scored against THIS clock's own true minute
                    # (steering toward what's already there should be close to a no-op),
                    # not the target -- everything else is scored against the target.
                    score_target = true_minute if dir_name == "own_angle_direction" else target_minute

                    # Same delta added to EVERY image-token position (mode="add" broadcasts
                    # it across all masked rows -- see _apply_add_patch; unchanged mechanics,
                    # already exercised for "add" mode by the prior final-token version and
                    # for multi-position masks by Experiment A's --verify'd "replace" patches).
                    delta_np = alpha * ref_norm * dir_hat  # (H,)

                    # Adding the SAME delta to every image-token position shifts their MEAN
                    # by exactly that delta -- no extra forward pass needed to check the probe
                    # readout moved, same trick the final-token version used.
                    h_meanpool_after = h_meanpool_before + delta_np
                    angle_after = apply_saved_probe(direction, h_meanpool_after)
                    minute_after_probe = angle_deg_to_minute(angle_after)
                    probe_moved, probe_shift = score_shift(minute_before_probe, minute_after_probe, score_target)

                    delta_t = torch.from_numpy(delta_np.astype(np.float32))
                    with patched(decoder_layers, layer, image_mask, delta_t, mode="add"):
                        ans = adapter.generate_answer(base["inputs"], max_new_tokens)
                    pred_hour, pred_minute, ok = parse_time_answer(ans)
                    moved, shift = score_shift(baseline_minute, pred_minute if ok else None, score_target)

                    # Transfer-to-target-baseline: does the steered answer match what the
                    # model ITSELF says (unpatched) about a real, same-hour clock showing
                    # the target minute -- rather than only whether it moved toward the
                    # abstract true target minute (see module docstring on why the latter
                    # alone is confounded). Always scored against the TARGET (not
                    # score_target), including for own_angle_direction rows -- that's what
                    # makes own_angle_direction's numbers a meaningful control: if it shows
                    # an ELEVATED transfer-to-target rate despite pointing at a different
                    # angle, the intervention is disruptive rather than semantically real.
                    minute_transfer_to_target = hour_transfer_to_target = exact_transfer_to_target = float("nan")
                    if ok and target_base is not None and target_base["parse_success"]:
                        minute_transfer_to_target = float(pred_minute == target_base["pred_minute"])
                        hour_transfer_to_target = float(pred_hour == target_base["pred_hour"])
                        exact_transfer_to_target = float(bool(minute_transfer_to_target) and bool(hour_transfer_to_target))

                    # Perturbation norm relative to the residual-stream norm AT THE IMAGE
                    # TOKENS (req: know steering isn't just wrecking activations). ||delta||
                    # is the same at every position (uniform add), but each position's OWN
                    # norm differs, so the RELATIVE perturbation isn't uniform even though the
                    # absolute one is -- report the spread, not just a single alpha-equals-ratio.
                    pert_norm = float(np.linalg.norm(delta_np))
                    rel_per_pos = pert_norm / np.clip(per_position_norms, 1e-8, None)

                    rows.append({
                        "file": row["filename"], "hour": true_hour, "true_minute": true_minute,
                        "target_minute": target_minute, "layer": layer, "num_layers": num_layers,
                        "relative_depth": relative_depth(layer, num_layers),
                        "direction": dir_name, "alpha": alpha,
                        "baseline_hour": base["pred_hour"], "baseline_minute": baseline_minute,
                        "patched_hour": pred_hour if ok else None, "patched_minute": pred_minute if ok else None,
                        "patched_answer": ans,
                        "moved_toward": moved, "shift_score": shift,
                        "answer_changed": (ans != base["raw_answer"]),
                        "probe_angle_before_deg": angle_before, "probe_angle_after_deg": angle_after,
                        "probe_moved_toward": probe_moved, "probe_shift_score": probe_shift,
                        "target_image_file": target_row["filename"] if target_row is not None else None,
                        "target_baseline_hour": target_base["pred_hour"] if target_base is not None else None,
                        "target_baseline_minute": target_base["pred_minute"] if target_base is not None else None,
                        "target_baseline_answer": target_base["raw_answer"] if target_base is not None else None,
                        "minute_transfer_to_target": minute_transfer_to_target,
                        "hour_transfer_to_target": hour_transfer_to_target,
                        "exact_transfer_to_target": exact_transfer_to_target,
                        "ref_norm_image_tokens": ref_norm, "pert_norm": pert_norm,
                        "pert_rel_norm_mean": float(rel_per_pos.mean()),
                        "pert_rel_norm_min": float(rel_per_pos.min()),
                        "pert_rel_norm_max": float(rel_per_pos.max()),
                    })

                    n_timed += 1
                    if n_timed == 6:
                        elapsed = time.perf_counter() - t_start
                        per_trial = elapsed / n_timed
                        print(f"\n[time estimate] ~{per_trial:.2f}s/trial -> ~{per_trial * n_total / 60:.1f} min total\n")

    trials_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    trials_df.to_csv(os.path.join(out_dir, "experiment_b_trials.csv"), index=False)
    n_target_usable = trials_df["target_image_file"].notna().sum()
    print(f"\nExperiment B: {len(trials_df)} trials over {len(trial_images)} images written to "
          f"'{out_dir}/experiment_b_trials.csv'. Same-hour target-minute baseline found for "
          f"{n_target_usable}/{len(trials_df)} trial rows ({trials_df['file'].nunique()} unique images).")
    return trials_df


def summarize_experiment_b(trials_df, out_dir):
    trials_df = trials_df.copy()
    for col in ("minute_transfer_to_target", "exact_transfer_to_target"):
        if col not in trials_df.columns:
            trials_df[col] = np.nan
    group_cols = ["layer", "direction", "alpha"] if "layer" in trials_df.columns else ["direction", "alpha"]
    agg_kwargs = dict(
        n=("answer_changed", "size"),
        n_parsed=("moved_toward", lambda s: s.notna().sum()),
        pct_moved_toward=("moved_toward", "mean"),
        mean_shift_score=("shift_score", "mean"),
        pct_answer_changed=("answer_changed", "mean"),
        pct_probe_moved_toward=("probe_moved_toward", "mean"),
        mean_probe_shift_score=("probe_shift_score", "mean"),
        n_target_usable=("minute_transfer_to_target", lambda s: s.notna().sum()),
        pct_minute_transfer_to_target=("minute_transfer_to_target", "mean"),
        pct_exact_transfer_to_target=("exact_transfer_to_target", "mean"),
    )
    if "pert_rel_norm_mean" in trials_df.columns:
        agg_kwargs["mean_pert_rel_norm"] = ("pert_rel_norm_mean", "mean")
    summary = trials_df.groupby(group_cols).agg(**agg_kwargs).reset_index()
    summary.to_csv(os.path.join(out_dir, "experiment_b_summary.csv"), index=False)
    return summary


def plot_experiment_b(summary_df, out_dir):
    """One row of 3 plots (stated-answer shift, probe-readout shift, %
    answer-changed) PER STEERED LAYER, so the per-layer picture (does
    steering work in the readout window and not outside it) is visible in
    one figure rather than needing one file per layer."""
    colors = {"probe_direction": "tab:blue", "random_direction": "tab:gray", "own_angle_direction": "tab:green"}
    layers = sorted(summary_df["layer"].unique()) if "layer" in summary_df.columns else [None]

    fig, axes = plt.subplots(len(layers), 3, figsize=(16, 4.2 * len(layers)), squeeze=False)
    for row_idx, layer in enumerate(layers):
        layer_df = summary_df[summary_df["layer"] == layer] if layer is not None else summary_df
        for dir_name, color in colors.items():
            s = layer_df[layer_df["direction"] == dir_name].sort_values("alpha")
            if len(s) == 0:
                continue
            axes[row_idx, 0].plot(s["alpha"], s["mean_shift_score"], marker="o", color=color, label=dir_name)
            axes[row_idx, 1].plot(s["alpha"], s["mean_probe_shift_score"], marker="s", color=color, label=dir_name)
            axes[row_idx, 2].plot(s["alpha"], s["pct_answer_changed"], marker="o", color=color, label=dir_name)

        axes[row_idx, 0].axhline(0, color="lightgray", linewidth=1)
        axes[row_idx, 1].axhline(0, color="lightgray", linewidth=1)
        layer_label = f"layer {layer}" if layer is not None else ""
        axes[row_idx, 0].set_title(f"{layer_label}: stated answer shift toward target")
        axes[row_idx, 1].set_title(f"{layer_label}: probe readout shift toward target")
        axes[row_idx, 2].set_title(f"{layer_label}: % answer changed")
        for ax in axes[row_idx]:
            ax.set_xlabel("alpha (x mean image-token residual norm)")
        axes[row_idx, 0].set_ylabel("mean shift score")
        axes[row_idx, 1].set_ylabel("mean shift score")
        axes[row_idx, 2].set_ylabel("fraction")

    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Experiment B: steering the probe direction at image-token positions, per layer")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "experiment_b_steering.png"), dpi=120)
    plt.close(fig)


def print_experiment_b_summary(summary_df):
    lines = ["=== EXPERIMENT B SUMMARY (image-token steering, per layer) ===", ""]
    lines.append("'toward true target' columns compare the steered answer to the ABSTRACT true target")
    lines.append("minute -- confounded the same way as Experiment A's old metric (see README): the model")
    lines.append("states the true minute correctly only ~2-3% of the time even unpatched. 'transfer to")
    lines.append("target baseline' compares instead to what the model ITSELF says (unpatched) about a REAL,")
    lines.append("SAME-HOUR clock that actually shows the target minute -- well-defined regardless of")
    lines.append("whether that answer is correct. Read the second set as primary.")
    lines.append("")
    lines.append("own_angle_direction steers toward THIS clock's own TRUE minute (not the target) -- since")
    lines.append("the representation already encodes something close to that, this should be close to a")
    lines.append("no-op. If it changes the answer or elevates 'transfer to target baseline' about as much as")
    lines.append("probe_direction does, the intervention is generically disruptive at this magnitude rather")
    lines.append("than doing anything specific to the minute -- compare it here, not just random_direction.")
    lines.append("")

    header = f"{'layer':>5} {'direction':<19}{'alpha':>7}{'n':>5} | {'toward true target':^30} | " \
             f"{'transfer to target baseline':^19} | {'probe (internal)':^21} | {'pert/resid':>10}"
    lines.append(header)
    subheader = f"{'':>5} {'':<19}{'':>7}{'':>5} | {'moved%':>9}{'shift':>8}{'chg%':>8}{'':>5} | " \
                f"{'minute%':>10}{'exact%':>9} | {'moved%':>10}{'shift':>11} | {'rel_norm':>10}"
    lines.append(subheader)

    sort_cols = ["layer", "direction", "alpha"] if "layer" in summary_df.columns else ["direction", "alpha"]
    for _, r in summary_df.sort_values(sort_cols).iterrows():
        layer_str = f"{int(r['layer']):>5} " if "layer" in summary_df.columns else f"{'':>5} "
        pert_str = f"{r['mean_pert_rel_norm']:>10.2f}" if "mean_pert_rel_norm" in summary_df.columns else f"{'':>10}"
        lines.append(
            f"{layer_str}{r['direction']:<19}{r['alpha']:>7.2f}{int(r['n']):>5} | "
            f"{r['pct_moved_toward']:>9.1%}{r['mean_shift_score']:>8.2f}{r['pct_answer_changed']:>8.1%}{'':>5} | "
            f"{r['pct_minute_transfer_to_target']:>10.1%}{r['pct_exact_transfer_to_target']:>9.1%} | "
            f"{r['pct_probe_moved_toward']:>10.1%}{r['mean_probe_shift_score']:>11.2f} | {pert_str}"
        )
    lines.append("")
    lines.append("'probe moved%'/'probe shift' (internal) verify the intervention internally: they measure")
    lines.append("whether the probe's OWN readout of the angle moved toward the target, independent of")
    lines.append("whether the stated answer did. If probe shift tracks alpha closely but the output columns")
    lines.append("stay flat, steering is moving the representation but the output ignores it -- not that")
    lines.append("steering failed. 'pert/resid' is the mean ratio of the perturbation's norm to each")
    lines.append("image-token position's OWN residual-stream norm (report req: confirm steering isn't just")
    lines.append("wrecking activations) -- by construction this tracks alpha closely; see")
    lines.append("pert_rel_norm_min/max in the trials CSV for how much it varies position-to-position.")

    text = "\n".join(lines)
    print("\n" + text)
    return text


# ---------------------------------------------------------------------------
# --verify: prove the interventions actually land, before trusting a null
# ---------------------------------------------------------------------------
#
# A clean null (patching/steering doesn't move the answer) and a silently
# broken hook (patching/steering doesn't run at all) look IDENTICAL from the
# experiment output alone. These checks don't touch the experiment logic --
# they just look, independently, at whether the tensors an intervention is
# supposed to write actually get written and actually propagate.

def verify_vision_swap(adapter, decoder_layers, image_features_owners, vision_method,
                        pairs, images_dir, max_new_tokens, rng):
    """For a few (A, B) pairs: confirm that patching B's image features
    into A's forward pass (a) actually writes a tensor different from A's
    own, (b) that difference propagates into the LLM's hidden states
    (checked at layer 0 = the merged embeddings, and layer 1 = one decoder
    block downstream -- if layer 1 is unaffected, the swap isn't reaching
    the part of the forward pass that matters), and, once, (c) that the
    model's ANSWER responds at all to a maximally aggressive version of the
    same intervention: zeroing the image features entirely, or replacing
    them with a random-noise image's. If (c) never changes the answer, the
    interception is not wired into the path used for generation -- a bug,
    not a finding, regardless of what (a) and (b) show.

    All patching goes through `apply_vision_replacement`, using whichever
    `vision_method` was determined to actually work for this model (see
    `determine_vision_interception_method`) -- NOT a forward hook on the
    vision tower's own module, which was the original, now-disproven
    approach (see module docstring)."""
    records = []
    lines = [f"--- Vision-encoder swap verification (method: {vision_method}) ---"]

    for pair_idx, (a, b) in enumerate(pairs):
        a_path = os.path.join(images_dir, a["filename"])
        b_path = os.path.join(images_dir, b["filename"])
        base_a = run_baseline(adapter, image_features_owners, vision_method, a_path,
                               max_new_tokens=max_new_tokens, require_vision_capture=True)
        base_b = run_baseline(adapter, image_features_owners, vision_method, b_path,
                               max_new_tokens=max_new_tokens, require_vision_capture=True)

        # (a) the swap writes a genuinely different tensor
        vision_diff = relative_l2_diff(base_a["vision_output"], base_b["vision_output"])

        # (b) propagation into the LLM: same A input, only the image features
        # differ, so ANY difference at layer 0/1 must come from the swap.
        replacement_b = vision_replacement_from(vision_method, base_b)
        with apply_vision_replacement(vision_method, decoder_layers, image_features_owners,
                                       base_a["image_mask"], replacement_b):
            hs_patched = forward_hidden_states(adapter.model, base_a["inputs"])
        hs_unpatched = base_a["hidden_states"]
        mask = base_a["image_mask"]
        layer0_diff = relative_l2_diff(hs_unpatched[0][mask], hs_patched[0][mask])
        layer1_diff = relative_l2_diff(hs_unpatched[1][mask], hs_patched[1][mask])

        record = {
            "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"], "vision_method": vision_method,
            "vision_output_rel_l2_diff": vision_diff,
            "layer0_rel_l2_diff_at_image_tokens": layer0_diff,
            "layer1_rel_l2_diff_at_image_tokens": layer1_diff,
        }
        lines.append(
            f"pair {pair_idx} ({a['filename']} <- {b['filename']}): "
            f"vision_output rel L2 diff={vision_diff:.4f}, "
            f"layer0 rel L2 diff at image tokens={layer0_diff:.4f}, "
            f"layer1 rel L2 diff at image tokens={layer1_diff:.4f}"
        )

        # (c) extreme tests -- only need to run once, cheap enough to do so.
        if pair_idx == 0:
            zero_vision = zeroed_vision_replacement(vision_method, base_a)
            with apply_vision_replacement(vision_method, decoder_layers, image_features_owners,
                                           base_a["image_mask"], zero_vision):
                ans_zero = adapter.generate_answer(base_a["inputs"], max_new_tokens)
            changed_zero = (ans_zero != base_a["raw_answer"])

            noise_image = make_random_noise_image(size=512, rng=rng)
            noise_base = run_baseline(adapter, image_features_owners, vision_method, noise_image,
                                       max_new_tokens=1, require_vision_capture=True)
            noise_replacement = vision_replacement_from(vision_method, noise_base)
            changed_noise, ans_noise = None, None
            if noise_replacement is not None and noise_replacement.shape == replacement_b.shape:
                with apply_vision_replacement(vision_method, decoder_layers, image_features_owners,
                                               base_a["image_mask"], noise_replacement):
                    ans_noise = adapter.generate_answer(base_a["inputs"], max_new_tokens)
                changed_noise = (ans_noise != base_a["raw_answer"])
            else:
                lines.append("  (skipped random-noise-image test: its image features shape didn't match "
                             "the clock images' -- unexpected, but doesn't affect the other checks)")

            record.update({
                "extreme_zero_answer": ans_zero, "extreme_zero_answer_changed": changed_zero,
                "extreme_noise_image_answer": ans_noise, "extreme_noise_image_answer_changed": changed_noise,
            })
            lines.append(f"  baseline answer: {base_a['raw_answer']!r}")
            lines.append(f"  EXTREME zero-vision answer: {ans_zero!r} (changed={changed_zero})")
            if changed_noise is not None:
                lines.append(f"  EXTREME random-noise-image answer: {ans_noise!r} (changed={changed_noise})")

        records.append(record)

    return lines, records


def verify_decoder_patch(adapter, decoder_layers, image_features_owners, vision_method,
                          pairs, images_dir, layers, max_new_tokens):
    """For a few (A, B) pairs and a few representative layers: report what
    fraction of sequence positions each position set actually covers (so a
    silently-empty 'image_tokens' mask -- meaning nothing was ever patched
    -- is impossible to miss), and confirm a patch's effect (1) is present
    at the patched layer itself and (2) propagates to a downstream layer
    AND the final layer -- not just tautologically true at the exact spot
    it was written."""
    records = []
    lines = ["--- Decoder-layer patch verification ---"]

    for pair_idx, (a, b) in enumerate(pairs):
        a_path = os.path.join(images_dir, a["filename"])
        b_path = os.path.join(images_dir, b["filename"])
        base_a = run_baseline(adapter, image_features_owners, vision_method, a_path,
                               max_new_tokens=max_new_tokens)
        base_b = run_baseline(adapter, image_features_owners, vision_method, b_path,
                               max_new_tokens=max_new_tokens)

        pos_masks = position_masks(base_a["image_mask"], base_a["seq_len"])
        final_hs_idx = len(base_a["hidden_states"]) - 1

        for pos_name, mask in pos_masks.items():
            n_positions = int(mask.sum())
            frac = n_positions / base_a["seq_len"]
            lines.append(f"pair {pair_idx}, position_set='{pos_name}': covers {n_positions}/"
                         f"{base_a['seq_len']} positions ({frac:.1%})")

            for layer in layers:
                with patched(decoder_layers, layer, mask, base_b["hidden_states"][layer], mode="replace"):
                    hs_patched = forward_hidden_states(adapter.model, base_a["inputs"])
                hs_unpatched = base_a["hidden_states"]

                diff_at_layer = relative_l2_diff(hs_unpatched[layer][mask], hs_patched[layer][mask])
                downstream_layer = min(layer + 1, final_hs_idx)
                diff_downstream = relative_l2_diff(hs_unpatched[downstream_layer][mask], hs_patched[downstream_layer][mask])
                diff_final = relative_l2_diff(hs_unpatched[final_hs_idx][mask], hs_patched[final_hs_idx][mask])

                records.append({
                    "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"],
                    "position_set": pos_name, "n_positions": n_positions, "frac_positions": frac,
                    "layer": layer, "rel_l2_diff_at_layer": diff_at_layer,
                    "downstream_layer": downstream_layer, "rel_l2_diff_downstream": diff_downstream,
                    "rel_l2_diff_final_layer": diff_final,
                })
                lines.append(
                    f"  layer {layer}: diff-at-patch={diff_at_layer:.4f}, "
                    f"diff-at-layer-{downstream_layer}={diff_downstream:.4f}, "
                    f"diff-at-final-layer={diff_final:.4f}"
                )

    return lines, records


def build_verification_verdict(vision_records, decoder_records):
    """A blunt PASS/FAIL read of the numbers above, so a reviewer doesn't
    have to eyeball a table of floats to know whether the hooks are
    actually wired in."""
    lines = ["--- Verdict ---"]
    problems = []

    vdf = pd.DataFrame(vision_records)
    if len(vdf) == 0:
        problems.append("No vision-swap verification data was collected at all -- if this is reached, "
                         "verify_vision_swap must have found zero usable pairs (a hard failure on the "
                         "capture itself raises directly, rather than leaving an empty result to reach "
                         "here silently -- see run_baseline's require_vision_capture).")
    else:
        if (vdf["layer1_rel_l2_diff_at_image_tokens"] < 1e-6).all():
            method = vdf["vision_method"].iloc[0] if "vision_method" in vdf.columns else "unknown"
            problems.append(f"The vision swap (method: {method}) produces a BIT-IDENTICAL layer-1 hidden "
                             "state in EVERY pair tested -- whatever this method is patching, it is not "
                             "the actual point of consumption in this transformers version.")
        extreme_cols = [c for c in ("extreme_zero_answer_changed", "extreme_noise_image_answer_changed")
                         if c in vdf.columns]
        any_extreme_changed = any(vdf[c].fillna(False).any() for c in extreme_cols) if extreme_cols else False
        if extreme_cols and not any_extreme_changed:
            problems.append("Neither the zero-vision nor the random-noise-image EXTREME test changed the "
                             "answer in ANY tested case. If even total replacement of the visual input "
                             "doesn't move the answer, the hook is not wired into generation -- any null "
                             "result from the real experiment is unverified until this is fixed.")

    ddf = pd.DataFrame(decoder_records)
    if len(ddf) == 0:
        problems.append("No decoder-patch verification data was collected at all.")
    else:
        empty_masks = sorted(ddf.loc[ddf["frac_positions"] == 0, "position_set"].unique().tolist())
        if empty_masks:
            problems.append(f"Position set(s) {empty_masks} covered ZERO sequence positions in at least one "
                             "pair -- that position set's sweep results are meaningless, since nothing was "
                             "ever actually patched.")
        if (ddf["rel_l2_diff_final_layer"] < 1e-6).all():
            problems.append("Every decoder-layer patch tested produced a BIT-IDENTICAL final-layer hidden "
                             "state -- patches are not propagating to the output at all, regardless of "
                             "layer or position set.")

    if problems:
        lines.append("FAIL -- do not trust the experiment's null result yet:")
        for p in problems:
            lines.append(f"  - {p}")
    else:
        lines.append("PASS: the vision swap and decoder patches both write genuinely different tensors that")
        lines.append("propagate downstream, and the extreme vision test(s) DO change the answer -- the")
        lines.append("intervention mechanism is demonstrably wired into the forward path. A null result from")
        lines.append("the main experiment reflects the model's actual behavior, not a broken hook.")

    return lines


def run_verification(adapter, decoder_layers, image_features_owners, vision_method,
                      df, images_dir, out_dir, n_verify_pairs=3, verify_layers=None,
                      min_gap=MIN_GAP_MINUTES, max_new_tokens=MAX_NEW_TOKENS, seed=SEED):
    if verify_layers is None:
        num_layers = len(decoder_layers)
        verify_layers = sorted(set([0, 1, num_layers // 2, num_layers]))
    rng = np.random.RandomState(seed)

    pairs = build_pairs(df, n_pairs=n_verify_pairs, min_gap=min_gap, seed=seed)
    if len(pairs) == 0:
        raise ValueError("No (A, B) pairs available for --verify -- check --min_gap against this dataset.")

    import transformers
    print(f"\n{'=' * 70}\nVERIFICATION: confirming interventions actually land, before trusting "
          f"any null result\n{'=' * 70}")
    print(f"model: {adapter.short_name} ({type(adapter).__name__})   transformers=={transformers.__version__}")
    print(f"Using {len(pairs)} pair(s), decoder layers {verify_layers}, vision method '{vision_method}'.")
    if vision_method == "get_image_features":
        print("NOTE: get_image_features interception has failed --verify on three prior rounds on this "
              "project's transformers version FOR QWEN (wrong field; wrong owner object; right owner+field "
              "but the merge step still didn't read the mutated value) -- being re-tried here because "
              "--vision_method explicitly requested it. For a DIFFERENT model this is its FIRST real-weight "
              "test of this mechanism (adapters.py's Gemma3Adapter/InternVLAdapter were built from reading "
              "source code, not run against real weights) -- treat a PASS as newly earned, not assumed, "
              "regardless of what worked for Qwen.")

    vision_lines, vision_records = verify_vision_swap(
        adapter, decoder_layers, image_features_owners, vision_method,
        pairs, images_dir, max_new_tokens, rng)
    decoder_lines, decoder_records = verify_decoder_patch(
        adapter, decoder_layers, image_features_owners, vision_method, pairs, images_dir,
        verify_layers, max_new_tokens)
    verdict_lines = build_verification_verdict(vision_records, decoder_records)

    all_lines = ["=== VERIFICATION REPORT ===", ""] + vision_lines + [""] + decoder_lines + [""] + verdict_lines
    text = "\n".join(all_lines)
    print("\n" + text)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "verification_report.txt"), "w") as f:
        f.write(text + "\n")
    pd.DataFrame(vision_records).to_csv(os.path.join(out_dir, "verification_vision_swap.csv"), index=False)
    pd.DataFrame(decoder_records).to_csv(os.path.join(out_dir, "verification_decoder_patch.csv"), index=False)
    print(f"\nVerification report saved to '{out_dir}/verification_report.txt' "
          f"(+ verification_vision_swap.csv / verification_decoder_patch.csv).")

    return text, vision_records, decoder_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_layers_cli(value, num_layers):
    """--layers / --steer_layers / --verify_layers CLI value -> a list of
    ABSOLUTE layer indices, once `num_layers` is known (i.e. after the
    adapter has loaded the model -- this can't happen at argparse parse
    time, unlike the old parse_layers_arg, since 'rel:...' needs to know how
    many decoder layers THIS model has). Accepts plain comma-separated
    absolute indices, 'rel:f1,f2,...' (relative depth in [0,1] -- see
    adapters.resolve_layers_arg), or 'readout_window' (this project's
    Qwen-3B-specific absolute-layer finding, kept literal rather than
    reinterpreted as relative). None passes through unchanged (callers use
    their own default range)."""
    if value is None:
        return None
    if value.strip() == "readout_window":
        return list(READOUT_WINDOW_LAYERS_A)
    return resolve_layers_arg(value, num_layers)


def parse_alphas_arg(value):
    return [float(x) for x in value.split(",") if x.strip() != ""]


def setup_model_and_vision(args):
    """Resolve --model_id/--adapter to a concrete adapter (see adapters.py),
    load it, and determine ONCE which mechanism actually intercepts the
    image representation for it -- shared by the normal experiment/--verify
    path and --analyze_only's optional baseline-recompute step, so neither
    has to duplicate this."""
    adapter = get_adapter(model_id=args.model_id, adapter_name=getattr(args, "adapter", None))
    adapter.load(args.model_id)   # None falls through to the adapter's own default_model_id (see adapters.py)
    decoder_layers = adapter.decoder_layers()
    image_features_owners, _owners_method_name = adapter.image_features_owners()
    print(f"Model: {adapter.short_name} ({type(adapter).__name__}). Found {len(decoder_layers)} decoder layers.")

    requested = args.vision_method
    if requested == "get_image_features" and not adapter.supports_get_image_features:
        print(f"WARNING: --vision_method get_image_features was requested, but {adapter.short_name}'s adapter "
              "doesn't support it (image_features_owners() returned none -- see adapters.py's module "
              "docstring on why that's fine in general). Falling back to layer0_embed, the trusted default "
              "for every model.")
        requested = "layer0_embed"
    vision_method = determine_vision_interception_method(requested=requested)
    return adapter, decoder_layers, image_features_owners, vision_method


def run_analyze_only(args):
    """--analyze_only: skip the sweep, load an existing experiment_a_trials.csv
    from --out_dir, backfill baseline-answer columns if needed and
    requested (--recompute_baselines), and (re)compute + print + save the
    transfer-to-baseline summary. Does NOT load the model unless
    --recompute_baselines is actually needed -- a pure re-analysis of an
    already-complete CSV is a CPU-only, seconds-long operation."""
    trials_path = os.path.join(args.out_dir, "experiment_a_trials.csv")
    if not os.path.exists(trials_path):
        raise FileNotFoundError(f"--analyze_only: '{trials_path}' not found -- run the sweep "
                                 "(without --analyze_only) at least once first.")
    trials_a = pd.read_csv(trials_path)
    print(f"--analyze_only: loaded {len(trials_a)} trial(s) from '{trials_path}'.")

    has_baselines = ("a_baseline_hour" in trials_a.columns and not trials_a["a_baseline_hour"].isna().all() and
                      "b_baseline_hour" in trials_a.columns and not trials_a["b_baseline_hour"].isna().all())
    if not has_baselines:
        if not args.recompute_baselines:
            raise ValueError(
                f"'{trials_path}' is missing baseline-answer columns (a_baseline_hour/b_baseline_hour) -- "
                "this looks like an older run. Pass --recompute_baselines to backfill them cheaply (one "
                "generate() call per unique image, no patching), or re-run the full sweep.")
        adapter, decoder_layers, image_features_owners, vision_method = setup_model_and_vision(args)
        unique_files = pd.concat([trials_a["a_file"], trials_a["b_file"]]).dropna().unique().tolist()
        baselines_df = recompute_baselines_for_files(
            adapter, image_features_owners, vision_method, args.images_dir,
            unique_files, max_new_tokens=args.max_new_tokens)
        os.makedirs(args.out_dir, exist_ok=True)
        baselines_df.to_csv(os.path.join(args.out_dir, "experiment_a_baselines.csv"), index=False)
        trials_a = backfill_baselines(trials_a, baselines_df)
        trials_a.to_csv(trials_path, index=False)
        print(f"Backfilled and re-saved '{trials_path}' -- future --analyze_only runs on this out_dir "
              "won't need --recompute_baselines again.")

    summary_a = summarize_experiment_a(trials_a, args.out_dir)
    plot_experiment_a(summary_a, args.out_dir)
    # vision_method isn't necessarily known here (no model loaded if baselines were
    # already present) -- read it back from the vision-ceiling rows' own record if
    # possible, else report "unknown" rather than guessing.
    vision_method_seen = "unknown (not re-determined in --analyze_only; see run's own console log)"
    text_a = print_experiment_a_summary(summary_a, trials_a, vision_method_seen)
    transfer_summary_a = summarize_transfer_a(trials_a, args.out_dir)
    agreement_stats_a = baseline_agreement_stats(trials_a)
    text_transfer_a = print_transfer_summary_a(transfer_summary_a, agreement_stats_a, trials_a)
    per_layer_summary_a = summarize_per_layer_transfer_a(trials_a, args.out_dir)
    # num_layers isn't known here without loading the model -- read it back from the
    # trials CSV itself if this run saved it (every run since the multi-model adapter
    # rewrite does); NUM_DECODER_LAYERS_HINT (Qwen-3B-specific, see its docstring) is
    # only a fallback for a CSV old enough to predate that column.
    if "num_layers" in trials_a.columns and trials_a["num_layers"].notna().any():
        num_layers_seen = int(trials_a["num_layers"].dropna().iloc[0])
    else:
        num_layers_seen = NUM_DECODER_LAYERS_HINT
        print(f"NOTE: this trials.csv predates saving num_layers -- assuming {NUM_DECODER_LAYERS_HINT} "
              "(NUM_DECODER_LAYERS_HINT, Qwen-3B-specific) for the final-layer mechanics-artifact caveat. "
              "If this run was actually a different model, that caveat may be misattributed to the wrong "
              "layer -- re-run the sweep to get an accurate num_layers column.")
    text_per_layer_a = print_per_layer_transfer_table_a(per_layer_summary_a, num_layers=num_layers_seen)
    with open(os.path.join(args.out_dir, "experiment_a_summary.txt"), "w") as f:
        f.write(text_a + "\n\n" + text_transfer_a + "\n\n" + text_per_layer_a + "\n")
    print(f"\n--analyze_only done. Summaries (re)written to '{args.out_dir}/'.")


def main():
    parser = argparse.ArgumentParser(
        description="Causal interventions (activation patching + steering) on the clock-reading failure.")
    parser.add_argument("--experiment", choices=["a", "b", "both", "none"], default="both",
                         help="'none' runs no experiment -- useful with --verify to just check the "
                              "intervention mechanism without committing to a full sweep")
    parser.add_argument("--verify", action="store_true",
                         help="before running any selected experiment, verify that the vision-encoder "
                              "swap and decoder-layer patches actually write different tensors and that "
                              "the difference propagates downstream (see module docstring) -- prints and "
                              "saves a verification report into --out_dir. Combine with --experiment none "
                              "for a quick standalone check.")
    parser.add_argument("--verify_pairs", type=int, default=3,
                         help="--verify: number of (A, B) pairs to check (kept small -- this is a sanity "
                              "check, not a statistical sweep)")
    parser.add_argument("--verify_layers", type=str, default=None,
                         help="--verify: comma-separated decoder layers to check, or 'rel:f1,f2,...' for "
                              "relative depth in [0,1] (resolved once the model's layer count is known -- "
                              "see resolve_layers_cli); default: 0, 1, a middle layer, and the final layer")
    parser.add_argument("--vision_method", choices=["auto", "get_image_features", "layer0_embed"], default="auto",
                         help="which mechanism intercepts the image representation for the vision-ceiling "
                              "condition and its --verify checks. 'layer0_embed' patches hidden_states[0] at "
                              "the image-token positions (the same, already-verified mechanism the decoder-"
                              "patch sweep uses). 'get_image_features' monkey-patches that method instead -- "
                              "has failed --verify on three prior rounds on this project's transformers "
                              "version; not recommended without re-verifying. 'auto' (default) currently "
                              "resolves to 'layer0_embed' for that reason -- see determine_vision_interception_method.")
    parser.add_argument("--data_csv", type=str, default=DATA_CSV)
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--direction_dir", type=str, default=DIRECTION_DIR,
                         help="Experiment B: directory containing the per-layer .npz direction files saved "
                              "by `python probe.py --stage direction --direction_layers ...` "
                              "(see direction_path_for_layer)")
    parser.add_argument("--direction_representation", type=str, default=DIRECTION_REPRESENTATION,
                         choices=["hidden_last", "hidden_meanpool", "vision_encoder"],
                         help="Experiment B: which representation the steering directions were fit on -- "
                              "must be 'hidden_meanpool' (the default) since Experiment B steers "
                              "IMAGE-TOKEN positions and the probe-readout check reads off their "
                              "mean-pooled vector; only change this if you know what you're doing.")
    parser.add_argument("--direction_hand", type=str, default=DIRECTION_HAND, choices=["minute", "hour"],
                         help="Experiment B: which hand's angle the steering directions were fit for")
    parser.add_argument("--steer_layers", type=str, default=None,
                         help=f"Experiment B: comma-separated layers to steer at, or 'rel:f1,f2,...' for "
                              f"relative depth (default: {READOUT_WINDOW_LAYERS_B}, the readout window found "
                              "by Experiment A's transfer analysis -- see module docstring). Also accepts "
                              "'readout_window' for Experiment A's (longer) window list.")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR,
                         help="results are written to <out_dir>/<model_short_name>/ (see "
                              "adapters.output_dir_for) -- different models never overwrite each other's results")
    parser.add_argument("--model_id", type=str, default=None,
                         help="Hugging Face model id. Default: the resolved adapter's own default (Qwen2.5-VL-3B "
                              "if neither --model_id nor --adapter is given -- this project's original model).")
    parser.add_argument("--adapter", type=str, default=None,
                         help="which model-family adapter to use (see adapters.py) -- inferred from "
                              "--model_id if omitted; only needed to force a specific adapter for an "
                              "unrecognized --model_id.")
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS,
                         help="Experiment A: number of (A, B) pairs. Experiment B: number of steered images.")
    parser.add_argument("--max_pairs", type=int, default=None,
                         help="cap on pairs/images actually used, for a quick smoke test (overrides --n_pairs downward)")
    parser.add_argument("--layers", type=str, default=None,
                         help="Experiment A: comma-separated layers to sweep, e.g. '0,1,21,36'; 'rel:f1,f2,...' "
                              "for relative depth in [0,1] (resolved once the model's layer count is known -- "
                              "same window works across models with different depths); or 'readout_window' "
                              f"for the layers 16-24 window shorthand ({READOUT_WINDOW_LAYERS_A}, this "
                              "project's Qwen-3B-specific finding) (default: every layer 0..num_layers)")
    parser.add_argument("--min_gap", type=int, default=MIN_GAP_MINUTES,
                         help="Experiment A: minimum circular minute gap between A and B")
    parser.add_argument("--target_offset", type=int, default=TARGET_OFFSET_MINUTES,
                         help="Experiment B: steering target = (true_minute + this) %% 60")
    parser.add_argument("--alphas", type=parse_alphas_arg, default=None,
                         help=f"Experiment B: comma-separated alpha values (default: {DEFAULT_ALPHAS})")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--analyze_only", action="store_true",
                         help="skip the sweep entirely: load an existing experiment_a_trials.csv from "
                              "--out_dir and (re)compute the transfer-to-baseline summary from it. Ignores "
                              "--verify/--experiment. Combine with --recompute_baselines if that CSV "
                              "predates saving baseline answers (an older run).")
    parser.add_argument("--recompute_baselines", action="store_true",
                         help="with --analyze_only: if the loaded trials.csv is missing baseline-answer "
                              "columns, recompute them cheaply (one generate() call per unique image, no "
                              "patching -- see recompute_baselines_for_files) instead of redoing the full "
                              "sweep. Needs the model loaded, unlike a plain --analyze_only.")
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)

    if args.analyze_only:
        run_analyze_only(args)
        return

    adapter, decoder_layers, image_features_owners, vision_method = setup_model_and_vision(args)
    num_layers = len(decoder_layers)
    out_dir = output_dir_for(args.out_dir, adapter)   # outputs/<short_name>/... -- never collides across models
    layers = resolve_layers_cli(args.layers, num_layers)
    verify_layers = resolve_layers_cli(args.verify_layers, num_layers)
    steer_layers = resolve_layers_cli(args.steer_layers, num_layers) or list(READOUT_WINDOW_LAYERS_B)

    os.makedirs(out_dir, exist_ok=True)

    if args.verify:
        run_verification(
            adapter, decoder_layers, image_features_owners, vision_method,
            df, args.images_dir, out_dir, n_verify_pairs=args.verify_pairs,
            verify_layers=verify_layers, min_gap=args.min_gap,
            max_new_tokens=args.max_new_tokens, seed=args.seed)

    if args.experiment in ("a", "both"):
        trials_a = run_experiment_a(
            adapter, decoder_layers, image_features_owners, vision_method,
            df, args.images_dir, out_dir, n_pairs=args.n_pairs, max_pairs=args.max_pairs,
            min_gap=args.min_gap, layers=layers, max_new_tokens=args.max_new_tokens, seed=args.seed)
        summary_a = summarize_experiment_a(trials_a, out_dir)
        plot_experiment_a(summary_a, out_dir)
        text_a = print_experiment_a_summary(summary_a, trials_a, vision_method)
        transfer_summary_a = summarize_transfer_a(trials_a, out_dir)
        agreement_stats_a = baseline_agreement_stats(trials_a)
        text_transfer_a = print_transfer_summary_a(transfer_summary_a, agreement_stats_a, trials_a)
        per_layer_summary_a = summarize_per_layer_transfer_a(trials_a, out_dir)
        text_per_layer_a = print_per_layer_transfer_table_a(per_layer_summary_a, num_layers=num_layers)
        with open(os.path.join(out_dir, "experiment_a_summary.txt"), "w") as f:
            f.write(text_a + "\n\n" + text_transfer_a + "\n\n" + text_per_layer_a + "\n")

    if args.experiment in ("b", "both"):
        try:
            directions = load_directions_for_layers(
                args.direction_dir, args.direction_representation, args.direction_hand, steer_layers)
        except FileNotFoundError as e:
            print(f"\nSkipping Experiment B: {e}")
            directions = None
        if directions is not None:
            trials_b = run_experiment_b(
                adapter, decoder_layers, image_features_owners, vision_method,
                directions, df, args.images_dir, out_dir, n_trials=args.n_pairs, max_pairs=args.max_pairs,
                target_offset=args.target_offset, alphas=args.alphas,
                max_new_tokens=args.max_new_tokens, seed=args.seed)
            summary_b = summarize_experiment_b(trials_b, out_dir)
            plot_experiment_b(summary_b, out_dir)
            text_b = print_experiment_b_summary(summary_b)
            with open(os.path.join(out_dir, "experiment_b_summary.txt"), "w") as f:
                f.write(text_b + "\n")

    print(f"\nAll outputs written to '{out_dir}/'.")


if __name__ == "__main__":
    main()
