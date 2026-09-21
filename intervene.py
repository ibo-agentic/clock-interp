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
    Add a scaled version of the fitted probe's weight direction to the
    residual stream at its layer, and see whether that pushes the stated
    minute toward a target -- and, independently, whether it moves the
    PROBE's own readout of the angle (so we can tell "steering failed to
    move the representation" apart from "the representation moved but the
    output ignored it").

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

Reuses model loading, decoder-layer/image-token-id finding, and answer
parsing from probe.py / eval_behavior.py rather than re-implementing them
(see probe.py's `load_model`, `find_image_token_id`, `load_probe_direction`
/ `apply_saved_probe` for Experiment B's direction). The vision-side
interception is NOT reused from probe.py, for the reason above.

Usage:
    # 1. Make sure Experiment B has a direction to load (Experiment A does
    #    not need this):
    python probe.py --stage direction

    # 2. Smoke-test on a couple of pairs and a handful of layers first:
    python intervene.py --max_pairs 1 --layers 0,1,21,36

    # 3. Then the full sweep:
    python intervene.py

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

from eval_behavior import MODEL_ID, PROMPT, SEED, parse_time_answer
from probe import apply_saved_probe, find_image_token_id, load_model, load_probe_direction

DATA_CSV = "data_balanced/data.csv"
IMAGES_DIR = "data_balanced"
DIRECTION_PATH = "probe_output/probe_results/probe_direction_hidden_last_minute.npz"
OUT_DIR = "intervene_output"

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

def find_decoder_layers(model):
    """Locate the LLM's stack of transformer decoder blocks by scanning the
    module tree for a ModuleList whose children look like decoder layers --
    same reasoning as probe.py's `find_vision_module`: robust to the exact
    attribute path (e.g. `model.model.layers` vs `model.language_model.layers`)
    moving between transformers versions, since the class naming has stayed
    recognizable. The first match is what we want (an outer ModuleList
    won't itself contain other ModuleLists of decoder layers in practice)."""
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            if "DecoderLayer" in type(module[0]).__name__:
                return module
    raise AttributeError("Could not find the LLM decoder layer stack in this model's module tree.")


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


def find_all_image_features_owners(model):
    """Find EVERY distinct object in the model's hierarchy that defines a
    callable `get_image_features` -- not just the first one found.

    A prior version of this fix picked ONE owner (top-level model, else
    model.model) and monkey-patched only that. A --verify run proved that
    was the wrong one: `Qwen2_5_VLForConditionalGeneration` (top-level) DOES
    define/inherit `get_image_features`, but the call that actually matters
    happens inside `Qwen2_5_VLModel.forward` (model.model) as
    `self.get_image_features(...)` -- a COMPLETELY SEPARATE Python object
    with its own, independently-resolved attributes. Patching the outer
    object has zero effect on the inner one. Since we can't be sure in
    advance which object's method is the one that actually executes for a
    given transformers version, we patch ALL of them (see
    `image_features_patched`) and rely on --verify's hard-failure check to
    prove at least one patch fired for real.
    """
    seen_ids = set()
    owners = []

    def add(name, obj):
        if obj is None or id(obj) in seen_ids:
            return
        if callable(getattr(obj, "get_image_features", None)):
            owners.append((name, obj))
            seen_ids.add(id(obj))

    add("model", model)
    add("model.model", getattr(model, "model", None))
    for name, module in model.named_modules():
        add(f"model.{name}", module)

    if not owners:
        raise AttributeError("Could not find any object with a `get_image_features` method "
                              "anywhere in this model's hierarchy.")
    print(f"Found get_image_features() on: {[name for name, _ in owners]}")
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

def build_inputs(processor, image_path_or_image, prompt=PROMPT):
    """`image_path_or_image` is normally a path (all the normal experiment
    code passes one). --verify's random-noise-image check also passes an
    already-in-memory PIL.Image directly (no need to round-trip it through
    disk just to satisfy this function)."""
    if isinstance(image_path_or_image, str):
        image = Image.open(image_path_or_image).convert("RGB")
    else:
        image = image_path_or_image.convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                              {"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[chat_text], images=[image], return_tensors="pt")


@torch.no_grad()
def generate_answer(model, processor, inputs, max_new_tokens=MAX_NEW_TOKENS):
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


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
def run_baseline(model, processor, image_path, image_token_id, image_features_owners, vision_method,
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
                        "embeds_fallback", hidden_states[0] at the
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
    inputs = build_inputs(processor, image_path, prompt).to(model.device)

    if vision_method == "get_image_features":
        holder = {}
        with image_features_patched(image_features_owners, capture_holder=holder):
            outputs = model(**inputs, output_hidden_states=True)
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
        outputs = model(**inputs, output_hidden_states=True)
        vision_output = None  # filled in below once hidden_states[0] exists

    hidden_states = tuple(h[0].float().cpu() for h in outputs.hidden_states)
    input_ids = inputs["input_ids"][0]
    image_mask = (input_ids == image_token_id).cpu()
    del outputs

    if vision_method == "embeds_fallback":
        vision_output = hidden_states[0][image_mask].clone()

    raw_answer = generate_answer(model, processor, inputs, max_new_tokens)
    pred_hour, pred_minute, ok = parse_time_answer(raw_answer)

    return {
        "inputs": inputs, "hidden_states": hidden_states, "image_mask": image_mask,
        "seq_len": int(input_ids.shape[0]), "vision_output": vision_output,
        "raw_answer": raw_answer, "pred_hour": pred_hour, "pred_minute": pred_minute,
        "parse_success": ok,
    }


def determine_vision_interception_method(model, processor, image_features_owners, sample_image_path):
    """Probe, ONCE, which mechanism actually intercepts the image
    representation the LLM reads for THIS model: try monkey-patching
    get_image_features() first (the most surgical option -- it targets
    exactly the image representation, before any merge-level processing).
    If that doesn't fire during a real forward pass (as happened when the
    wrong owner was patched), fall back to patching hidden_states[0] -- the
    fully merged embeddings, at the image-token positions, right before the
    decoder stack runs. That fallback targets the SAME tensor the LLM
    actually consumes regardless of where or whether get_image_features is
    involved at all, and reuses the already-verified decoder-layer-0 patch
    mechanism (`patched`) rather than introducing new untested code.

    Returns "get_image_features" or "embeds_fallback". Never returns a
    method that doesn't work: the fallback is a plain forward hook via
    PyTorch's own hook registration (not a monkey-patched Python method),
    so unlike get_image_features() it cannot silently fail to fire -- if
    the model runs at all, layer-0 patching runs.
    """
    inputs = build_inputs(processor, sample_image_path).to(model.device)
    holder = {}
    with image_features_patched(image_features_owners, capture_holder=holder):
        model(**inputs, output_hidden_states=False)

    if holder.get("fired"):
        print(f"Vision interception: get_image_features() on '{holder['fired_by']}' fires correctly "
              "-- using it directly.")
        return "get_image_features"

    print(f"Vision interception: get_image_features() did not fire on any of "
          f"{[name for name, _ in image_features_owners]} during a real forward pass. Falling back to "
          "patching hidden_states[0] (the merged embeddings) at the image-token positions -- this "
          "targets the same tensor the decoder stack actually consumes, independent of where "
          "get_image_features lives.")
    return "embeds_fallback"


def vision_replacement_from(vision_method, source_base):
    """Pull the correctly-shaped replacement tensor for `apply_vision_replacement`
    out of a `run_baseline(...)` result -- shape/meaning depends on
    `vision_method`: for "get_image_features", (n_image_tokens, H); for
    "embeds_fallback", the FULL (seq_len, H) hidden_states[0] (only its
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


# ---------------------------------------------------------------------------
# Experiment A: the sweep
# ---------------------------------------------------------------------------

def run_experiment_a(model, processor, decoder_layers, image_features_owners, vision_method, image_token_id,
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

        base_a = run_baseline(model, processor, a_path, image_token_id, image_features_owners, vision_method,
                               max_new_tokens=max_new_tokens)
        base_b = run_baseline(model, processor, b_path, image_token_id, image_features_owners, vision_method,
                               max_new_tokens=max_new_tokens)

        same_row = find_same_minute_partner(df, a, exclude=already_used, rng=rng)
        base_same = None
        if same_row is not None:
            same_path = os.path.join(images_dir, same_row["filename"])
            base_same = run_baseline(model, processor, same_path, image_token_id, image_features_owners,
                                      vision_method, max_new_tokens=max_new_tokens)

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
                    ans = generate_answer(model, processor, base_a["inputs"], max_new_tokens)
                pred_hour, pred_minute, ok = parse_time_answer(ans)
                moved, shift = score_shift(baseline_a_minute, pred_minute if ok else None, int(b["minute"]))
                rows.append({
                    "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"], "condition": "real_b",
                    "layer": layer, "position_set": pos_name, "a_true_minute": int(a["minute"]),
                    "b_true_minute": int(b["minute"]), "source_minute": int(b["minute"]),
                    "baseline_minute": baseline_a_minute, "patched_minute": pred_minute if ok else None,
                    "moved_toward": moved, "shift_score": shift,
                    "answer_changed": (ans != base_a["raw_answer"]),
                })

                # --- control (a): same-minute patch -- should change nothing ---
                if base_same is not None:
                    with patched(decoder_layers, layer, mask, base_same["hidden_states"][layer], mode="replace"):
                        ans_same = generate_answer(model, processor, base_a["inputs"], max_new_tokens)
                    pred_hour_s, pred_minute_s, ok_s = parse_time_answer(ans_same)
                    moved_s, shift_s = score_shift(baseline_a_minute, pred_minute_s if ok_s else None, int(a["minute"]))
                    rows.append({
                        "pair": pair_idx, "a_file": a["filename"], "b_file": same_row["filename"],
                        "condition": "same_minute", "layer": layer, "position_set": pos_name,
                        "a_true_minute": int(a["minute"]), "b_true_minute": int(same_row["minute"]),
                        "source_minute": int(a["minute"]),
                        "baseline_minute": baseline_a_minute, "patched_minute": pred_minute_s if ok_s else None,
                        "moved_toward": moved_s, "shift_score": shift_s,
                        "answer_changed": (ans_same != base_a["raw_answer"]),
                    })

                # --- control (b): matched-norm random noise ---
                noise_vals = make_matched_noise(base_a["hidden_states"][layer], rng)
                with patched(decoder_layers, layer, mask, noise_vals, mode="replace"):
                    ans_noise = generate_answer(model, processor, base_a["inputs"], max_new_tokens)
                pred_hour_n, pred_minute_n, ok_n = parse_time_answer(ans_noise)
                moved_n, shift_n = score_shift(baseline_a_minute, pred_minute_n if ok_n else None, int(b["minute"]))
                rows.append({
                    "pair": pair_idx, "a_file": a["filename"], "b_file": None, "condition": "noise",
                    "layer": layer, "position_set": pos_name, "a_true_minute": int(a["minute"]),
                    "b_true_minute": int(b["minute"]), "source_minute": int(b["minute"]),
                    "baseline_minute": baseline_a_minute, "patched_minute": pred_minute_n if ok_n else None,
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
                ans_v = generate_answer(model, processor, base_a["inputs"], max_new_tokens)
            pred_hour_v, pred_minute_v, ok_v = parse_time_answer(ans_v)
            moved_v, shift_v = score_shift(baseline_a_minute, pred_minute_v if ok_v else None, int(b["minute"]))
            rows.append({
                "pair": pair_idx, "a_file": a["filename"], "b_file": b["filename"], "condition": "real_b",
                "layer": -1, "position_set": "vision_encoder", "a_true_minute": int(a["minute"]),
                "b_true_minute": int(b["minute"]), "source_minute": int(b["minute"]),
                "baseline_minute": baseline_a_minute, "patched_minute": pred_minute_v if ok_v else None,
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


def print_experiment_a_summary(summary_df, trials_df):
    """A compact text table: for each position set, the BEST layer's
    real_b result next to the same layer's controls."""
    lines = ["=== EXPERIMENT A SUMMARY (activation patching) ===", ""]
    n_pairs = trials_df["pair"].nunique()
    lines.append(f"n pairs: {n_pairs}")
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
        lines.append(f"vision-encoder ceiling (whole visual representation swapped): "
                      f"moved toward B {c['pct_moved_toward']:.1%}, mean shift {c['mean_shift_score']:.2f}, "
                      f"answer changed {c['pct_answer_changed']:.1%}")

    lines.append("")
    lines.append("Read 'moved%' RELATIVE TO ITS OWN same-min/noise columns, not against an assumed 50%:")
    lines.append("when A and B's minutes are far apart (up to 30, the maximum on a 60-minute circle),")
    lines.append("baseline sits near the worst possible starting point, so almost ANY perturbation --")
    lines.append("including pure noise -- has good odds of looking 'closer' by chance. The noise and")
    lines.append("same-minute columns above already reflect that same geometry, so if real_b's numbers")
    lines.append("stay close to noise's at the same layer/position (not clearly higher), and 'chg%' stays")
    lines.append("low across the WHOLE sweep including the vision-encoder ceiling, THAT null result is")
    lines.append("the finding: the angle information is present but the model's answer doesn't use it.")

    text = "\n".join(lines)
    print("\n" + text)
    return text


# ---------------------------------------------------------------------------
# Experiment B: steering along the probe direction
# ---------------------------------------------------------------------------

def run_experiment_b(model, processor, decoder_layers, image_token_id, image_features_owners, vision_method,
                      direction, df, images_dir, out_dir, n_trials=N_PAIRS, max_pairs=None,
                      target_offset=TARGET_OFFSET_MINUTES, alphas=None,
                      max_new_tokens=MAX_NEW_TOKENS, seed=SEED):
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if max_pairs is not None:
        n_trials = min(n_trials, max_pairs)

    layer = direction["layer"]
    w_sin, w_cos = direction["w_sin_full"], direction["w_cos_full"]
    print(f"Experiment B: steering at layer {layer} (from '{direction['representation']}'/"
          f"'{direction['hand']}', train R^2={direction['r2_train']:.3f}).")

    rng = np.random.RandomState(seed)
    trial_images = df.sample(frac=1, random_state=seed).reset_index(drop=True).head(n_trials)

    # One fixed random direction, reused for every trial/alpha, so the
    # probe-direction vs. random-direction comparison is apples to apples.
    random_dir = rng.standard_normal(w_sin.shape[0])
    random_dir_hat = random_dir / np.linalg.norm(random_dir)

    n_total = len(trial_images) * len(alphas) * 2
    print(f"Experiment B: {len(trial_images)} image(s) x {len(alphas)} alpha(s) x 2 directions "
          f"(probe / random) = {n_total} generate() calls.")

    rows = []
    t_start = time.perf_counter()
    n_timed = 0

    for _, row in tqdm(trial_images.iterrows(), total=len(trial_images), desc="Experiment B images"):
        path = os.path.join(images_dir, row["filename"])
        base = run_baseline(model, processor, path, image_token_id, image_features_owners, vision_method,
                             max_new_tokens=max_new_tokens)

        true_minute = int(row["minute"])
        target_minute = (true_minute + target_offset) % 60
        target_rad = np.radians(target_minute * 6.0)  # minute -> degrees, matching hand_angles' convention

        h_final = base["hidden_states"][layer][-1].numpy().astype(np.float64)  # the steered position
        h_norm = float(np.linalg.norm(h_final))
        angle_before = apply_saved_probe(direction, h_final)
        minute_before_probe = angle_deg_to_minute(angle_before)

        v = w_sin * np.sin(target_rad) + w_cos * np.cos(target_rad)
        v_hat = v / np.linalg.norm(v)

        final_mask = torch.zeros(base["seq_len"], dtype=torch.bool)
        final_mask[-1] = True
        baseline_minute = base["pred_minute"] if base["parse_success"] else None

        for alpha in alphas:
            for dir_name, dir_hat in (("probe_direction", v_hat), ("random_direction", random_dir_hat)):
                delta_np = alpha * h_norm * dir_hat
                h_after = h_final + delta_np
                angle_after = apply_saved_probe(direction, h_after)
                minute_after_probe = angle_deg_to_minute(angle_after)
                probe_moved, probe_shift = score_shift(minute_before_probe, minute_after_probe, target_minute)

                delta_t = torch.from_numpy(delta_np.astype(np.float32))
                with patched(decoder_layers, layer, final_mask, delta_t, mode="add"):
                    ans = generate_answer(model, processor, base["inputs"], max_new_tokens)
                pred_hour, pred_minute, ok = parse_time_answer(ans)
                moved, shift = score_shift(baseline_minute, pred_minute if ok else None, target_minute)

                rows.append({
                    "file": row["filename"], "true_minute": true_minute, "target_minute": target_minute,
                    "direction": dir_name, "alpha": alpha, "baseline_minute": baseline_minute,
                    "patched_minute": pred_minute if ok else None, "moved_toward": moved, "shift_score": shift,
                    "answer_changed": (ans != base["raw_answer"]),
                    "probe_angle_before_deg": angle_before, "probe_angle_after_deg": angle_after,
                    "probe_moved_toward": probe_moved, "probe_shift_score": probe_shift,
                })

                n_timed += 1
                if n_timed == 6:
                    elapsed = time.perf_counter() - t_start
                    per_trial = elapsed / n_timed
                    print(f"\n[time estimate] ~{per_trial:.2f}s/trial -> ~{per_trial * n_total / 60:.1f} min total\n")

    trials_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    trials_df.to_csv(os.path.join(out_dir, "experiment_b_trials.csv"), index=False)
    print(f"\nExperiment B: {len(trials_df)} trials over {len(trial_images)} images written to "
          f"'{out_dir}/experiment_b_trials.csv'.")
    return trials_df


def summarize_experiment_b(trials_df, out_dir):
    summary = trials_df.groupby(["direction", "alpha"]).agg(
        n=("answer_changed", "size"),
        n_parsed=("moved_toward", lambda s: s.notna().sum()),
        pct_moved_toward=("moved_toward", "mean"),
        mean_shift_score=("shift_score", "mean"),
        pct_answer_changed=("answer_changed", "mean"),
        pct_probe_moved_toward=("probe_moved_toward", "mean"),
        mean_probe_shift_score=("probe_shift_score", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(out_dir, "experiment_b_summary.csv"), index=False)
    return summary


def plot_experiment_b(summary_df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"probe_direction": "tab:blue", "random_direction": "tab:gray"}

    for dir_name, color in colors.items():
        s = summary_df[summary_df["direction"] == dir_name].sort_values("alpha")
        axes[0].plot(s["alpha"], s["mean_shift_score"], marker="o", color=color, label=dir_name)
        axes[1].plot(s["alpha"], s["mean_probe_shift_score"], marker="s", color=color, label=dir_name)
        axes[2].plot(s["alpha"], s["pct_answer_changed"], marker="o", color=color, label=dir_name)

    axes[0].axhline(0, color="lightgray", linewidth=1)
    axes[0].set_title("Stated answer: shift toward target")
    axes[1].axhline(0, color="lightgray", linewidth=1)
    axes[1].set_title("Probe readout: shift toward target\n(did steering move the REPRESENTATION?)")
    axes[2].set_title("% stated answer changed at all")
    for ax in axes:
        ax.set_xlabel("alpha (x this image's own activation norm)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("mean shift score")
    axes[1].set_ylabel("mean shift score")
    axes[2].set_ylabel("fraction")

    fig.suptitle("Experiment B: steering along the probe direction")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "experiment_b_steering.png"), dpi=120)
    plt.close(fig)


def print_experiment_b_summary(summary_df):
    lines = ["=== EXPERIMENT B SUMMARY (steering) ===", ""]
    header = f"{'direction':<17}{'alpha':>7}{'n':>5}{'out_moved%':>11}{'out_shift':>10}{'out_chg%':>9}" \
             f"{'probe_moved%':>13}{'probe_shift':>12}"
    lines.append(header)
    for _, r in summary_df.sort_values(["direction", "alpha"]).iterrows():
        lines.append(
            f"{r['direction']:<17}{r['alpha']:>7.2f}{int(r['n']):>5}{r['pct_moved_toward']:>11.1%}"
            f"{r['mean_shift_score']:>10.2f}{r['pct_answer_changed']:>9.1%}"
            f"{r['pct_probe_moved_toward']:>13.1%}{r['mean_probe_shift_score']:>12.2f}"
        )
    lines.append("")
    lines.append("'probe_moved%'/'probe_shift' verify the intervention internally: they measure whether")
    lines.append("the probe's OWN readout of the angle moved toward the target, independent of whether")
    lines.append("the stated answer did. If probe_shift tracks alpha closely but out_shift stays flat,")
    lines.append("steering is moving the representation but the output ignores it -- not that steering failed.")

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

def verify_vision_swap(model, processor, decoder_layers, image_features_owners, vision_method, image_token_id,
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
        base_a = run_baseline(model, processor, a_path, image_token_id, image_features_owners, vision_method,
                               max_new_tokens=max_new_tokens, require_vision_capture=True)
        base_b = run_baseline(model, processor, b_path, image_token_id, image_features_owners, vision_method,
                               max_new_tokens=max_new_tokens, require_vision_capture=True)

        # (a) the swap writes a genuinely different tensor
        vision_diff = relative_l2_diff(base_a["vision_output"], base_b["vision_output"])

        # (b) propagation into the LLM: same A input, only the image features
        # differ, so ANY difference at layer 0/1 must come from the swap.
        replacement_b = vision_replacement_from(vision_method, base_b)
        with apply_vision_replacement(vision_method, decoder_layers, image_features_owners,
                                       base_a["image_mask"], replacement_b):
            hs_patched = forward_hidden_states(model, base_a["inputs"])
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
                ans_zero = generate_answer(model, processor, base_a["inputs"], max_new_tokens)
            changed_zero = (ans_zero != base_a["raw_answer"])

            noise_image = make_random_noise_image(size=512, rng=rng)
            noise_base = run_baseline(model, processor, noise_image, image_token_id, image_features_owners,
                                       vision_method, max_new_tokens=1, require_vision_capture=True)
            noise_replacement = vision_replacement_from(vision_method, noise_base)
            changed_noise, ans_noise = None, None
            if noise_replacement is not None and noise_replacement.shape == replacement_b.shape:
                with apply_vision_replacement(vision_method, decoder_layers, image_features_owners,
                                               base_a["image_mask"], noise_replacement):
                    ans_noise = generate_answer(model, processor, base_a["inputs"], max_new_tokens)
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


def verify_decoder_patch(model, processor, decoder_layers, image_token_id, image_features_owners, vision_method,
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
        base_a = run_baseline(model, processor, a_path, image_token_id, image_features_owners, vision_method,
                               max_new_tokens=max_new_tokens)
        base_b = run_baseline(model, processor, b_path, image_token_id, image_features_owners, vision_method,
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
                    hs_patched = forward_hidden_states(model, base_a["inputs"])
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


def run_verification(model, processor, decoder_layers, image_features_owners, vision_method, image_token_id,
                      df, images_dir, out_dir, n_verify_pairs=3, verify_layers=None,
                      min_gap=MIN_GAP_MINUTES, max_new_tokens=MAX_NEW_TOKENS, seed=SEED):
    if verify_layers is None:
        num_layers = len(decoder_layers)
        verify_layers = sorted(set([0, 1, num_layers // 2, num_layers]))
    rng = np.random.RandomState(seed)

    pairs = build_pairs(df, n_pairs=n_verify_pairs, min_gap=min_gap, seed=seed)
    if len(pairs) == 0:
        raise ValueError("No (A, B) pairs available for --verify -- check --min_gap against this dataset.")

    print(f"\n{'=' * 70}\nVERIFICATION: confirming interventions actually land, before trusting "
          f"any null result\n{'=' * 70}")
    print(f"Using {len(pairs)} pair(s), decoder layers {verify_layers}, vision method '{vision_method}'.")

    vision_lines, vision_records = verify_vision_swap(
        model, processor, decoder_layers, image_features_owners, vision_method, image_token_id,
        pairs, images_dir, max_new_tokens, rng)
    decoder_lines, decoder_records = verify_decoder_patch(
        model, processor, decoder_layers, image_token_id, image_features_owners, vision_method, pairs, images_dir,
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

def parse_layers_arg(value):
    if value is None:
        return None
    return [int(x) for x in value.split(",") if x.strip() != ""]


def parse_alphas_arg(value):
    return [float(x) for x in value.split(",") if x.strip() != ""]


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
    parser.add_argument("--verify_layers", type=parse_layers_arg, default=None,
                         help="--verify: comma-separated decoder layers to check (default: 0, 1, a middle "
                              "layer, and the final layer)")
    parser.add_argument("--data_csv", type=str, default=DATA_CSV)
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--direction_path", type=str, default=DIRECTION_PATH,
                         help="Experiment B: path to the .npz saved by `python probe.py --stage direction`")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS,
                         help="Experiment A: number of (A, B) pairs. Experiment B: number of steered images.")
    parser.add_argument("--max_pairs", type=int, default=None,
                         help="cap on pairs/images actually used, for a quick smoke test (overrides --n_pairs downward)")
    parser.add_argument("--layers", type=parse_layers_arg, default=None,
                         help="Experiment A: comma-separated layers to sweep, e.g. '0,1,21,36' "
                              "(default: every layer 0..num_layers)")
    parser.add_argument("--min_gap", type=int, default=MIN_GAP_MINUTES,
                         help="Experiment A: minimum circular minute gap between A and B")
    parser.add_argument("--target_offset", type=int, default=TARGET_OFFSET_MINUTES,
                         help="Experiment B: steering target = (true_minute + this) %% 60")
    parser.add_argument("--alphas", type=parse_alphas_arg, default=None,
                         help=f"Experiment B: comma-separated alpha values (default: {DEFAULT_ALPHAS})")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)

    model, processor = load_model(args.model_id)
    image_token_id = find_image_token_id(model, processor)
    decoder_layers = find_decoder_layers(model)
    image_features_owners = find_all_image_features_owners(model)
    print(f"Found {len(decoder_layers)} decoder layers.")

    # Determine ONCE (not per-call) which mechanism actually intercepts the
    # image representation for this model, and share it across --verify and
    # both experiments -- see determine_vision_interception_method.
    sample_path = os.path.join(args.images_dir, df.iloc[0]["filename"])
    vision_method = determine_vision_interception_method(model, processor, image_features_owners, sample_path)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.verify:
        run_verification(
            model, processor, decoder_layers, image_features_owners, vision_method, image_token_id,
            df, args.images_dir, args.out_dir, n_verify_pairs=args.verify_pairs,
            verify_layers=args.verify_layers, min_gap=args.min_gap,
            max_new_tokens=args.max_new_tokens, seed=args.seed)

    if args.experiment in ("a", "both"):
        trials_a = run_experiment_a(
            model, processor, decoder_layers, image_features_owners, vision_method, image_token_id,
            df, args.images_dir, args.out_dir, n_pairs=args.n_pairs, max_pairs=args.max_pairs,
            min_gap=args.min_gap, layers=args.layers, max_new_tokens=args.max_new_tokens, seed=args.seed)
        summary_a = summarize_experiment_a(trials_a, args.out_dir)
        plot_experiment_a(summary_a, args.out_dir)
        text_a = print_experiment_a_summary(summary_a, trials_a)
        with open(os.path.join(args.out_dir, "experiment_a_summary.txt"), "w") as f:
            f.write(text_a + "\n")

    if args.experiment in ("b", "both"):
        if not os.path.exists(args.direction_path):
            print(f"\nSkipping Experiment B: '{args.direction_path}' not found. "
                  f"Run `python probe.py --stage direction` first.")
        else:
            direction = load_probe_direction(args.direction_path)
            trials_b = run_experiment_b(
                model, processor, decoder_layers, image_token_id, image_features_owners, vision_method,
                direction, df, args.images_dir, args.out_dir, n_trials=args.n_pairs, max_pairs=args.max_pairs,
                target_offset=args.target_offset, alphas=args.alphas,
                max_new_tokens=args.max_new_tokens, seed=args.seed)
            summary_b = summarize_experiment_b(trials_b, args.out_dir)
            plot_experiment_b(summary_b, args.out_dir)
            text_b = print_experiment_b_summary(summary_b)
            with open(os.path.join(args.out_dir, "experiment_b_summary.txt"), "w") as f:
                f.write(text_b + "\n")

    print(f"\nAll outputs written to '{args.out_dir}/'.")


if __name__ == "__main__":
    main()
