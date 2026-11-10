"""
adapters.py -- one class per model family, exposing the SAME interface, so
eval_behavior.py / describe_check.py / text_only_check.py / probe.py /
intervene.py can run against any of them without per-model branches
scattered through the analysis code.

WHY THIS EXISTS: every script in this project was written against
Qwen2.5-VL-3B specifically (hardcoded chat-template calls, hardcoded
`get_image_features`/decoder-layer discovery). To replicate the findings on
other models (Qwen2.5-VL-7B, Gemma 3, InternVL3) without copy-pasting each
script three times, the model-specific parts move HERE, once each, behind
one shared interface. The analysis code (probe fitting, activation
patching, transfer metrics) stays untouched -- it already didn't care which
model produced the tensors it operates on.

THE INTERFACE (see ModelAdapter): every concrete adapter implements
  .load()                            -- sets self.model / self.processor
  .build_inputs(image, prompt)       -- image=None => text-only (text_only_check.py)
  .generate_answer(inputs, max_new_tokens) -- raw text, ALREADY TRIMMED to
                                        just the newly generated tokens (this
                                        varies by model -- see InternVLAdapter)
  .image_token_positions(inputs)     -- (seq_len,) bool tensor
  .decoder_layers()                  -- nn.ModuleList of decoder blocks
  .num_layers                        -- property, len(decoder_layers())
  .vision_module()                   -- the vision tower submodule (probe.py
                                        Step 2's "vision_encoder" representation)
  .image_features_owners()           -- (owners, method_name) for the
                                        get_image_features-style vision-ceiling
                                        interception in intervene.py's
                                        Experiment A -- see below for why this
                                        is allowed to be unsupported (empty
                                        owners list) for a given model.

MUCH OF THIS WAS ALREADY MODEL-AGNOSTIC BY DESIGN, from earlier rounds
chasing Qwen-specific bugs: `find_decoder_layers_generic` and
`find_all_image_features_owners_generic` below (renamed, otherwise
UNCHANGED, from intervene.py) scan the module tree by CLASS NAME PATTERN
("...DecoderLayer", a callable "get_image_features"/etc.), not by a
hardcoded attribute path -- so they work for Gemma3DecoderLayer /
Qwen2DecoderLayer (InternVL's inner LLM) without any per-model code. Only
three things actually differ per model: how to build a prompt+image into
model inputs, how to decode a generated answer, and how to find the
image-token id.

TRUST LEVELS (read before using a non-Qwen adapter for anything that
matters):
  - QwenVLAdapter: the SAME code this project has run and verified since
    round 1 (see intervene.py's module docstring for the 3-round history of
    what turned out to be wrong before this was trusted) -- just relocated
    here, not rewritten. Regression-tested against a fake model to confirm
    the relocation didn't change behavior (test_adapters_qwen_regression.py).
  - Gemma3Adapter: built by reading THIS environment's actual installed
    transformers source (site-packages/transformers/models/gemma3/
    modeling_gemma3.py, processing_gemma3.py) -- not from memory. Confirmed
    from that source: `Gemma3DecoderLayer` (matches the generic scanner),
    `self.config.image_token_id` (a real config attribute, not guessed),
    `get_image_features()` returns an object whose `.pooler_output` holds
    the merged embedding (matches `_extract_image_features_tensor`'s
    existing priority order unchanged). NOT run against real weights --
    google/gemma-3-4b-it is a GATED repo this environment has no access to.
  - InternVLAdapter: REWRITTEN for the transformers-native "-hf" checkpoints
    (OpenGVLab/InternVL3-2B-hf / -8B-hf; AutoModelForImageTextToText +
    AutoProcessor + the model's own chat template). The earlier version targeted
    the custom-code checkpoints and bypassed the chat template entirely, so it
    was replaced rather than fixed. Facts (template, 256 tokens per 448 tile,
    no default system prompt, crop_to_patches handling, bf16 cards vs this
    project's fp16) are documented at the class. NOT run against real weights
    yet -- use `python adapters.py --smoke --model_id <id>` first.

RUN --verify FOR EVERY MODEL BEFORE TRUSTING ANY INTERVENTION RESULT FROM
IT. This is not boilerplate caution: this project's own history is that
Qwen's `get_image_features` interception looked correct by every
superficial check for two rounds before --verify caught it doing nothing.
There is no reason a different model's adapter would be immune to the same
class of bug on the first attempt with no real-weight testing.

WHY get_image_features SUPPORT IS OPTIONAL PER MODEL: intervene.py's
vision-ceiling condition and --verify's vision-swap check need to intercept
"the vision representation right before it's merged into the LLM's input
embeddings". Two ways to do that exist in this codebase: (1) monkey-patch a
`get_image_features`-style method (fragile -- failed --verify 3 times for
Qwen; see intervene.py's module docstring), or (2) patch
`hidden_states[0]` at the image-token positions, right before the decoder
stack runs (`layer0_embed` -- the SAME already-verified mechanism the
decoder-patch sweep itself uses, and the one this project's OWN history
says to default to). `image_features_owners()` returning `([], None)` just
means "only layer0_embed is available for this model" -- intervene.py's
`determine_vision_interception_method` already treats that as the trusted
default, not a degraded fallback.
"""

import os

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Generic, model-agnostic reflection helpers (moved from intervene.py,
# UNCHANGED -- these never assumed anything Qwen-specific in the first
# place; see the module docstring above).
# ---------------------------------------------------------------------------

def find_decoder_layers_generic(model):
    """Locate a model's stack of transformer decoder blocks by scanning the
    module tree for a ModuleList whose children's CLASS NAME contains
    "DecoderLayer" -- robust to the exact attribute path (which differs
    across Qwen2_5_VLForConditionalGeneration, Gemma3ForConditionalGeneration,
    and InternVL's wrapped Qwen2ForCausalLM/LlamaForCausalLM alike), since
    that naming convention has stayed consistent across all of them."""
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            if "DecoderLayer" in type(module[0]).__name__:
                return module
    raise AttributeError("Could not find a decoder layer stack (ModuleList of "
                          "*DecoderLayer modules) in this model's module tree.")


def find_vision_module_generic(model):
    """Locate the vision tower by scanning for a class name that looks like
    a vision transformer, instead of a hardcoded attribute path (matches
    probe.py's original `find_vision_module`, unchanged)."""
    for _, module in model.named_modules():
        cls_name = type(module).__name__
        if "VisionTransformer" in cls_name or ("Vision" in cls_name and cls_name.endswith("Model")):
            return module
    raise AttributeError("Could not find the vision tower in this model's module tree.")


def find_all_image_features_owners_generic(model, method_name="get_image_features"):
    """Find EVERY distinct object in the model's hierarchy that defines a
    callable `method_name` -- not just the first one found. See
    intervene.py's module docstring for why patching only one owner is
    wrong in general (Qwen defines get_image_features on two separate
    objects; only the inner one's calls actually matter for generation)."""
    seen_ids = set()
    owners = []

    def add(name, obj):
        if obj is None or id(obj) in seen_ids:
            return
        if callable(getattr(obj, method_name, None)):
            owners.append((name, obj))
            seen_ids.add(id(obj))

    add("model", model)
    add("model.model", getattr(model, "model", None))
    for name, module in model.named_modules():
        add(f"model.{name}", module)
    return owners


# ---------------------------------------------------------------------------
# Layer depth utilities -- shared across models with different depths, so
# --layers can be given as absolute indices OR relative depth, and results
# always report both (see intervene.py's --layers rel:... support).
# ---------------------------------------------------------------------------

def relative_depth(layer, num_layers):
    """Layer 0 (embeddings) -> 0.0, layer num_layers (final block output)
    -> 1.0 -- the same [0, 1] scale regardless of how many decoder layers
    the model actually has, so a "readout window" found on one model can be
    compared to another's by POSITION IN THE STACK, not absolute index."""
    if num_layers <= 0:
        return float("nan")
    return layer / num_layers


def resolve_layers_arg(spec, num_layers):
    """Parse a --layers value into a list of ABSOLUTE layer indices for a
    model with `num_layers` decoder layers. Accepts:
      - plain comma-separated absolute indices: "0,1,21,36"
      - relative depth, prefixed "rel:": "rel:0.4,0.5,0.6,0.7" -> rounded to
        the nearest absolute layer for THIS model's depth (so the same
        --layers value sweeps "the same relative window" on a 28-layer and
        a 36-layer model, rather than the same absolute numbers landing at
        very different fractions of each model's depth)
      - "readout_window": kept as a literal string in intervene.py, resolved
        to READOUT_WINDOW_LAYERS_A there (Qwen-3B-specific absolute layers
        from the n=60 transfer analysis) -- NOT reinterpreted here, since
        that shorthand is this project's OWN prior finding for one specific
        model, not a general relative-depth spec.
    Returns a sorted list of unique ints in [0, num_layers].
    """
    if spec.startswith("rel:"):
        fractions = [float(x) for x in spec[len("rel:"):].split(",") if x.strip() != ""]
        layers = sorted(set(int(round(f * num_layers)) for f in fractions))
        return [l for l in layers if 0 <= l <= num_layers]
    return sorted(set(int(x) for x in spec.split(",") if x.strip() != ""))


# ---------------------------------------------------------------------------
# The shared adapter interface
# ---------------------------------------------------------------------------

class ModelAdapter:
    """Base class -- documents the interface; concrete adapters below
    implement every method. Not meant to be instantiated directly."""

    short_name = None        # used for outputs/<short_name>/ (see main scripts)
    default_model_id = None
    supports_get_image_features = False  # see module docstring

    def load(self, model_id=None):
        raise NotImplementedError

    def build_inputs(self, image, prompt):
        """`image`: a PIL.Image, an image path (str), or None for a
        text-only input (text_only_check.py). Returns a plain dict of
        on-device tensors suitable for `model(**inputs, ...)` /
        `model.generate(**inputs, ...)` -- NOT a framework-specific
        BatchFeature, so downstream code (run_baseline, forward_hidden_states)
        can stay model-agnostic."""
        raise NotImplementedError

    def generate_answer(self, inputs, max_new_tokens=16, use_cache=True):
        raise NotImplementedError

    def image_token_positions(self, inputs):
        raise NotImplementedError

    def decoder_layers(self):
        raise NotImplementedError

    @property
    def num_layers(self):
        return len(self.decoder_layers())

    def vision_module(self):
        raise NotImplementedError

    def image_features_owners(self):
        """Returns (owners, method_name). `owners` is a list of
        (name, object) pairs as `find_all_image_features_owners_generic`
        returns; `method_name` is what to monkey-patch on them. An empty
        owners list means this model only supports the `layer0_embed`
        vision-interception method (see module docstring) -- NOT an error."""
        return [], None


def _open_image(image):
    if image is None:
        return None
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


# ---------------------------------------------------------------------------
# Qwen2.5-VL (both 3B and 7B -- same architecture, same everything, just a
# bigger checkpoint) -- the ORIGINAL code this project has run and verified
# since round 1, relocated here unchanged (see test_adapters_qwen_regression.py).
# ---------------------------------------------------------------------------

class QwenVLAdapter(ModelAdapter):
    short_name = "qwen2.5-vl-3b"
    default_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    supports_get_image_features = True

    def __init__(self, prompt=None):
        self.model = None
        self.processor = None
        self.prompt = prompt  # eval_behavior.py's PROMPT; unused by build_inputs directly

    def load(self, model_id=None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model_id = model_id or self.default_model_id
        print(f"Loading {model_id} in float16 ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()
        return self.model, self.processor

    def build_inputs(self, image, prompt):
        image = _open_image(image)
        if image is not None:
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is not None:
            inputs = self.processor(text=[chat_text], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=[chat_text], return_tensors="pt")
        return dict(inputs.to(self.model.device))

    @torch.no_grad()
    def generate_answer(self, inputs, max_new_tokens=16, use_cache=True):
        extra = {}
        ov = rep_penalty_override()           # None (env unset) -> the call below is identical to before
        if ov is not None:
            extra["repetition_penalty"] = ov
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                             use_cache=use_cache, **extra)
        trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

    def image_token_positions(self, inputs):
        image_token_id = find_qwen_image_token_id(self.model, self.processor)
        return (inputs["input_ids"][0] == image_token_id).cpu()

    def decoder_layers(self):
        return find_decoder_layers_generic(self.model)

    def vision_module(self):
        return find_vision_module_generic(self.model)

    def image_features_owners(self):
        return find_all_image_features_owners_generic(self.model, "get_image_features"), "get_image_features"


def find_qwen_image_token_id(model, processor):
    """Unchanged from probe.py's original `find_image_token_id` -- kept as
    a free function (not just a QwenVLAdapter method) since probe.py's
    Step 2 code historically imported it directly."""
    token_id = getattr(model.config, "image_token_id", None)
    if token_id is not None:
        return token_id
    return processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


class Qwen7BAdapter(QwenVLAdapter):
    """Identical to QwenVLAdapter -- Qwen2.5-VL-7B-Instruct is the SAME
    architecture as the 3B, just more decoder layers and a bigger hidden
    size. Only `short_name`/`default_model_id` differ, so this is a
    one-line subclass rather than a copy of the whole adapter."""
    short_name = "qwen2.5-vl-7b"
    default_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"


# ---------------------------------------------------------------------------
# Gemma 3 -- different company, different vision tower (SigLIP), but a
# standard HF multimodal processor (ProcessorMixin) and a standardized
# get_image_features() method (see module docstring for what was verified
# from the installed transformers source vs. what's unverified without
# gated-repo access).
# ---------------------------------------------------------------------------

def _gemma_dtype():
    """dtype for the Gemma adapters only, from env CLOCK_DTYPE (float16 default; bfloat16; float32). Returns (torch dtype, name)."""
    name = os.environ.get("CLOCK_DTYPE", "float16").lower()
    table = {"float16": (torch.float16, "float16"), "fp16": (torch.float16, "float16"), "half": (torch.float16, "float16"),
             "bfloat16": (torch.bfloat16, "bfloat16"), "bf16": (torch.bfloat16, "bfloat16"),
             "float32": (torch.float32, "float32"), "fp32": (torch.float32, "float32")}
    if name not in table:
        raise ValueError(f"CLOCK_DTYPE must be float16, bfloat16 or float32 for Gemma (got '{name}')")
    return table[name]


def _gemma_max_memory():
    """Optional per-GPU weight cap for the Gemma adapters only, from env CLOCK_MAX_MEMORY, e.g. '0:9.5GiB,1:13.5GiB' (None when unset = unchanged behaviour).
    Leaves headroom on GPU 0 for the SigLIP vision tower's activations (notebook G0c)."""
    raw = os.environ.get("CLOCK_MAX_MEMORY", "").strip()
    if not raw:
        return None
    out = {}
    for part in raw.split(","):
        k, v = part.split(":", 1)
        out[int(k)] = v.strip()
    return out


class Gemma3Adapter(ModelAdapter):
    short_name = "gemma-3-4b-it"
    default_model_id = "google/gemma-3-4b-it"
    supports_get_image_features = True

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self, model_id=None):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        model_id = model_id or self.default_model_id
        self.torch_dtype, self.dtype_name = _gemma_dtype()
        print(f"Loading {model_id} in {self.dtype_name} (CLOCK_DTYPE={os.environ.get('CLOCK_DTYPE', '<unset>')}) ...")
        print(f"NOTE: {model_id} is a GATED HF repo -- this call will fail with a 401 "
              "unless HF_TOKEN is set (env var or `huggingface-cli login`) AND access has been "
              f"granted at https://huggingface.co/{model_id}.")
        mm = _gemma_max_memory()
        extra = {"max_memory": mm} if mm else {}
        if mm:
            print(f"max_memory (CLOCK_MAX_MEMORY): {mm}")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, dtype=self.torch_dtype, device_map="auto", **extra)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()
        return self.model, self.processor

    # Facts for the Method section / smoke output (checked against the installed transformers source, 5.17; Kaggle runs 5.20.2).
    expected_image_tokens = 256        # mm_tokens_per_image (SigLIP 896 px -> 4096 patches -> 4x4 avg-pool -> 256); pan-and-scan is OFF by default in the processor

    def device_map_record(self):
        """Full module->device map (accelerate hf_device_map), counts per device and the max_memory cap in force; for run_config.json."""
        dmap = getattr(self.model, "hf_device_map", None) or {}
        return {"device_map": {str(k): str(v) for k, v in dmap.items()},
                "device_map_counts": {str(d): sum(1 for v in dmap.values() if v == d) for d in set(dmap.values())},
                "max_memory": {str(k): v for k, v in (_gemma_max_memory() or {}).items()} or None,
                "disk_offload_note": "modules mapped to 'disk' are offloaded by accelerate and streamed in per forward pass; identical arithmetic, only slower"}

    def run_config_notes(self):
        return {**(self.device_map_record() if self.model is not None else {}),
                "chat_template": "Gemma3 turn format; <bos> comes from the template (tokenised via apply_chat_template, so no double BOS); no system prompt injected",
                "image": "one image = 256 soft tokens (processor default, do_pan_and_scan False); processor also returns token_type_ids (1 = image) used for the bidirectional image-block mask in prefill",
                "decoding": "greedy (do_sample=False), checkpoint sampling defaults top_k/top_p cleared, no repetition penalty (the checkpoint ships none)",
                "dtype": f"{getattr(self, 'dtype_name', 'float16')} (default float16; env CLOCK_DTYPE=bfloat16|float32). Measured in notebook G0: float16 gives non-finite logits for gemma-3-4b-it and -12b-it on T4"}

    def smoke_checks(self, inputs):
        """Gemma-only extra smoke checks (run_smoke calls this only if the attribute exists). Returns a list of problem strings."""
        probs = []
        ids = inputs["input_ids"][0]
        bos = self.processor.tokenizer.bos_token_id
        if bos is not None and int((ids[:2] == bos).sum()) != 1:
            probs.append(f"BOS appears {int((ids[:2] == bos).sum())}x in the first two tokens (double BOS)")
        if "token_type_ids" in inputs and int((inputs["token_type_ids"][0] == 1).sum()) != int(self.image_token_positions(inputs).sum()):
            probs.append("token_type_ids image count differs from image-token count")
        return probs

    def build_inputs(self, image, prompt):
        image = _open_image(image)
        if image is not None:
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        # tokenize=True via the template path: the template itself writes <bos>, and this route does not add a second one
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
        out = dict(inputs.to(self.model.device))
        if "pixel_values" in out:
            out["pixel_values"] = out["pixel_values"].to(self.model.dtype)
        return out

    @torch.no_grad()
    def generate_answer(self, inputs, max_new_tokens=16, use_cache=True):
        # Same trimming convention as Qwen: Gemma3's processor/generate also
        # returns the full (prompt + new tokens) sequence -- confirmed from
        # the installed transformers source (GenerationMixin.generate's
        # standard contract, which Gemma3ForConditionalGeneration doesn't
        # override), unlike InternVL below.
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                             top_k=None, top_p=None, use_cache=use_cache)
        trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

    def image_token_positions(self, inputs):
        # Confirmed from modeling_gemma3.py: self.config.image_token_id is a
        # real attribute (used internally at `input_ids == self.config.image_token_id`),
        # not a guess -- see module docstring.
        image_token_id = self.model.config.image_token_id
        return (inputs["input_ids"][0] == image_token_id).cpu()

    def decoder_layers(self):
        return find_decoder_layers_generic(self.model)

    def vision_module(self):
        return find_vision_module_generic(self.model)

    def image_features_owners(self):
        return find_all_image_features_owners_generic(self.model, "get_image_features"), "get_image_features"


class Gemma3_12BAdapter(Gemma3Adapter):
    """Same code path as the 4B; only the id differs (48 decoder layers vs 34; ~24 GB in fp16, split over the two T4s by device_map='auto')."""
    short_name = "gemma-3-12b-it"
    default_model_id = "google/gemma-3-12b-it"


# ---------------------------------------------------------------------------
# InternVL3 -- TRANSFORMERS-NATIVE "-hf" checkpoints (OpenGVLab/InternVL3-2B-hf,
# OpenGVLab/InternVL3-8B-hf), loaded with AutoModelForImageTextToText +
# AutoProcessor, i.e. the same standard path as Qwen. This REPLACES an earlier
# adapter for the custom-code checkpoints (trust_remote_code), which fed the raw
# string "<image>\n{prompt}" to the model with NO chat template at all and
# preprocessed the image by hand -- so a "same prompt" comparison against Qwen
# would have been invalid before any weights loaded.
#
# Facts used here, from the -hf model cards / raw repo files (not guessed):
#   * chat template: "<|im_start|>{role}\n{content}<|im_end|>\n" per turn,
#     generation prompt "<|im_start|>assistant\n"; an image renders as
#     "<IMG_CONTEXT>\n" BEFORE the text; NO default system prompt is injected
#     (Qwen2.5-VL's template injects one -- a known template difference between
#     the two model families, recorded in run_config.json via run_config_notes()).
#   * image_seq_length 256 per 448x448 tile; the processor expands each
#     <IMG_CONTEXT> into <img> + 256 x <IMG_CONTEXT> + </img> per tile.
#   * the hub preprocessor_config.json says crop_to_patches=false, but
#     InternVLProcessor's own _defaults (transformers 5.17 checked) set
#     crop_to_patches=True and that can override the hub value. So this adapter
#     passes crop_to_patches=False EXPLICITLY on every image call: every clock is
#     exactly ONE 448x448 tile = 256 image tokens, independent of transformers
#     version. (Method-section fact; recorded in run_config.json.)
#   * the cards use bfloat16; this project runs float16 (T4 has no native bf16).
#     float16 is the default; set env CLOCK_DTYPE=bfloat16 to switch ONLY this
#     adapter if the --smoke check shows NaN/inf or gibberish in fp16.
#
# NOT verified against real weights from this environment (no GPU): run
# `python adapters.py --smoke --model_id OpenGVLab/InternVL3-2B-hf` first.
# ---------------------------------------------------------------------------

INTERNVL_IMAGE_SIZE = 448          # one tile
INTERNVL_TOKENS_PER_TILE = 256     # image_seq_length in the -hf processor_config.json


def _internvl_dtype():
    name = os.environ.get("CLOCK_DTYPE", "float16").lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16, "float16"
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16, "bfloat16"
    raise ValueError(f"CLOCK_DTYPE must be float16 or bfloat16 (got '{name}')")


class InternVLAdapter(ModelAdapter):
    short_name = "internvl3-2b"
    default_model_id = "OpenGVLab/InternVL3-2B-hf"
    supports_get_image_features = False   # only layer0_embed interception is offered for this family
    expected_image_tokens = INTERNVL_TOKENS_PER_TILE   # single tile; checked by --smoke

    def __init__(self):
        self.model = None
        self.processor = None
        self.dtype, self.dtype_name = _internvl_dtype()

    def load(self, model_id=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model_id = model_id or self.default_model_id
        self.loaded_model_id = model_id
        print(f"Loading {model_id} in {self.dtype_name} (transformers-native InternVL, device_map='auto') ...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=self.dtype, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()
        return self.model, self.processor

    def build_inputs(self, image, prompt):
        image = _open_image(image)
        if image is not None:
            content = [{"type": "image"}, {"type": "text", "text": prompt}]   # image first, then text
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is not None:
            # crop_to_patches=False: ONE 448x448 tile / 256 image tokens, whatever the
            # processor's own default is on this transformers version (see banner above).
            inputs = self.processor(text=[chat_text], images=[image], return_tensors="pt",
                                    crop_to_patches=False)
        else:
            inputs = self.processor(text=[chat_text], return_tensors="pt")
        return dict(inputs.to(self.model.device, dtype=self.dtype))   # casts floating tensors only

    @torch.no_grad()
    def generate_answer(self, inputs, max_new_tokens=16, use_cache=True):
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                             use_cache=use_cache)
        trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

    def image_token_positions(self, inputs):
        return (inputs["input_ids"][0] == self.model.config.image_token_id).cpu()

    def decoder_layers(self):
        return find_decoder_layers_generic(self.model)

    def vision_module(self):
        return find_vision_module_generic(self.model)

    def image_features_owners(self):
        return [], None   # layer0_embed only (this project's trusted default for every model)

    def run_config_notes(self):
        """Merged into run_config.json by text_only_v2.py / conversion_test.py (Qwen adapters have no such
        method, so their configs are unchanged)."""
        return {
            "family": "InternVL3 (transformers-native -hf checkpoint)",
            "model_class": type(self.model).__name__ if self.model is not None else None,
            "dtype": self.dtype_name,
            "image_preprocessing": ("single 448x448 tile: processor called with crop_to_patches=False on every image, "
                                    f"so {INTERNVL_TOKENS_PER_TILE} image tokens (<img> + 256 x <IMG_CONTEXT> + </img>) "
                                    "per image regardless of the processor's own default"),
            "system_prompt": ("NONE: the InternVL3-hf chat template injects no default system prompt. Qwen2.5-VL's "
                              "template injects one by default -- a known template difference between the two "
                              "families; within each family text-only and image runs use the same template."),
            "image_position": "image first, then text (template renders '<IMG_CONTEXT>' + newline before the text)",
            "chat_template": "<|im_start|>{role} newline {content}<|im_end|> newline ... <|im_start|>assistant newline",
        }


class InternVL3_8BAdapter(InternVLAdapter):
    """Same family, bigger checkpoint (Qwen2-7B-class backbone, 28 layers, hidden 3584); only the ids differ."""
    short_name = "internvl3-8b"
    default_model_id = "OpenGVLab/InternVL3-8B-hf"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "qwen2.5-vl-3b": QwenVLAdapter,
    "qwen2.5-vl-7b": Qwen7BAdapter,
    "gemma-3-4b-it": Gemma3Adapter,
    "gemma-3-12b-it": Gemma3_12BAdapter,
    "internvl3-2b": InternVLAdapter,
    "internvl3-8b": InternVL3_8BAdapter,
}

# Model-id substrings -> short_name, so --model_id alone (without --adapter)
# picks the right adapter for any of the model ids this project actually uses.
_MODEL_ID_HINTS = [
    ("qwen2.5-vl-7b", "qwen2.5-vl-7b"),
    ("qwen2.5-vl-3b", "qwen2.5-vl-3b"),
    ("gemma-3-4b-it", "gemma-3-4b-it"),
    ("gemma-3-12b-it", "gemma-3-12b-it"),
    ("internvl3-8b", "internvl3-8b"),
    ("internvl3-2b", "internvl3-2b"),
]


def rep_penalty_override():
    """float from env CLOCK_REP_PENALTY (e.g. "1.0"), or None when unset (then every generate call is EXACTLY as before).
    Only the Qwen adapter honours it: the Qwen checkpoints ship repetition_penalty=1.05 in generation_config.json, InternVL3 ships none."""
    v = os.environ.get("CLOCK_REP_PENALTY", "").strip()
    return float(v) if v else None


def decoding_record(adapter):
    """Extra keys for run_config.json['decoding']: what repetition penalty the checkpoint ships and what (if anything) overrides it."""
    gc = getattr(getattr(adapter, "model", None), "generation_config", None)
    return {"checkpoint_default_repetition_penalty": getattr(gc, "repetition_penalty", None),
            "repetition_penalty_override_applied": rep_penalty_override() if isinstance(adapter, QwenVLAdapter) else None}


def get_adapter(model_id=None, adapter_name=None):
    """Resolve a --model_id and/or --adapter CLI value to a concrete adapter
    INSTANCE (not yet loaded -- call .load() separately, since scripts want
    to print/log before committing to the slow model download). If
    `adapter_name` is given, it must be a key in `_ADAPTERS`. Otherwise,
    `model_id` is matched against `_MODEL_ID_HINTS` (case-insensitive
    substring match); if it doesn't match any known model, defaults to
    QwenVLAdapter with a warning (this project's original, always-supported
    model) rather than silently guessing at a different architecture's
    conventions.
    """
    if adapter_name is not None:
        if adapter_name not in _ADAPTERS:
            raise ValueError(f"Unknown --adapter '{adapter_name}'. Choices: {sorted(_ADAPTERS.keys())}")
        return _ADAPTERS[adapter_name]()

    if model_id is not None:
        lower = model_id.lower()
        for hint, short_name in _MODEL_ID_HINTS:
            if hint in lower:
                return _ADAPTERS[short_name]()
        print(f"WARNING: '{model_id}' doesn't match any known adapter by name -- defaulting to "
              f"QwenVLAdapter's conventions (chat template + standard processor). Pass --adapter "
              f"explicitly ({sorted(_ADAPTERS.keys())}) if this model needs different handling.")

    return QwenVLAdapter()


def output_dir_for(base_out_dir, adapter):
    """outputs/<short_name>/... convention (see run_model.sh /
    compare_models.py) -- every script that writes per-model results calls
    this instead of using --out_dir directly, so results from different
    models never collide in the same folder."""
    return os.path.join(base_out_dir, adapter.short_name)


# ---------------------------------------------------------------------------
# CLI checks (python adapters.py --smoke / --qwen_regression)
# ---------------------------------------------------------------------------

def run_smoke(model_id=None, adapter_name=None, max_new_tokens=32, tmp_dir="adapter_smoke_images"):
    """Sanity-check an adapter on real weights BEFORE any long run:
      * 3 prompts WITH an image + 3 prompts WITHOUT, raw outputs printed;
      * 5 describe_check.py prompts on clocks, raw outputs printed, with the
        project's regexes applied so you can see early whether they fit this
        model's wording;
      * hard checks: logits finite (fp16 overflow shows up here), no empty
        outputs, and (adapters that declare expected_image_tokens) the image-token
        count and pixel_values shape.
    Returns True on PASS."""
    from transformers import set_seed
    from clocks import generate_clock_image
    from describe_check import PROMPT as DESCRIBE_PROMPT, parse_reply
    from eval_behavior import PROMPT as READ_PROMPT
    set_seed(0)

    adapter = get_adapter(model_id=model_id, adapter_name=adapter_name)
    adapter.load(model_id)
    problems = []
    print(f"\n=== SMOKE: {adapter.short_name}  ({type(adapter).__name__}) ===")
    print(f"dtype: {getattr(adapter, 'dtype_name', 'float16 (Qwen/Gemma adapters hardcode it)')}   "
          f"torch {torch.__version__}")
    try:
        import transformers
        print(f"transformers {transformers.__version__}")
    except Exception:
        pass
    dmap = getattr(adapter.model, "hf_device_map", None)
    if dmap:
        print("device map (module groups per device):",
              {str(d): sum(1 for v in dmap.values() if v == d) for d in set(dmap.values())})
    if hasattr(adapter, "run_config_notes"):
        for k, v in adapter.run_config_notes().items():
            print(f"  [{k}] {v}")

    os.makedirs(tmp_dir, exist_ok=True)
    times = [(10, 10), (3, 45), (7, 20), (12, 0), (5, 35)]
    paths = []
    for i, (h, m) in enumerate(times):
        p = os.path.join(tmp_dir, f"smoke_{i}_{h:02d}{m:02d}.png")
        generate_clock_image(h, m, p)
        paths.append(p)

    def forward_is_finite(inputs):
        with torch.no_grad():
            logits = adapter.model(**inputs).logits
        return bool(torch.isfinite(logits).all().item())

    print("\n--- 3 prompts WITH an image (eval_behavior.py prompt) ---")
    for p, (h, m) in list(zip(paths, times))[:3]:
        inputs = adapter.build_inputs(p, READ_PROMPT)
        n_img = int(adapter.image_token_positions(inputs).sum())
        fin = forward_is_finite(inputs)
        raw = adapter.generate_answer(inputs, max_new_tokens=max_new_tokens)
        if hasattr(adapter, "smoke_checks"):
            problems.extend(adapter.smoke_checks(inputs))
        shape = tuple(inputs["pixel_values"].shape) if "pixel_values" in inputs else None
        print(f"[{h}:{m:02d}] image_tokens={n_img} pixel_values={shape} logits_finite={fin}\n    raw: {raw!r}")
        exp = getattr(adapter, "expected_image_tokens", None)
        if exp is not None and n_img != exp:
            problems.append(f"image token count {n_img} != expected {exp}")
        if not fin:
            problems.append("non-finite logits with image")
        if not raw.strip():
            problems.append("empty output with image")

    print("\n--- 3 prompts WITHOUT an image ---")
    text_prompts = [
        "The long minute hand is pointing at the number 4, and the short hour hand is pointing at the number 9. "
        "What time does the clock show? Answer only in HH:MM format.",
        "The long minute hand is pointing at the number 7. How many minutes past the hour does this represent? "
        "Answer with a number only.",
        "What is 7 times 5? Answer with a number only.",
    ]
    for tp in text_prompts:
        inputs = adapter.build_inputs(None, tp)
        n_img = int(adapter.image_token_positions(inputs).sum())
        fin = forward_is_finite(inputs)
        raw = adapter.generate_answer(inputs, max_new_tokens=max_new_tokens)
        print(f"image_tokens={n_img} logits_finite={fin}\n    prompt: {tp!r}\n    raw: {raw!r}")
        if n_img != 0:
            problems.append("text-only input contains image tokens")
        if not fin:
            problems.append("non-finite logits text-only")
        if not raw.strip():
            problems.append("empty output text-only")

    print("\n--- 5 describe_check.py prompts (does the project's regex parsing fit this model's wording?) ---")
    fit = 0
    for p, (h, m) in zip(paths, times):
        inputs = adapter.build_inputs(p, DESCRIBE_PROMPT)
        raw = adapter.generate_answer(inputs, max_new_tokens=120)
        mn, hn, tm = parse_reply(raw)
        ok = mn is not None and hn is not None and tm is not None
        fit += int(ok)
        print(f"[{h}:{m:02d}] parsed minute_number={mn} hour_number={hn} time={tm} (all three parsed: {ok})\n    raw: {raw!r}")
        if not raw.strip():
            problems.append("empty output on describe prompt")
    print(f"\nregex fit: {fit}/5 describe replies fully parsed by describe_check.py's regexes.")
    if fit < 4:
        print("  -> LOW FIT: describe_check's regexes were tuned on Qwen's wording. Do NOT trust "
              "minute_desc_ok/hour_desc_ok for this model yet; propose tolerant-parse columns "
              "(additive, original columns untouched) and get approval before changing describe_check.py.")

    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            print(f"cuda:{d} peak memory {torch.cuda.max_memory_allocated(d) / 2**30:.1f} GiB")
    print("\nVERDICT:", "PASS" if not problems else "FAIL")
    for pr in sorted(set(problems)):
        print("  -", pr)
    if problems and getattr(adapter, "dtype_name", "") == "float16":
        print("  If non-finite logits or gibberish: retry with env CLOCK_DTYPE=bfloat16 (slow on T4) and record it.")
    return not problems


def run_qwen_regression(data_csv="data/data.csv", images_dir="data",
                        ref_csv="results/qwen2.5-vl-3b/results.csv", n=5, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Proof that Qwen behavior did not change: re-generate the first `n` rows of an EXISTING Qwen results
    CSV through the adapter path (eval_behavior.py's prompt, max_new_tokens=16, set_seed(0)) and compare
    raw outputs to the stored ones."""
    import pandas as pd
    from transformers import set_seed
    from eval_behavior import PROMPT as READ_PROMPT
    set_seed(0)
    ref = pd.read_csv(ref_csv).head(n)
    adapter = get_adapter(model_id=model_id)
    adapter.load(model_id)
    same = 0
    for _, r in ref.iterrows():
        inputs = adapter.build_inputs(os.path.join(images_dir, r["filename"]), READ_PROMPT)
        raw = adapter.generate_answer(inputs, max_new_tokens=16)
        ok = raw == str(r["raw_answer"]).strip()
        same += int(ok)
        print(f"{r['filename']}: {'IDENTICAL' if ok else 'DIFFERENT'}  now={raw!r}  stored={r['raw_answer']!r}")
    print(f"\nQWEN REGRESSION: {same}/{len(ref)} identical to {ref_csv}")
    return same == len(ref)


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Adapter checks on real weights (run before any long job).")
    ap.add_argument("--smoke", action="store_true", help="3 image + 3 text-only + 5 describe prompts; prints raw outputs")
    ap.add_argument("--qwen_regression", action="store_true",
                    help="re-generate the first 5 rows of an existing Qwen results CSV and compare")
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--data_csv", type=str, default="data/data.csv")
    ap.add_argument("--images_dir", type=str, default="data")
    ap.add_argument("--ref_csv", type=str, default="results/qwen2.5-vl-3b/results.csv")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    if args.smoke:
        sys.exit(0 if run_smoke(args.model_id, args.adapter, args.max_new_tokens) else 1)
    elif args.qwen_regression:
        sys.exit(0 if run_qwen_regression(args.data_csv, args.images_dir, args.ref_csv, args.n,
                                          args.model_id or "Qwen/Qwen2.5-VL-3B-Instruct") else 1)
    else:
        ap.print_help()
