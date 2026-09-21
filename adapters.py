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
  - InternVLAdapter: built by downloading and reading the ACTUAL custom
    modeling code from OpenGVLab/InternVL3-2B's real HF repo (trust_remote_code
    -- see modeling_internvl_chat.py's `forward`/`generate`/`chat` methods).
    Confirmed from that source: no standard `get_image_features` (the
    analogous method is `extract_feature`, and this adapter doesn't attempt
    to intercept it -- see "why get_image_features support is optional"
    below); `forward()` requires an extra `image_flags` tensor `generate()`
    does not; the generated sequence from `.generate()` is NOT prefixed with
    the prompt (no trimming needed, unlike Qwen/Gemma3). Simplified to a
    SINGLE image tile (no dynamic multi-tile splitting) since our clock
    renders are simple synthetic images that don't need multi-crop detail --
    this keeps `num_image_token` fixed and avoids a large chunk of
    InternVL's usual preprocessing complexity. NOT run against real weights
    (no local GPU).

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

    def generate_answer(self, inputs, max_new_tokens=16):
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
    def generate_answer(self, inputs, max_new_tokens=16):
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
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
        print(f"Loading {model_id} in float16 ...")
        print("NOTE: google/gemma-3-4b-it is a GATED HF repo -- this call will fail with a 401 "
              "unless HF_TOKEN is set (env var or `huggingface-cli login`) AND access has been "
              "granted at https://huggingface.co/google/gemma-3-4b-it.")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
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
    def generate_answer(self, inputs, max_new_tokens=16):
        # Same trimming convention as Qwen: Gemma3's processor/generate also
        # returns the full (prompt + new tokens) sequence -- confirmed from
        # the installed transformers source (GenerationMixin.generate's
        # standard contract, which Gemma3ForConditionalGeneration doesn't
        # override), unlike InternVL below.
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
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


# ---------------------------------------------------------------------------
# InternVL3 -- different vision encoder (InternViT) AND a fundamentally
# different input-construction convention (custom `trust_remote_code`
# modeling, no standard HF processor -- see module docstring for how this
# was reverse-engineered from the real modeling_internvl_chat.py).
# ---------------------------------------------------------------------------

# ImageNet mean/std -- InternVL's own preprocessing (from its model card /
# modeling_intern_vit.py's expected input normalization) uses these, not the
# CLIP-style stats Qwen/Gemma3's processors apply internally.
INTERNVL_IMAGE_MEAN = (0.485, 0.456, 0.406)
INTERNVL_IMAGE_STD = (0.229, 0.224, 0.225)
INTERNVL_IMAGE_SIZE = 448   # InternVL3's standard single-tile input resolution
INTERNVL_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
INTERNVL_IMG_START_TOKEN = "<img>"
INTERNVL_IMG_END_TOKEN = "</img>"


class InternVLAdapter(ModelAdapter):
    short_name = "internvl3-2b"
    default_model_id = "OpenGVLab/InternVL3-2B"
    supports_get_image_features = False   # see module docstring: extract_feature() isn't intercepted

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.num_image_token = None

    def load(self, model_id=None):
        from transformers import AutoModel, AutoTokenizer
        model_id = model_id or self.default_model_id
        print(f"Loading {model_id} in float16 (trust_remote_code=True -- this repo ships its own "
              "modeling code rather than transformers' native InternVL class; see adapters.py's "
              "module docstring) ...")
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model.eval()

        # img_context_token_id is NOT read from config -- InternVLChatModel's
        # own forward()/generate() require it be set as a plain attribute
        # before any call (confirmed from modeling_internvl_chat.py's chat()
        # method, which does exactly this every time).
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(INTERNVL_IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        # num_image_token is computed once in the model's __init__ from its
        # vision config (image_size, patch_size, downsample_ratio) -- already
        # an attribute on the loaded model, no need to recompute it here.
        self.num_image_token = self.model.num_image_token

        # This project's `.processor` naming convention (used by build_inputs
        # callers elsewhere, e.g. probe.py's find_image_token_id signature)
        # doesn't apply here -- InternVL has no combined processor, only a
        # tokenizer plus manual image preprocessing (see build_inputs below).
        self.processor = self.tokenizer
        return self.model, self.tokenizer

    def _preprocess_image(self, image):
        """A SINGLE 448x448 tile (no dynamic multi-tile splitting -- see
        module docstring for why: our clock renders are simple synthetic
        images that don't need multi-crop detail, and this keeps
        num_image_token fixed at exactly self.num_image_token, avoiding a
        large chunk of InternVL's usual dynamic-preprocessing complexity in
        a code path that can't be tested against real weights here)."""
        image = image.resize((INTERNVL_IMAGE_SIZE, INTERNVL_IMAGE_SIZE), Image.BICUBIC)
        arr = np.asarray(image).astype(np.float32) / 255.0
        mean = np.array(INTERNVL_IMAGE_MEAN, dtype=np.float32)
        std = np.array(INTERNVL_IMAGE_STD, dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(torch.float16)

    def build_inputs(self, image, prompt):
        image = _open_image(image)
        if image is not None:
            pixel_values = self._preprocess_image(image).to(self.model.device)
            # num_patches=1 (single tile, see _preprocess_image) -> exactly
            # self.num_image_token placeholder tokens, matching pixel_values'
            # single row -- see modeling_internvl_chat.py's chat()/generate(),
            # which require len(pixel_values) == sum(num_patches_list) and
            # selected.sum() == vit_embeds.numel()/C.
            image_tokens = (INTERNVL_IMG_START_TOKEN + INTERNVL_IMG_CONTEXT_TOKEN * self.num_image_token +
                             INTERNVL_IMG_END_TOKEN)
            query = f"<image>\n{prompt}".replace("<image>", image_tokens, 1)
            image_flags = torch.ones((1, 1), dtype=torch.long, device=self.model.device)
        else:
            pixel_values = None
            query = prompt
            image_flags = None

        encoded = self.tokenizer(query, return_tensors="pt")
        inputs = {
            "input_ids": encoded["input_ids"].to(self.model.device),
            "attention_mask": encoded["attention_mask"].to(self.model.device),
        }
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
            inputs["image_flags"] = image_flags   # forward() needs this; generate() ignores extra kwargs it doesn't use
        return inputs

    @torch.no_grad()
    def generate_answer(self, inputs, max_new_tokens=16):
        # UNLIKE Qwen/Gemma3: InternVLChatModel.generate() calls
        # self.language_model.generate(inputs_embeds=..., ...) internally
        # (confirmed from modeling_internvl_chat.py) -- when generation
        # starts from embeddings rather than input_ids, HF's generate()
        # returns ONLY the newly generated token ids, not prompt+continuation.
        # Trimming by prompt length (Qwen/Gemma3's convention) would be
        # WRONG here and would silently eat the first several real answer
        # tokens -- confirmed by reading modeling_internvl_chat.py's own
        # chat() method, which decodes the raw generate() output with NO
        # slicing at all.
        gen_kwargs = {k: v for k, v in inputs.items() if k != "image_flags"}
        generated_ids = self.model.generate(**gen_kwargs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    def image_token_positions(self, inputs):
        return (inputs["input_ids"][0] == self.model.img_context_token_id).cpu()

    def decoder_layers(self):
        # Scans the FULL model tree, so this finds language_model.model.layers
        # (a Qwen2DecoderLayer or LlamaDecoderLayer stack, depending on which
        # backbone this checkpoint uses -- see module docstring) without
        # needing to know that attribute path explicitly.
        return find_decoder_layers_generic(self.model)

    def vision_module(self):
        # InternVisionModel -- matches find_vision_module_generic's
        # "Vision" + endswith("Model") pattern (confirmed from
        # modeling_intern_vit.py's class name).
        return find_vision_module_generic(self.model)

    def image_features_owners(self):
        # InternVL's analogous method is `extract_feature`, not
        # `get_image_features` -- not intercepted by this adapter (see
        # module docstring on why that's fine: layer0_embed is the trusted
        # default anyway, for every model including Qwen).
        return [], None


class InternVL3_8BAdapter(InternVLAdapter):
    """Same architecture family as InternVL3-2B; only the checkpoint (and
    therefore its LLM backbone size) differs -- num_image_token is
    recomputed from THIS checkpoint's own vision config at load() time, not
    hardcoded, so this subclass needs nothing beyond the id override."""
    short_name = "internvl3-8b"
    default_model_id = "OpenGVLab/InternVL3-8B"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "qwen2.5-vl-3b": QwenVLAdapter,
    "qwen2.5-vl-7b": Qwen7BAdapter,
    "gemma-3-4b-it": Gemma3Adapter,
    "internvl3-2b": InternVLAdapter,
    "internvl3-8b": InternVL3_8BAdapter,
}

# Model-id substrings -> short_name, so --model_id alone (without --adapter)
# picks the right adapter for any of the model ids this project actually uses.
_MODEL_ID_HINTS = [
    ("qwen2.5-vl-7b", "qwen2.5-vl-7b"),
    ("qwen2.5-vl-3b", "qwen2.5-vl-3b"),
    ("gemma-3-4b-it", "gemma-3-4b-it"),
    ("internvl3-8b", "internvl3-8b"),
    ("internvl3-2b", "internvl3-2b"),
]


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
