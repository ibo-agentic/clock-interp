"""mech_lib.py -- shared machinery for the pre-registered mechanism experiments (PREREG_MECHANISM.md, E3/E1/E2).

Model-agnostic: everything goes through adapters.py (decoder_layers(), processor/tokenizer, build_inputs).
Nothing here edits intervene.py / knockout.py / adapters.py (they stay byte-identical); the few helpers that
knockout.py also has (attention-module discovery, eager forcing, mask arg handling) are COPIED here so this
file has no dependency on the heavy intervene.py import chain.

Contents
  prompts/tokens   : TextPrompt (chat-formatted text-only prompt, token ids, character->token spans of N and H)
  teacher forcing  : cand_tokens, score_candidates (log-prob of minute strings after a forced "HH:" prefix)
  residual hooks   : capture_residuals, patch_residuals (replace the residual stream at chosen positions/layers)
  logit lens       : lens_logprobs
  attention block  : BlockState, block_attention (text-query -> chosen key positions, per layer window), eager_attention
  generation       : greedy_generate_blocked (no KV cache; supports the block above)
  statistics       : cluster_bootstrap_ci, sign_flip_p
  labels           : label_trial (identical priority order to analysis/error_taxonomy.py)
"""
import contextlib
import re

import numpy as np
import torch

# ----------------------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------------------

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 11: "eleven", 12: "twelve"}
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")


def hour12(h):
    return 12 if h % 12 == 0 else h % 12


def hour_prefix(h):
    return f"{hour12(h):02d}:"


def minute_str(n):
    """Minute string for clock number n (1..12): 12 -> '00', 1 -> '05' ..."""
    return f"{(n % 12) * 5:02d}"


def _hs(out):
    return out[0] if isinstance(out, tuple) else out


def _with_hs(out, new):
    return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


# --- multi-device helpers (device_map='auto' puts layers / lm_head on different GPUs; hooks see per-layer devices) ---

def _param_device(module):
    """Device of a module's (first) parameter. Isolated so a test can emulate a multi-GPU split on CPU."""
    return next(module.parameters()).device


def _to(x, device):
    """Move tensor x to `device` (single place for every cross-device transfer in this file)."""
    return x.to(device)


# ----------------------------------------------------------------------------------------------------------
# model internals
# ----------------------------------------------------------------------------------------------------------

def lm_parts(adapter):
    """(layers, final_norm, lm_head). final_norm is the module named `norm` that lives next to `layers`."""
    layers = adapter.decoder_layers()
    parent = None
    for _, mod in adapter.model.named_modules():
        if getattr(mod, "layers", None) is layers:
            parent = mod
            break
    if parent is None or not hasattr(parent, "norm"):
        raise AttributeError("could not find the module holding decoder `layers` and its final `norm`")
    head = adapter.model.get_output_embeddings()
    if head is None:
        raise AttributeError("model has no output embeddings (lm_head)")
    return layers, parent.norm, head


def find_attention_modules(layers):
    """Copy of knockout.find_attention_modules: exactly one submodule with a callable o_proj per decoder layer."""
    mods = []
    for i, layer in enumerate(layers):
        c = [m for _, m in layer.named_modules() if hasattr(m, "o_proj") and callable(getattr(m, "o_proj", None))]
        if len(c) != 1:
            raise AttributeError(f"decoder layer {i}: expected exactly one attention module, found {len(c)}")
        mods.append(c[0])
    return mods


@contextlib.contextmanager
def eager_attention(adapter):
    """Force eager attention for the duration (an additive 4-D mask is only exposed to hooks in eager mode).
    Uses the public set_attn_implementation when available, else patches every config's _attn_implementation.
    Always restored."""
    model = adapter.model
    restore = []
    used_api = False
    if hasattr(model, "set_attn_implementation"):
        try:
            old = getattr(model.config, "_attn_implementation", None)
            model.set_attn_implementation("eager")
            restore.append(lambda: model.set_attn_implementation(old if old else "sdpa"))
            used_api = True
        except Exception:  # noqa: BLE001
            used_api = False
    if not used_api:
        seen = set()
        for m in model.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None and id(cfg) not in seen and hasattr(cfg, "_attn_implementation"):
                seen.add(id(cfg))
                restore.append((lambda c, o: (lambda: setattr(c, "_attn_implementation", o)))(cfg, cfg._attn_implementation))
                cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        for r in restore:
            r()


# ----------------------------------------------------------------------------------------------------------
# prompts and tokens
# ----------------------------------------------------------------------------------------------------------

class TextPrompt:
    """A text-only chat prompt: ids (1,P) on the model device, plus token index spans of the first / second
    'number X' occurrence (N-span = minute number, H-span = hour number, INCLUDING the space token before the digits),
    the tokens between them, the final token, and the trailing question tokens (used for random-position controls)."""

    def __init__(self, adapter, prompt, find_spans=True):
        self.prompt = prompt
        inputs = adapter.build_inputs(None, prompt)
        self.ids = inputs["input_ids"][:, :].to(adapter.model.device)
        self.P = self.ids.shape[1]
        self.spans = None
        if find_spans:
            self.spans = self._find_spans(adapter, prompt)

    def _find_spans(self, adapter, prompt):
        proc = adapter.processor
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tok = proc.tokenizer
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc["input_ids"]) != self.ids[0].tolist():
            raise AssertionError("tokenizer(chat_text) differs from the adapter's input_ids -- cannot align spans")
        offs = enc["offset_mapping"]
        base = text.index(prompt)
        ms = list(re.finditer(r"number (\d+)", prompt))
        if len(ms) < 2:
            return None
        out = {}
        for name, m in (("N", ms[0]), ("H", ms[1])):
            a, b = base + m.start(1) - 1, base + m.end(1)       # include the preceding space
            out[name] = [i for i, (s, e) in enumerate(offs) if e > a and s < b and e > s]
        a_between, b_between = base + ms[0].end(1), base + ms[1].start(1) - 1
        out["between"] = [i for i, (s, e) in enumerate(offs) if e > a_between and s < b_between and e > s]
        q = prompt.find("What time")
        out["question"] = ([i for i, (s, e) in enumerate(offs) if s >= base + q and e > s and i < self.P - 5]
                           if q >= 0 else [])
        out["last"] = [self.P - 1]
        # sanity: N/H spans must decode to the digits
        for k, m in (("N", ms[0]), ("H", ms[1])):
            dec = tok.decode([int(self.ids[0, i]) for i in out[k]]).strip()
            if dec != m.group(1):
                raise AssertionError(f"{k}-span decodes to {dec!r}, expected {m.group(1)!r}")
        return out


def cand_tokens(tok, hour_pref, minute):
    """(prefix_ids, cand_ids) for the forced answer '<HH>:' + '<MM>'. The full-string tokenisation must equal
    prefix + candidate (true for digit-split tokenizers); asserted."""
    p = tok(hour_pref, add_special_tokens=False)["input_ids"]
    f = tok(hour_pref + minute, add_special_tokens=False)["input_ids"]
    if f[:len(p)] != p or len(f) - len(p) != len(minute):
        raise AssertionError(f"tokenisation not prefix-stable for {hour_pref!r}+{minute!r}: {p} vs {f}")
    return p, f[len(p):]


def forward_logits(adapter, ids):
    attn = torch.ones_like(ids)
    out = adapter.model(input_ids=ids, attention_mask=attn, use_cache=False)
    return out.logits


@contextlib.contextmanager
def capture_residuals(layers):
    """Collect every decoder layer's output hidden state (B,T,d) into a list (index = layer)."""
    store = [None] * len(layers)
    hs = []
    for i, ly in enumerate(layers):
        def hook(_m, _a, out, i=i):
            store[i] = _hs(out).detach()
        hs.append(ly.register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


@contextlib.contextmanager
def patch_residuals(layers, layer_ids, positions, source):
    """Replace output[:, positions] of the chosen decoder layers by source[layer][positions] (source: list of (T,d)
    tensors; broadcast over the batch). Only positions < source length are touched; always removed."""
    pos = torch.as_tensor(sorted(set(int(p) for p in positions)), dtype=torch.long)
    hs = []
    for li in layer_ids:
        def hook(_m, _a, out, li=li):
            h = _hs(out)
            if h.shape[1] <= int(pos.max()):
                return None
            new = h.clone()
            src = _to(source[li][_to(pos, source[li].device)], h.device).to(h.dtype)
            new[:, _to(pos, h.device)] = src
            return _with_hs(out, new)
        hs.append(layers[li].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


# ----------------------------------------------------------------------------------------------------------
# teacher-forced scoring + logit lens
# ----------------------------------------------------------------------------------------------------------

@torch.no_grad()
def score_candidates(adapter, prompt_ids, prefix_ids, cands, return_resid=False):
    """Log-prob of each candidate (list of equal-length token lists) after prompt+prefix. Returns tensor (B,)
    (float32); with return_resid also the per-layer residual stores (B,T,d)."""
    layers = adapter.decoder_layers()
    dev = prompt_ids.device
    pre = torch.tensor(prefix_ids, dtype=torch.long, device=dev)[None]
    rows = [torch.cat([prompt_ids, pre, torch.tensor(c, dtype=torch.long, device=dev)[None]], dim=1) for c in cands]
    ids = torch.cat(rows, dim=0)
    q = prompt_ids.shape[1] + len(prefix_ids) - 1            # position whose output predicts cand[0]
    if return_resid:
        with capture_residuals(layers) as store:
            logits = forward_logits(adapter, ids)
    else:
        logits = forward_logits(adapter, ids)
        store = None
    lsm = torch.log_softmax(logits[:, q:q + len(cands[0]), :].float(), dim=-1)
    tgt = _to(ids[:, q + 1:q + 1 + len(cands[0])], lsm.device)      # logits may live on another GPU than the ids
    lp = lsm.gather(-1, tgt[..., None])[..., 0].sum(-1)
    return (lp, store, q) if return_resid else lp


@torch.no_grad()
def lens_logprobs(adapter, store_row, q, cand):
    """Logit-lens log-prob of one candidate at EVERY layer: store_row = list of (T,d) residuals for that candidate's
    row; positions q.. predict cand tokens. Returns (n_layers,) float32 numpy.
    Multi-GPU safe: each residual lives on its own layer's device; it is moved to the final norm's device, then to the
    lm_head's device (they can differ, and the head may be tied to the embedding on GPU 0), and the target ids follow
    the logits."""
    layers, norm, head = lm_parts(adapter)
    nd, hd = _param_device(norm), _param_device(head)
    pos = torch.arange(q, q + len(cand))
    tgt = torch.tensor(cand, dtype=torch.long)
    out = []
    for h in store_row:
        x = _to(h[_to(pos, h.device)][None], nd)                  # (1,len,d) on the norm's device
        x = norm(x)
        logits = head(_to(x, hd))[0].float()                      # on the head's device
        lsm = torch.log_softmax(logits, dim=-1)
        out.append(lsm.gather(-1, _to(tgt, lsm.device)[:, None])[:, 0].sum().item())
    return np.array(out, dtype=np.float32)


# ----------------------------------------------------------------------------------------------------------
# attention knockout (query positions >= q_start  ->  key positions key_pos), greedy loop without KV cache
# ----------------------------------------------------------------------------------------------------------

class BlockState:
    def __init__(self):
        self.q_start = None      # None -> no blocking
        self.key_pos = None      # LongTensor of key indices
        self.calls = 0           # number of hook invocations that modified a mask (for the reach check)
        self.seen = 0            # number of hook invocations at all


def _extract_mask(args, kwargs):
    if len(args) > 1 and torch.is_tensor(args[1]):
        return args[1], "arg1"
    if kwargs.get("attention_mask") is not None:
        return kwargs["attention_mask"], "kwarg"
    return None, None


def _make_hook(state):
    def hook(_module, args, kwargs):
        state.seen += 1
        if state.q_start is None or state.key_pos is None or len(state.key_pos) == 0:
            return None
        mask, where = _extract_mask(args, kwargs)
        if mask is None or mask.dim() != 4 or not torch.is_floating_point(mask):
            raise RuntimeError("block_attention needs eager attention with a float 4-D additive mask; got "
                               f"{None if mask is None else (tuple(mask.shape), mask.dtype)}")
        new = mask.clone()
        rows = new[:, :, state.q_start:, :]
        rows[..., state.key_pos.to(new.device)] = torch.finfo(new.dtype).min
        state.calls += 1
        if where == "arg1":
            return (args[0], new) + tuple(args[2:]), kwargs
        kw = dict(kwargs)
        kw["attention_mask"] = new
        return args, kw
    return hook


@contextlib.contextmanager
def block_attention(adapter, layer_range, state):
    """Register the blocking pre-hook on attention modules of layers [start, end). Eager attention is forced
    for the duration. state.q_start / state.key_pos are read at every forward (set them from the loop)."""
    layers = adapter.decoder_layers()
    attn = find_attention_modules(layers)
    s, e = layer_range
    handles = []
    with eager_attention(adapter):
        for i in range(s, min(e, len(attn))):
            handles.append(attn[i].register_forward_pre_hook(_make_hook(state), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()


def guard_block(key_pos, prompt_len):
    """Pre-registered guard: refuse a block that removes (almost) the whole prompt."""
    if len(set(int(k) for k in key_pos)) > 0.5 * prompt_len:
        raise ValueError(f"refusing to block {len(key_pos)} of {prompt_len} prompt positions (destroys the prompt)")


@torch.no_grad()
def greedy_generate_blocked(adapter, prompt_ids, state, key_pos=None, max_new=24):
    """Greedy decode WITHOUT a KV cache (full forward each step). If key_pos is given, from the moment the first ':'
    has been generated, the colon token and all later positions cannot attend to key_pos (minute-slot queries).
    Stops as soon as 'D?D:DD' has been generated. Returns the generated text."""
    tok = adapter.processor.tokenizer
    P = prompt_ids.shape[1]
    ids = prompt_ids.clone()
    gen = []
    eos = tok.eos_token_id
    state.q_start, state.key_pos = None, (None if key_pos is None else torch.as_tensor(key_pos, dtype=torch.long))
    if key_pos is not None:
        guard_block(key_pos, P)
    colon_found = False
    for _ in range(max_new):
        logits = forward_logits(adapter, ids)[:, -1, :]
        nxt = int(logits.argmax(-1))
        if nxt == eos or nxt in getattr(tok, "all_special_ids", []):
            break
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        text = tok.decode(gen, skip_special_tokens=True)
        if not colon_found and ":" in text:
            colon_found = True
            for i in range(len(gen)):
                if ":" in tok.decode(gen[:i + 1], skip_special_tokens=True):
                    if key_pos is not None:
                        state.q_start = P + i
                    break
        if TIME_RE.search(text):
            break
    state.q_start = None
    return tok.decode(gen, skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------------------------------------
# statistics (unit of analysis = the (N,H) cell)
# ----------------------------------------------------------------------------------------------------------

def cluster_bootstrap_ci(values, clusters, n_boot=10000, seed=0, alpha=0.05):
    """Mean of `values` with a percentile CI from resampling whole clusters. Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    keys, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(keys))
    cnt = np.bincount(inv, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def sign_flip_p(values, clusters, n_perm=10000, seed=0):
    """Two-sided paired sign-flip permutation test on cluster means of per-trial differences."""
    v = np.asarray(values, dtype=float)
    keys, inv = np.unique(np.asarray(clusters), return_inverse=True)
    cm = np.bincount(inv, weights=v, minlength=len(keys)) / np.bincount(inv, minlength=len(keys))
    obs = abs(cm.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(cm)))
    perm = np.abs((signs * cm).mean(1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


# ----------------------------------------------------------------------------------------------------------
# error labels (same priority order as analysis/error_taxonomy.py)
# ----------------------------------------------------------------------------------------------------------

def parse_hhmm(text):
    m = TIME_RE.search(text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def label_answer(text, hour, truth_minute, image_minute=None):
    h, m = parse_hhmm(text)
    if m is None:
        return "unparsed", h, m
    leak = (int(hour) % 12) * 5
    if m == truth_minute:
        lab = "correct"
    elif m == leak:
        lab = "leak"
    elif image_minute is not None and m == image_minute:
        lab = "follows_image"
    elif m in ((truth_minute + 5) % 60, (truth_minute - 5) % 60):
        lab = "adjacent"
    elif m in (0, 30):
        lab = "default_0_30"
    else:
        lab = "other"
    return lab, h, m
