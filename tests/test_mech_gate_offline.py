"""Offline tests for the mechanism verify gate (no GPU needed).

  python tests/test_mech_gate_offline.py <path-to-tiny-internvl-model>      (or env TINY_INTERNVL)

What it covers
  T1  a FAKE 2-GPU split: tensors carry a virtual device tag (cuda:0 / cuda:1), mixing two tags in one op raises, exactly like
      PyTorch does on real GPUs; module hooks mimic accelerate's device_map='auto' (layers 0..L/2 on cuda:0, the rest on
      cuda:1, final norm on the last device, lm_head on cuda:0 [tied, like Qwen-3B] or cuda:1 [untied, like Qwen-7B], logits
      returned on cuda:0).
      T1a  the ORIGINAL lens_logprobs (copied below) fails under the fake split with the same two errors seen on Kaggle
           -> the emulator reproduces the bug;
      T1b  the FIXED mech_lib functions (lens, score_candidates, capture/patch, a no-cache greedy loop, the attention block)
           run under the split and give the same numbers as on a single device.
  T2  the complete part_verify gate under the fake split: no check may crash (a crash would show an `error` key).
  T3  G2 diagnostics (gold vs own prefix, per-category breakdown, token-level match) with canned free-generation strings,
      because the random-weight tiny model never writes a colon.
It is an emulator, not real CUDA: it cannot prove real-GPU behaviour. The Kaggle gate-only notebook does that.
"""
import argparse
import os
import sys
import types

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mech_e3 as E  # noqa: E402
import mech_lib as ML  # noqa: E402
from adapters import get_adapter  # noqa: E402


# ----------------------------------------------------------------------------------------------------------
# fake multi-device layer
# ----------------------------------------------------------------------------------------------------------

class VT(torch.Tensor):
    """Tensor with a virtual device index (_vd). `.device` reports cuda:<_vd>."""
    _vd = None

    @property
    def device(self):
        return torch.device("cuda", self._vd)


def tag(t, k):
    if not isinstance(t, torch.Tensor):
        return t
    plain = t.as_subclass(torch.Tensor) if isinstance(t, VT) else t
    v = plain.as_subclass(VT)
    v._vd = k
    return v


def _flat(x):
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flat(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _flat(y)
    else:
        yield x


def _map(x, f):
    if isinstance(x, torch.Tensor):
        return f(x)
    if isinstance(x, torch.device):
        return f(x)
    if isinstance(x, tuple):
        return tuple(_map(y, f) for y in x)
    if isinstance(x, list):
        return [_map(y, f) for y in x]
    if isinstance(x, dict):
        if type(x) is dict:
            return {k: _map(v, f) for k, v in x.items()}
        import copy
        new = copy.copy(x)                      # e.g. transformers ModelOutput: keep the type
        for k in list(x.keys()):
            new[k] = _map(x[k], f)
        return new
    return x


class FakeCuda(TorchFunctionMode):
    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat = list(_flat((args, kwargs)))
        tags = {t._vd for t in flat if isinstance(t, VT)}
        devs = [t for t in flat if isinstance(t, torch.device) and t.type == "cuda"]
        if len(tags) > 1 and not devs:
            raise RuntimeError(f"Expected all tensors to be on the same device, but found devices cuda:{sorted(tags)} "
                               f"(in {getattr(func, '__name__', func)})")

        def strip(x):
            if isinstance(x, VT):
                return x.as_subclass(torch.Tensor)
            if isinstance(x, torch.device) and x.type == "cuda":
                return torch.device("cpu")
            return x

        a, kw = _map(args, strip), _map(kwargs, strip)
        out = func(*a, **kw)
        k = devs[-1].index if devs else (next(iter(tags)) if len(tags) == 1 else None)
        if k is None:
            return out
        if isinstance(out, torch.Tensor):
            return tag(out, k)
        if isinstance(out, (tuple, list)) and any(isinstance(o, torch.Tensor) for o in out):
            return tuple(tag(o, k) if isinstance(o, torch.Tensor) else o for o in out)
        return out


def install_split(adapter, head_dev):
    """accelerate-like hooks. Returns (handles, module->device dict)."""
    model = adapter.model
    layers = adapter.decoder_layers()
    L = len(layers)
    _, norm, head = ML.lm_parts(adapter)
    emb = model.get_input_embeddings()
    dev_of = {id(emb): 0, id(norm): 1, id(head): head_dev}
    for i, ly in enumerate(layers):
        dev_of[id(ly)] = 0 if i < L // 2 else 1
    hs = []

    def to_k(x, k):
        return _map(x, lambda t: tag(t, k) if isinstance(t, torch.Tensor) and not isinstance(t, torch.nn.Parameter) else t)

    def pre(k):
        def f(_m, args, kwargs):
            return to_k(args, k), to_k(kwargs, k)
        return f

    def post(k):
        def f(_m, _a, out):
            return to_k(out, k)
        return f

    for mod in [emb, norm, head, *layers]:
        k = dev_of[id(mod)]
        hs.append(mod.register_forward_pre_hook(pre(k), with_kwargs=True))
        hs.append(mod.register_forward_hook(post(k)))
    # top level: inputs on cuda:0 and outputs returned to cuda:0 (accelerate io_same_device)
    hs.append(model.register_forward_pre_hook(pre(0), with_kwargs=True))
    hs.append(model.register_forward_hook(post(0)))
    ML._param_device = lambda m: torch.device("cuda", dev_of.get(id(m), 0))
    return hs


# ----------------------------------------------------------------------------------------------------------

def original_lens_logprobs(adapter, store_row, q, cand):
    """The lens_logprobs that crashed on Kaggle (verbatim copy of the old version)."""
    layers, norm, head = ML.lm_parts(adapter)
    pos = torch.arange(q, q + len(cand), device=store_row[0].device)
    tgt = torch.tensor(cand, dtype=torch.long, device=store_row[0].device)
    out = []
    for h in store_row:
        x = norm(h[pos][None])
        lsm = torch.log_softmax(head(x)[0].float(), dim=-1)
        out.append(lsm.gather(-1, tgt[:, None])[:, 0].sum().item())
    return np.array(out, dtype=np.float32)


class IdsTagging:
    """Make TextPrompt.ids live on 'cuda:0' (as model.device would on a real device_map='auto' model)."""
    def __enter__(self):
        self.orig = ML.TextPrompt.__init__

        def init(s, adapter, prompt, find_spans=True):
            self.orig(s, adapter, prompt, find_spans)
            s.ids = tag(s.ids, 0)
        ML.TextPrompt.__init__ = init

    def __exit__(self, *a):
        ML.TextPrompt.__init__ = self.orig


def make_ctx(adapter, smoke=True, g2_gate="own"):
    args = argparse.Namespace(out_dir=os.path.join(HERE, "_out"), prior_dir="prior", smoke=smoke, max_new_tokens=8,
                              d_mode="min", c_pairs=3, no_gate=False, g2_gate=g2_gate, parts="verify")
    return E.Ctx(adapter, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiny", nargs="?", default=os.environ.get("TINY_INTERNVL"))
    tiny = ap.parse_args().tiny
    assert tiny and os.path.isdir(tiny), "give the tiny InternVL model dir (arg or env TINY_INTERNVL)"
    ad = get_adapter(model_id=tiny, adapter_name="internvl3-2b")
    ad.load(tiny)
    ctx = make_ctx(ad)
    L = ctx.L
    n, h, lead = 4, 7, 0
    results = []

    def rec(name, ok, info=""):
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

    # single-device reference numbers
    tp = ctx.prompt(lead, n, h)
    pre, cs = E.cands_for(ctx, h, E.ALL_MINUTES)
    lp_ref, store_ref, q = ML.score_candidates(ad, tp.ids, pre, cs, return_resid=True)
    lens_ref = ML.lens_logprobs(ad, [s[3] for s in store_ref], q, cs[3])
    src_ref = [s[0] for s in store_ref]

    for head_dev in (0, 1):
        label = f"head on cuda:{head_dev} ({'tied-like' if head_dev == 0 else 'untied-like'})"
        ctx2 = make_ctx(ad)
        hs = install_split(ad, head_dev)
        try:
            with FakeCuda(), IdsTagging():
                ctx2._pc.clear()
                tp2 = ctx2.prompt(lead, n, h)
                lp, store, q2 = ML.score_candidates(ad, tp2.ids, pre, cs, return_resid=True)
                devs = sorted({s[0]._vd for s in store})
                rec(f"T1 {label}: layer outputs really spread over both fake devices", devs == [0, 1], str(devs))
                # T1a: the ORIGINAL lens must fail here
                try:
                    original_lens_logprobs(ad, [s[3] for s in store], q2, cs[3])
                    rec(f"T1a {label}: original lens_logprobs fails under the split (emulator reproduces the Kaggle bug)", False,
                        "it did NOT fail")
                except RuntimeError as e:
                    rec(f"T1a {label}: original lens_logprobs fails under the split (emulator reproduces the Kaggle bug)", True,
                        str(e)[:80])
                # T1b: the fixed functions
                lens = ML.lens_logprobs(ad, [s[3] for s in store], q2, cs[3])
                rec(f"T1b {label}: fixed lens_logprobs == single-device", np.allclose(lens, lens_ref, atol=1e-4),
                    f"max diff {np.abs(lens - lens_ref).max():.2e}")
                rec(f"T1b {label}: score_candidates == single-device", float((lp.as_subclass(torch.Tensor) - lp_ref).abs().max()) < 1e-4)
                src = [s[0] for s in store]
                with ML.patch_residuals(ctx2.layers, [0, L - 1], list(range(tp2.P)), src):
                    lp1 = ML.score_candidates(ad, tp2.ids, pre, cs)
                rec(f"T1b {label}: identity patch changes nothing", float((lp1 - lp).abs().max()) < 1e-4)
                # cross-prompt patch must run and change something
                tpo = ctx2.prompt(lead, h, n)
                with ML.capture_residuals(ctx2.layers) as st_o:
                    with torch.no_grad():
                        ML.forward_logits(ad, tpo.ids)
                with ML.patch_residuals(ctx2.layers, [0], tp2.spans["H"], [s[0] for s in st_o]):
                    lp2 = ML.score_candidates(ad, tp2.ids, pre, cs)
                rec(f"T1b {label}: swap patch runs and acts", float((lp2 - lp).abs().max()) > 1e-6)
                # block + greedy no-cache loop
                st = ML.BlockState()
                with ML.block_attention(ad, (0, L), st):
                    g = ML.greedy_generate_blocked(ad, tp2.ids, st, key_pos=tp2.spans["H"], max_new=3)
                rec(f"T1b {label}: blocked greedy loop runs", isinstance(g, str))
        finally:
            for x in hs:
                x.remove()
            ML._param_device = lambda m: next(m.parameters()).device

    # restore the real _param_device
    ML._param_device = lambda m: next(m.parameters()).device

    # T2 whole gate under the split, with canned free generation (adapter.generate is not emulated)
    canned = ["01:30", "1:30", "The time is 4:25", "no time at all", "07:35", "7:35"]
    for head_dev in (0, 1):
        ctx3 = make_ctx(ad)
        cnt = {"i": 0}

        def gen(text, max_new=None, cnt=cnt):
            cnt["i"] += 1
            return canned[cnt["i"] % len(canned)]
        ctx3.gen = gen
        hs = install_split(ad, head_dev)
        try:
            with FakeCuda(), IdsTagging():
                ctx3._pc.clear()
                E.part_verify(ctx3)
        finally:
            for x in hs:
                x.remove()
            ML._param_device = lambda m: next(m.parameters()).device
        import json
        v = json.load(open(os.path.join(ctx3.out, "verify.json")))
        crashed = {k: c["error"] for k, c in v["checks"].items() if "error" in c}
        rec(f"T2 part_verify under fake split (head on cuda:{head_dev}): no check crashed", not crashed, str(crashed)[:300])

    # T3 G2 diagnostics with canned strings (single device)
    ctx4 = make_ctx(ad)
    pick = [(0, 4, 7), (0, 10, 12), (1, 3, 5), (2, 6, 1), (0, 8, 2), (1, 1, 11)]
    outs = ["07:20", "7:00", "The time is 05:15", "12:40", "02:45", "no colon"]
    it = iter(outs)
    ctx4.gen = lambda text, max_new=None: next(it)
    g2 = E.g2_gate(ctx4, pick)
    classes = {b["prefix_class"]: b["n"] for b in g2["breakdown"]}
    want = {"exact_match": 2, "same_hour_unpadded": 1, "other_text_before_colon": 1, "wrong_hour_padded": 1, "no_colon": 1}
    # (4,7)->"07:" exact; (10,12) "7:" wrong-hour unpadded; (3,5) text; (6,1) "12:" wrong padded; (8,2) "02:" exact; (1,11) none
    want = {"exact_match": 2, "wrong_hour_unpadded": 1, "other_text_before_colon": 1, "wrong_hour_padded": 1, "no_colon": 1}
    rec("T3 prefix classes counted as expected", classes == want, f"{classes}")
    rec("T3 own-prefix scoring ran for every trial with a colon", g2["n_own"] == 5, f"n_own={g2['n_own']}")
    rec("T3 token-level match computed", g2["own_token_match"] is not None, f"{g2['own_token_match']}")

    bad = [n for n, ok in results if not ok]
    print("\nALL OFFLINE GATE TESTS PASSED" if not bad else f"\nFAILED: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
