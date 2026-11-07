"""Offline tests (CPU, tiny random InternVL with a real image path) for the E1/E2 machinery in mech_lib.py / mech_e2.py.
usage: python tests/test_e1_e2_offline.py <tiny_internvl_dir>      (or env TINY_INTERNVL)

E1  T1  query-set B, all layers: last-position logits are IDENTICAL for two different images (every text->image path is cut)   [invariance]
    T2  same, through CACHED generation (prefill + decode steps): identical generated tokens for two different images
    T3  inactive block == no hook (exact);  block on layer 0 only and on layer 1 only both change the logits (per-layer reach)
    T4  query-set A differs from B (A leaves prefill text rows connected to the image) and still changes the output
    T5  hook counters: decode steps are hit (decode_calls > 0)
E2  T6  identity patch (source = own hidden state) changes nothing at any k;  T7 embedding-level (k=0) all-image patch with B == showing image B
    T8  patch on a subset of positions changes the logits less than the all-image patch; sets (minute/hour/other) behave
    T9  pixel_mask_to_token_mask maps a single pixel to exactly one token;  T10 build_pairs: same hour, circular distance >= 3, deterministic
    T11 token_sets on real renders: minute-hand tokens non-empty, strictly smaller than 'all', 'other' = all minus minute minus hour
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import mech_lib as ML
import mech_e2 as E2
import conversion_test as CT
from adapters import get_adapter

TINY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TINY_INTERNVL")
assert TINY, "give the tiny model dir"
fails = []


def ck(name, ok, info=""):
    print(("PASS" if ok else "FAIL"), name, info, flush=True)
    if not ok:
        fails.append(name)


ad = get_adapter(model_id=TINY, adapter_name="internvl3-2b"); ad.load(TINY)
layers = ad.decoder_layers(); L = len(layers)
tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_e12_tmp"); os.makedirs(tmp, exist_ok=True)


def inputs_for(n, h, text_n=3, text_h=5):
    p, _, _ = CT.ensure_image("matching_clock", n, h, tmp, 0)
    return ad.build_inputs(p, CT.prompt_for(0, text_n, text_h))


@torch.no_grad()
def last_logits(inp):
    return ad.model(**inp, use_cache=False).logits[0, -1].float()


i1, i2 = inputs_for(3, 5), inputs_for(9, 5)
mask = ad.image_token_positions(i1)
assert bool((mask == ad.image_token_positions(i2)).all())
st = ML.ImageBlockState()
base1, base2 = last_logits(i1), last_logits(i2)
ck("T0 the two images give different logits when nothing is blocked", float((base1 - base2).abs().max()) > 1e-3)

with ML.eager_attention(ad):
    l_eager1 = last_logits(i1)
st.set(mask, "B", True)
with ML.block_image_attention(ad, (0, L), st):
    b1, b2 = last_logits(i1), last_logits(i2)
ck("T1 query set B, all layers: logits independent of the image", float((b1 - b2).abs().max()) < 1e-3, f"max diff {float((b1 - b2).abs().max()):.2e}")
ck("T1b ... and different from the unblocked run", float((b1 - base1).abs().max()) > 1e-3)

st.set(mask, "B", True)
with ML.block_image_attention(ad, (0, L), st):
    with torch.no_grad():
        g1 = ad.model.generate(**i1, max_new_tokens=6, do_sample=False)[0, i1["input_ids"].shape[1]:].tolist()
        dec = st.decode_calls
    st.set(mask, "B", True)
    with torch.no_grad():
        g2 = ad.model.generate(**i2, max_new_tokens=6, do_sample=False)[0, i2["input_ids"].shape[1]:].tolist()
ck("T2 cached generation (prefill+decode) identical for two images under the B block", g1 == g2, f"{g1} vs {g2}")
ck("T5 decode steps hit the hook", dec > 0, f"decode_calls {dec}")

st.set(mask, "B", False)
with ML.block_image_attention(ad, (0, L), st):
    inactive = last_logits(i1)
ck("T3 inactive block == eager without hook (exact)", float((inactive - l_eager1).abs().max()) < 1e-6)
diffs = []
for rng in ((0, 1), (1, 2)):
    st.set(mask, "B", True)
    with ML.block_image_attention(ad, rng, st):
        diffs.append(float((last_logits(i1) - base1).abs().max()))
ck("T3b blocking a single layer (0 or 1) changes the logits", min(diffs) > 1e-4, f"{diffs}")

st.set(mask, "A", True)
with ML.block_image_attention(ad, (0, L), st):
    a1, a2 = last_logits(i1), last_logits(i2)
ck("T4 query set A: changes the output but is not equal to B", float((a1 - base1).abs().max()) > 1e-4 and float((a1 - b1).abs().max()) > 1e-4,
   f"A-vs-intact {float((a1 - base1).abs().max()):.3g}, A-vs-B {float((a1 - b1).abs().max()):.3g}; image-independence of A: {float((a1 - a2).abs().max()):.3g}")

# ------------------------------------------------------------------ E2
pos = mask.nonzero().flatten().tolist()

with torch.no_grad(), ML.capture_hidden(layers) as store_a:
    ad.model(**i1, use_cache=False)
with torch.no_grad(), ML.capture_hidden(layers) as store_b:
    ad.model(**i2, use_cache=False)
worst = 0.0
for k in range(L + 1):
    with ML.patch_hidden(layers, k, pos, store_a[k][0, pos]):
        worst = max(worst, float((last_logits(i1) - base1).abs().max()))
ck("T6 identity patch changes nothing at any k (0..L)", worst < 1e-3, f"max {worst:.2e}")
with ML.patch_hidden(layers, 0, pos, store_b[0][0, pos]):
    pat = last_logits(i1)
ck("T7 embedding-level all-image patch with B == showing image B", float((pat - base2).abs().max()) < 1e-2, f"max {float((pat - base2).abs().max()):.2e}")
half = pos[: len(pos) // 4]
with ML.patch_hidden(layers, 0, half, store_b[0][0, half]):
    part = last_logits(i1)
ck("T8 patching a quarter of the image tokens moves the logits less than patching all", 1e-6 < float((part - base1).abs().max()) <= float((pat - base1).abs().max()) + 1e-3,
   f"quarter {float((part - base1).abs().max()):.3g}, all {float((pat - base1).abs().max()):.3g}")
with ML.patch_hidden(layers, L // 2 or 1, pos, store_b[L // 2 or 1][0, pos]):
    mid = last_logits(i1)
ck("T8b a patch at an intermediate layer also moves the logits", float((mid - base1).abs().max()) > 1e-4)

m = np.zeros((512, 512), dtype=bool); m[100, 300] = True
tm = ML.pixel_mask_to_token_mask(m, 16)
ck("T9 one pixel -> exactly one token (16x16 grid)", int(tm.sum()) == 1 and bool(tm.reshape(16, 16)[100 * 16 // 512, 300 * 16 // 512]))

pr = E2.build_pairs(100)
ok = all(p["n_b"] != p["n_a"] and E2.circ(p["n_a"], p["n_b"]) >= 3 and p["h_c"] != p["h"] for p in pr) and pr == E2.build_pairs(100) and len({(p["n_a"], p["h"]) for p in pr}) == 100
ck("T10 pairs: circular distance >= 3, same hour for A/B, c1 partner has another hour, 100 distinct A, deterministic", ok)

sets, grid = E2.token_sets(5, 15, 40, 256, tmp)
sizes = {k: int(v.sum()) for k, v in sets.items()}
ck("T11 token sets on real renders", 0 < sizes["minute_hand"] < 256 and sizes["hour_hand"] > 0 and sizes["other"] == 256 - int((sets["minute_hand"] | sets["hour_hand"]).sum()) and grid == 16, str(sizes))
print("\nALL E1/E2 OFFLINE TESTS PASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
