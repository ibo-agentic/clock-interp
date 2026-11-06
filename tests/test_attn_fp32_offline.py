"""Offline test (CPU, tiny random Qwen2.5-VL) for the attention fixes in mech_lib.py:
  T1 eager_attention() forces 'eager' on every decoder attention module and restores the original implementation
  T2 fp32 eager output == sdpa output (fp32 model)               -> the replacement is the same math
  T3 fp16 model whose q.k^T overflows: stock eager -> non-finite, eager_attention() (fp32 accumulation) -> finite and
     close to sdpa                                              -> reproduces & fixes 'eager garbage in fp16'
  T4 blocking a key position changes the logits (hook reaches the mask) and an empty block changes nothing
  T5 a stale model.rope_deltas cannot change a text-only no-cache forward; reset_stateful_positions clears it
  T6 cache_vs_nocache_probe: finite, tiny diff, same argmax
usage: python tests/test_attn_fp32_offline.py"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import mech_lib as ML
from adapters import QwenVLAdapter
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


def build(dtype):
    torch.manual_seed(0)
    cfg = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=512,
                         rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[2, 3, 3]), tie_word_embeddings=True),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2, fullatt_block_indexes=[0], window_size=56),
        image_token_id=290, video_token_id=291, vision_start_token_id=292, vision_end_token_id=293)
    m = Qwen2_5_VLForConditionalGeneration(cfg).eval().to(dtype)
    ad = QwenVLAdapter(); ad.model = m
    return ad


def run(ad, ids, impl=None):
    return ML.forward_logits(ad, ids)[0].float()


ids = torch.randint(5, 280, (1, 24))
fails = []

def ck(name, ok, info=""):
    print(("PASS" if ok else "FAIL"), name, info); 
    if not ok: fails.append(name)

# T1/T2 fp32
ad = build(torch.float32)
m = ad.model
base = ML.attn_impl_report(ad); ck("T1a default impl reported", len(base) == 1, str(base))
ref = run(ad, ids)
with ML.eager_attention(ad):
    rep = ML.attn_impl_report(ad); e32 = run(ad, ids)
ck("T1b every decoder attention module is eager inside the context", set(rep) == {"eager"} and sum(rep.values()) == 3, str(rep))
ck("T1c restored afterwards", ML.attn_impl_report(ad) == base, str(ML.attn_impl_report(ad)))
ck("T2 fp32-eager == sdpa (fp32 model)", float((ref - e32).abs().max()) < 1e-3, f"maxdiff {float((ref - e32).abs().max()):.2e}")
with ML.eager_attention(ad, fp32=False):
    stock = run(ad, ids)
ck("T2b fp32 replacement == stock eager (fp32 model)", float((stock - e32).abs().max()) < 1e-3, f"maxdiff {float((stock - e32).abs().max()):.2e}")

# T3 fp16 overflow
ad16 = build(torch.float16); m16 = ad16.model
for l in ML.find_attention_modules(ad16.decoder_layers()):
    l.q_proj.weight.data *= 600; l.k_proj.weight.data *= 600       # q.k^T explodes past fp16 range
    l.q_proj.bias.data *= 600; l.k_proj.bias.data *= 600
sd = run(ad16, ids)
with ML.eager_attention(ad16, fp32=False):
    st = run(ad16, ids)
with ML.eager_attention(ad16):
    fx = run(ad16, ids)
print("   sdpa finite", bool(torch.isfinite(sd).all()), "| stock eager finite", bool(torch.isfinite(st).all()), "| fp32 eager finite", bool(torch.isfinite(fx).all()))
ck("T3a (repro) stock fp16 eager is non-finite when q.k overflows", not bool(torch.isfinite(st).all()))
ck("T3b fp32-accumulating eager is finite", bool(torch.isfinite(fx).all()))
if torch.isfinite(sd).all():
    ck("T3c ... and close to sdpa", float((torch.log_softmax(sd, -1) - torch.log_softmax(fx, -1)).abs().max()) < 0.5)

# T4 blocking
ad = build(torch.float32); st_ = ML.BlockState()
with ML.eager_attention(ad):
    lp0 = ML.forward_logits(ad, ids)[0, -1].float()
q = 20
for tag, key in (("N", [3, 4, 5]), ("empty", [])):
    st_.q_start, st_.key_pos, st_.calls = q, torch.as_tensor(key, dtype=torch.long), 0
    with ML.block_attention(ad, (0, 3), st_):
        lp1 = ML.forward_logits(ad, ids)[0, -1].float()
    d = float((lp1 - lp0).abs().max())
    if tag == "N":
        ck("T4a blocking changes logits and fires on every layer", d > 1e-3 and st_.calls >= 3, f"diff {d:.3g} calls {st_.calls}")
    else:
        ck("T4b empty block changes nothing", d < 1e-4, f"diff {d:.3g}")

# T5 stale rope_deltas
ref = ML.forward_logits(ad, ids)[0].float()
ad.model.model.rope_deltas = torch.tensor([[11]])
stale = ML.forward_logits(ad, ids)[0].float()      # forward_logits resets it
ck("T5a stale rope_deltas does not change the no-cache logits", float((ref - stale).abs().max()) < 1e-5)
ad.model.model.rope_deltas = torch.tensor([[11]]); ML.reset_stateful_positions(ad.model)
ck("T5b reset clears it", ad.model.model.rope_deltas is None)
# even WITHOUT the reset (raw forward) a constant shift must not matter for text-only
ad.model.model.rope_deltas = torch.tensor([[11]])
raw = ad.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0].float()
ck("T5c raw forward with stale rope_deltas still equal (RoPE is relative)", float((ref - raw).abs().max()) < 1e-4, f"{float((ref - raw).abs().max()):.2e}")

# T6 cache vs no-cache
ad = build(torch.float32)
dfv, ag, ok = ML.cache_vs_nocache_probe(ad, ids, [11, 22, 33, 44, 55])
ck("T6 cached decode == no-cache forward (fp32)", ok and dfv < 1e-3 and ag == 1.0, f"diff {dfv:.2e} agree {ag}")
print("\nALL ATTENTION TESTS PASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
