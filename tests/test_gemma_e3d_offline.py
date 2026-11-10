"""Offline test of the Gemma-3 E3d machinery on a TINY RANDOM Gemma3 with a CHARACTER-level tokenizer (digits are single tokens, like the real tokenizer) and local
(sliding-window) + global layers (PREREG_MECHANISM.md tenth entry, item 5).  CPU, no download.  It tests code, not science.

Model A: sliding window (32) SMALLER than the prompts, so the local-layer mask path is exercised -> the blocking hook must edit the mask in local AND global layers,
the unblocked path must be unchanged, and gate G12 must FLAG the window (G12b and G12c fail).
Model B: sliding window (1024) larger than prompt + generated tokens -> G12a/b/c and G13 pass, part d (minimal cells) runs and the intact-parsed summary is written.
usage: python tests/test_gemma_e3d_offline.py"""
import argparse
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

import json  # noqa: E402

import torch  # noqa: E402
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import (Gemma3Config, Gemma3ForConditionalGeneration, Gemma3ImageProcessor, Gemma3Processor, Gemma3TextConfig,  # noqa: E402
                          PreTrainedTokenizerFast, SiglipVisionConfig)

import adapters  # noqa: E402
import mech_e3 as E3  # noqa: E402
import mech_lib as ML  # noqa: E402
import test_gemma_adapter_offline as T  # noqa: E402

CHARS = [chr(c) for c in range(32, 127)] + ["\n"]


def build_char_gemma(window, layer_types=("sliding_attention", "sliding_attention", "full_attention")):
    vocab = {t: i for i, t in enumerate(T.SPECIALS + CHARS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok.decoder = decoders.Fuse()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    fast.add_special_tokens({"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>", "<start_of_image>", "<end_of_image>", "<image_soft_token>"]})
    fast.boi_token, fast.eoi_token, fast.image_token = "<start_of_image>", "<end_of_image>", "<image_soft_token>"
    fast.boi_token_id, fast.eoi_token_id, fast.image_token_id = (vocab[t] for t in ("<start_of_image>", "<end_of_image>", "<image_soft_token>"))
    ip = Gemma3ImageProcessor(size={"height": 56, "width": 56}, do_pan_and_scan=False)
    proc = Gemma3Processor(image_processor=ip, tokenizer=fast, chat_template=T.TEMPLATE, image_seq_length=T.N_TOK)
    V = len(vocab)
    text = Gemma3TextConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=len(layer_types), num_attention_heads=2, num_key_value_heads=1,
                            head_dim=16, max_position_embeddings=2048, sliding_window=window, layer_types=list(layer_types))
    vis = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=56, patch_size=14)
    cfg = Gemma3Config(text_config=text, vision_config=vis, mm_tokens_per_image=T.N_TOK, image_token_id=vocab["<image_soft_token>"],
                       boi_token_id=vocab["<start_of_image>"], eoi_token_id=vocab["<end_of_image>"])
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration(cfg).eval()
    model.generation_config.suppress_tokens = [vocab["<image_soft_token>"], vocab["<start_of_image>"]]
    return model, proc


def make_ctx(model, proc, out):
    ad = adapters.Gemma3Adapter()
    ad.model, ad.processor = model, proc
    ad.torch_dtype, ad.dtype_name = torch.float32, "float32"
    args = argparse.Namespace(out_dir=out, prior_dir="prior", smoke=True, d_mode="min", max_new_tokens=32, rep_penalty=None, g2_gate="own", c_pairs=4,
                              no_gate=True, d_intact_parsed=True)
    return E3.Ctx(ad, args)


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")

    check("Gemma3Adapter carries has_sliding_layers, Qwen/InternVL adapters do not",
          adapters.Gemma3Adapter.has_sliding_layers and not any(getattr(c, "has_sliding_layers", False) for c in (adapters.QwenVLAdapter, adapters.Qwen7BAdapter, adapters.InternVLAdapter, adapters.InternVL3_8BAdapter)))
    with tempfile.TemporaryDirectory() as tmp:
        # ------------------------------------------------------------------ model A: window smaller than the prompts
        model, proc = build_char_gemma(window=32)
        ctx = make_ctx(model, proc, os.path.join(tmp, "A"))
        ad = ctx.adapter
        info = ML.sliding_layer_info(ad)
        check("sliding_layer_info reads window and layer types", info["window"] == 32 and info["n_local"] == 2 and info["n_global"] == 1, str(info))
        tp = ctx.prompt(0, 4, 9)
        check("tiny prompt is longer than the window (so local layers really mask)", tp.P > 32, f"P={tp.P}")
        check("spans found (digits are single tokens)", tp.spans and len(tp.spans["H"]) >= 1 and len(tp.spans["N"]) >= 1, str({k: len(v) for k, v in tp.spans.items()}))
        caps = ML.capture_visible_sets(ad, tp.ids)
        loc, glo = caps[0], caps[2]
        check("local layers see fewer keys than the global layer when window < prompt", int(loc.sum()) < int(glo.sum()) and bool((caps[0] == caps[1]).all()), f"{int(loc.sum())} vs {int(glo.sum())}")
        # unblocked path unchanged: empty block == no hook
        st = ML.BlockState()
        with ML.eager_attention(ad):
            a = ML.greedy_generate_blocked(ad, tp.ids, st, key_pos=None, max_new=6)
        with ML.block_attention(ad, (0, ctx.L), st):
            b = ML.greedy_generate_blocked(ad, tp.ids, st, key_pos=[], max_new=6)
        check("unblocked path unchanged: empty block equals no hook", a == b, f"{a!r} vs {b!r}")
        # hook edits the mask in local AND global layers: capture AFTER the blocking pre-hook (hooks chain)
        L = ctx.L
        q0 = tp.P - 3
        base = ML.capture_visible_sets(ad, tp.ids)
        st = ML.BlockState()
        st.q_start, st.key_pos = q0, torch.as_tensor(tp.spans["H"], dtype=torch.long)
        attn = ML.find_attention_modules(ad.decoder_layers())
        seen = [None] * L
        handles = []
        with ML.block_attention(ad, (0, L), st):
            for i, am in enumerate(attn):
                def mk(i):
                    def h(_m, args, kwargs):
                        mk_, _ = ML._extract_mask(args, kwargs)
                        seen[i] = (mk_[0, 0] == 0).cpu()
                    return h
                handles.append(am.register_forward_pre_hook(mk(i), with_kwargs=True))
            try:
                ML.forward_logits(ad, tp.ids)
            finally:
                for h in handles:
                    h.remove()
        H = torch.as_tensor(tp.spans["H"], dtype=torch.long)
        blocked_ok = all(bool((~seen[i][q0:, H]).all()) for i in range(L))
        rows_before_same = all(bool((seen[i][:q0] == base[i][:q0]).all()) for i in range(L))
        other_keys_same = all(bool((seen[i][q0:][:, [j for j in range(tp.P) if j not in set(H.tolist())]] == base[i][q0:][:, [j for j in range(tp.P) if j not in set(H.tolist())]]).all()) for i in range(L))
        check("block removes the H keys for the late queries in EVERY layer (local and global)", blocked_ok)
        check("rows before q_start and all non-H keys are untouched in every layer", rows_before_same and other_keys_same)
        check("hook fired once per layer", st.calls == L and st.seen == L, f"calls={st.calls} seen={st.seen}")
        # gate G12 flags the window
        ok_gate = E3.part_verify(ctx)
        v = json.load(open(os.path.join(ctx.out, "verify.json")))["checks"]
        check("G12a passes (layout read)", v["G12a_sliding_layout_read_from_config"]["pass"], str({k: v["G12a_sliding_layout_read_from_config"].get(k) for k in ("window", "layer_types")}))
        check("G12b FLAGS window shorter than the prompt", not v["G12b_all_prompts_shorter_than_sliding_window"]["pass"])
        check("G12c FLAGS local/global layers seeing different keys", not v["G12c_local_and_global_layers_see_identical_keys"]["pass"], str(v["G12c_local_and_global_layers_see_identical_keys"].get("n_different")))
        check("G13 passes (hook fires on every layer exactly once)", v["G13_blocking_hook_fires_on_every_layer_once"]["pass"], str(v["G13_blocking_hook_fires_on_every_layer_once"]))
        check("overall verify gate FAILS because of G12 (window < prompt)", ok_gate is False)

        # ------------------------------------------------------------------ model B: window larger than prompt + new tokens
        model, proc = build_char_gemma(window=1024)
        ctx = make_ctx(model, proc, os.path.join(tmp, "B"))
        E3.part_verify(ctx)
        v = json.load(open(os.path.join(ctx.out, "verify.json")))["checks"]
        for k in ("G12a_sliding_layout_read_from_config", "G12b_all_prompts_shorter_than_sliding_window", "G12c_local_and_global_layers_see_identical_keys", "G13_blocking_hook_fires_on_every_layer_once"):
            check(f"window 1024: {k} passes", v[k]["pass"], str({kk: vv for kk, vv in v[k].items() if kk not in ("pass", "gating")})[:160])
        E3.part_d(ctx)
        E3.summarize_d(ctx)
        import numpy as np
        import pandas as pd
        d = os.path.join(ctx.out, "e3d_knockout.csv")
        kd = pd.read_csv(d)
        check("part d (smoke cells) wrote intact / H_all / random_all", {"intact", "H_all", "random_all"} <= set(kd.cell), str(sorted(set(kd.cell))))
        # the random tiny model never writes a time (all replies unparsed), so test the summary logic on a synthetic knockout file:
        rng = np.random.RandomState(0)
        rows = []
        for k in range(24):
            lead, n, h = k % 3, 1 + k // 3, 1 + k % 12                   # 24 unique keys
            parsed = k >= 6                                           # the first 6 keys: intact reply unparsed (long chain-of-thought)
            for cell in ("intact", "H_all", "random_all"):
                if not parsed and cell == "intact":
                    lab = "unparsed"
                elif cell == "intact":
                    lab = "leak" if k % 2 == 0 else "correct"
                elif cell == "H_all":
                    lab = "correct" if parsed else "unparsed"
                else:
                    lab = "leak" if k % 2 == 0 else "correct"
                rows.append({"cell": cell, "span": "x", "start": 0, "end": 3, "lead": lead, "N": n, "H": h, "raw": "", "label": lab, "pred_hour": None, "pred_minute": None,
                             "minute_correct": lab == "correct", "hour_correct": False})
        pd.DataFrame(rows).to_csv(d, index=False)
        E3.summarize_d(ctx)
        sa, sp = pd.read_csv(os.path.join(ctx.out, "e3d_summary.csv")), pd.read_csv(os.path.join(ctx.out, "e3d_summary_intactparsed.csv"))
        check("all-key summary uses all 24 keys", int(sa[sa.cell == "intact"].n.iloc[0]) == 24)
        check("intact-parsed summary uses only the 18 keys whose intact reply parsed", int(sp[sp.cell == "intact"].n.iloc[0]) == 18)
        check("both verdict files written; only the intact-parsed one carries the subset label", "subset" not in json.load(open(os.path.join(ctx.out, "e3d_prereg_verdict.json")))
              and json.load(open(os.path.join(ctx.out, "e3d_prereg_verdict_intactparsed.json")))["subset"] == "keys whose intact reply parsed")
        check("without --d_intact_parsed no extra files are written", True)
        ctx.args.d_intact_parsed = False
        os.remove(os.path.join(ctx.out, "e3d_summary_intactparsed.csv"))
        E3.summarize_d(ctx)
        check("flag off: only the all-key summary is (re)written", not os.path.exists(os.path.join(ctx.out, "e3d_summary_intactparsed.csv")))
    return ok


if __name__ == "__main__":
    r = main()
    print("\nALL PASS" if r else "\nSOME FAILED")
    sys.exit(0 if r else 1)
