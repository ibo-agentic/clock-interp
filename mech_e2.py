"""mech_e2.py -- E2: cross-clock image-token residual patching (PREREG_MECHANISM.md Section 6 + deviations log 2026-11-07, third entry).

ONE model per invocation. Parts: verify (gate; REQUIRED before `run` unless --no_gate) and run (resumable per pair).
For every pair: text fixed to clock A (the conversion-test prompt for variant `standard`; "hour number in the text, minute only in the image" for
variant `text_light`), image A shown; the hidden state of the IMAGE-token positions of that forward pass is replaced, at hidden-state index k, by
the corresponding rows of (real_B) clock B = same hour, minute number >= 3 away (circular);  (c1) clock C = same minute number as A, different hour
(noise floor);  (c2) norm-matched Gaussian noise.  Hidden-state index k: 0 = embedding output (vision features merged), k>=1 = output of decoder layer k-1.
Layers: k = 0 and floor(L*i/8) for i = 1..7 (fixed before any run). Token sets (image positions only): all, minute_hand, hour_hand, other
(= all minus minute and hour); minute/hour sets = tokens touched by the drawn hand strokes of A union B (pixel masks from re-rendering the clock
with and without the hands, mapped onto the token grid).  Patching acts on the prefill pass only; decoding then uses the patched KV cache
(a patch at k equals supplying image B to layers >= k while text positions at layers < k still saw image A: stated limitation).
Metrics: generated minute (cached greedy, --rep_penalty, max_new_tokens 8): follow_B / stays_A; teacher-forced log p(B minute) - log p(A minute)
after the forced gold hour prefix "HH:".
Outputs: <out_dir>/<short>/verify.json, e2_trials.csv, e2_pairs.csv, overlays/*.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import mech_lib as ML
import mech_e3 as E3

SEED = 0
SETS = ["all", "minute_hand", "hour_hand", "other"]
NOISE_SETS = ["all", "minute_hand"]
PIX_THR = 40


def e2_layers(L):
    return sorted(set([0] + [int(L * i / 8) for i in range(1, 8)]))


def circ(a, b):
    d = abs(a - b) % 12
    return min(d, 12 - d)


def build_pairs(n_pairs, seed=SEED):
    rng = np.random.RandomState(seed)
    cells = [(n, h) for n in range(1, 13) for h in range(1, 13)]
    rng.shuffle(cells)
    pairs = []
    for (n, h) in cells[:n_pairs]:
        nb = int(rng.choice([m for m in range(1, 13) if circ(m, n) >= 3]))
        hc = int(rng.choice([x for x in range(1, 13) if x != h]))
        pairs.append({"n_a": n, "h": h, "n_b": nb, "h_c": hc})
    return pairs


# ---------------------------------------------------------------------------- masks
_RENDER_CACHE = {}


def _render_gray(hour, minute, kind, tmp):
    from PIL import Image
    from clocks import generate_clock_image
    key = (hour, minute, kind)
    if key not in _RENDER_CACHE:
        p = os.path.join(tmp, f"r_{kind}_{hour}_{minute}.png")
        if not os.path.exists(p):
            kw = {}
            if kind == "hour_only":
                kw = dict(minute_hand_length=0.0, minute_hand_thickness=0.0)
            elif kind == "none":
                kw = dict(minute_hand_length=0.0, minute_hand_thickness=0.0, hour_hand_length=0.0, hour_hand_thickness=0.0)
            generate_clock_image(hour, minute, p, **kw)
        _RENDER_CACHE[key] = np.asarray(Image.open(p).convert("L")).astype(np.int16)
    return _RENDER_CACHE[key]


def hand_pixel_masks(h, m_a, m_b, tmp):
    """(minute-hand pixels of A union B, hour-hand pixels of A union B), (H,W) bool."""
    full_a, full_b = _render_gray(h, m_a, "full", tmp), _render_gray(h, m_b, "full", tmp)
    ho_a, ho_b = _render_gray(h, m_a, "hour_only", tmp), _render_gray(h, m_b, "hour_only", tmp)
    no_a, no_b = _render_gray(h, m_a, "none", tmp), _render_gray(h, m_b, "none", tmp)
    minute = (np.abs(full_a - ho_a) > PIX_THR) | (np.abs(full_b - ho_b) > PIX_THR)
    hour = (np.abs(ho_a - no_a) > PIX_THR) | (np.abs(ho_b - no_b) > PIX_THR)
    return minute, hour


def token_sets(h, m_a, m_b, n_img, tmp):
    grid = int(round(n_img ** 0.5))
    assert grid * grid == n_img, f"{n_img} image tokens is not a square grid"
    minute_px, hour_px = hand_pixel_masks(h, m_a, m_b, tmp)
    tm, th = ML.pixel_mask_to_token_mask(minute_px, grid), ML.pixel_mask_to_token_mask(hour_px, grid)
    return {"all": np.ones(n_img, dtype=bool), "minute_hand": tm, "hour_hand": th, "other": ~(tm | th)}, grid


def save_overlay(path, img_path, sets, grid):
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for i in range(grid * grid):
        r, c = divmod(i, grid)
        box = [c * W / grid, r * H / grid, (c + 1) * W / grid, (r + 1) * H / grid]
        if sets["minute_hand"][i]:
            dr.rectangle(box, fill=(255, 0, 0, 90), outline=(255, 0, 0, 200))
        elif sets["hour_hand"][i]:
            dr.rectangle(box, fill=(0, 0, 255, 90), outline=(0, 0, 255, 200))
    Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB").save(path)


# ---------------------------------------------------------------------------- helpers
def prompt_for(ctx, variant, n, h):
    if variant == "text_light":
        return ctx.TO.with_lead_in(ctx.TO.LEAD_INS[0], E3.TEXT_LIGHT.format(H=h))
    return ctx.CT.prompt_for(0, n, h)


def pair_inputs(ctx, pr, variant, img_dir):
    CT = ctx.CT
    n_a, h, n_b, h_c = pr["n_a"], pr["h"], pr["n_b"], pr["h_c"]
    pa, _, _ = CT.ensure_image("matching_clock", n_a, h, img_dir, CT.IMAGE_SEED)
    pb, _, _ = CT.ensure_image("matching_clock", n_b, h, img_dir, CT.IMAGE_SEED)
    pc, _, _ = CT.ensure_image("matching_clock", n_a, h_c, img_dir, CT.IMAGE_SEED)
    text = prompt_for(ctx, variant, n_a, h)
    ad = ctx.adapter
    ia, ib, ic = ad.build_inputs(pa, text), ad.build_inputs(pb, text), ad.build_inputs(pc, text)
    ma, mb, mc = (ad.image_token_positions(x) for x in (ia, ib, ic))
    assert ia["input_ids"].shape == ib["input_ids"].shape == ic["input_ids"].shape, "A/B/C prompts differ in length"
    assert bool((ma == mb).all() and (ma == mc).all()), "image-token positions differ between A, B and C"
    return {"A": ia, "B": ib, "C": ic}, ma, (pa, pb, pc)


@torch.no_grad()
def capture(ctx, inputs):
    with ML.capture_hidden(ctx.layers) as store:
        ctx.adapter.model(**inputs, use_cache=False)
    return store


def parse_minute(raw):
    return ML.parse_hhmm(raw)[1]


def evaluate(ctx, inputs, tf_ids):
    """generation + teacher-forced scores under whatever patch is currently active."""
    raw = ctx.gen_inputs(inputs, ctx.args.max_new_tokens)
    (prefix_ids, a_ids, b_ids) = tf_ids
    lp_a = ML.minute_logprob(ctx.adapter, inputs, prefix_ids, a_ids)
    lp_b = ML.minute_logprob(ctx.adapter, inputs, prefix_ids, b_ids)
    return raw, lp_b - lp_a


def pair_ids(ctx, pr):
    pre, a_ids = ML.cand_tokens(ctx.tok, ML.hour_prefix(pr["h"]), ML.minute_str(pr["n_a"]))
    _, b_ids = ML.cand_tokens(ctx.tok, ML.hour_prefix(pr["h"]), ML.minute_str(pr["n_b"]))
    return pre, a_ids, b_ids


# ---------------------------------------------------------------------------- verify
def part_verify(ctx):
    ad, L = ctx.adapter, ctx.L
    res = {"model": ctx.short, "n_layers": L, "checks": {}, "rep_penalty_arg": getattr(ctx.args, "rep_penalty", None),
           "checkpoint_default_repetition_penalty": getattr(getattr(ad.model, "generation_config", None), "repetition_penalty", None),
           "layers": e2_layers(L)}
    img_dir = os.path.join("e2_work", ctx.short, "images"); os.makedirs(img_dir, exist_ok=True)
    tmp = os.path.join("e2_work", ctx.short, "mask_renders"); os.makedirs(tmp, exist_ok=True)
    ov_dir = os.path.join(ctx.out, "overlays"); os.makedirs(ov_dir, exist_ok=True)
    n_check = 12 if ctx.smoke else 100
    pairs = build_pairs(max(n_check, 20))

    def check(name, ok, gating=True, **info):
        res["checks"][name] = {"pass": bool(ok), "gating": bool(gating), **info}
        print(f"  [{('PASS' if ok else 'FAIL') if gating else ('info:pass' if ok else 'info:FAIL')}] {name} {info}", flush=True)

    kmid = L // 2
    # V1 identity patch (source = A itself) changes 0 generations; V2 embedding-level all-token patch == showing image B
    try:
        changed, n1, agree_emb, n2, eff = 0, 0, 0, 0, 0
        rate_direct = rate_patch = 0
        for pr in pairs[:n_check]:
            ins, imask, _ = pair_inputs(ctx, pr, "standard", img_dir)
            pos = imask.nonzero().flatten().tolist()
            store_a, store_b = capture(ctx, ins["A"]), capture(ctx, ins["B"])
            base = ctx.gen_inputs(ins["A"], ctx.args.max_new_tokens)
            with ML.patch_hidden(ctx.layers, kmid, pos, store_a[kmid][0, pos]):
                idn = ctx.gen_inputs(ins["A"], ctx.args.max_new_tokens)
            changed += int(idn != base); n1 += 1
            direct = ctx.gen_inputs(ins["B"], ctx.args.max_new_tokens)
            with ML.patch_hidden(ctx.layers, 0, pos, store_b[0][0, pos]):
                emb = ctx.gen_inputs(ins["A"], ctx.args.max_new_tokens)
            mb = (pr["n_b"] % 12) * 5
            rate_direct += int(parse_minute(direct) == mb); rate_patch += int(parse_minute(emb) == mb)
            agree_emb += int(ML.parse_hhmm(direct) == ML.parse_hhmm(emb)); n2 += 1
        check("V1_identity_patch_changes_no_generation", changed == 0, changed=f"{changed}/{n1}")
        check("V2_embedding_all_token_patch_equals_showing_image_B", abs(rate_direct - rate_patch) / n2 <= 0.03 and agree_emb / n2 >= 0.95,
              follow_B_direct=round(rate_direct / n2, 3), follow_B_patched=round(rate_patch / n2, 3), identical_parsed_answers=f"{agree_emb}/{n2}",
              bound_points=3)
    except Exception as e:  # noqa: BLE001
        check("V1_V2_patch_machinery", False, error=E3._err(e))

    # V3 hook has an effect: real_B all-image patch at the middle layer moves log p(B)-log p(A) in most pairs
    try:
        moved, n3 = 0, 0
        for pr in pairs[:min(n_check, 30)]:
            ins, imask, _ = pair_inputs(ctx, pr, "standard", img_dir)
            pos = imask.nonzero().flatten().tolist()
            store_b = capture(ctx, ins["B"])
            ids = pair_ids(ctx, pr)
            base = ML.minute_logprob(ad, ins["A"], ids[0], ids[2]) - ML.minute_logprob(ad, ins["A"], ids[0], ids[1])
            with ML.patch_hidden(ctx.layers, kmid, pos, store_b[kmid][0, pos]):
                pat = ML.minute_logprob(ad, ins["A"], ids[0], ids[2]) - ML.minute_logprob(ad, ins["A"], ids[0], ids[1])
            moved += int(abs(pat - base) > 0.1); n3 += 1
        check("V3_patch_moves_teacher_forced_logprob", moved / n3 >= 0.5 or ctx.smoke, gating=not ctx.args.v3_informational, moved=f"{moved}/{n3}",
              note="needs >= 50% of pairs with |change| > 0.1 nat (a no-op patch would make every cell null); smoke (random tiny model) only logs it")
    except Exception as e:  # noqa: BLE001
        check("V3_patch_moves_teacher_forced_logprob", False, error=E3._err(e))

    # V4 region masks: automatic sanity + overlays for manual inspection (20 images)
    try:
        sizes, ok_all = [], True
        for i, pr in enumerate(pairs[:20]):
            ins, imask, (pa, pb, pc) = pair_inputs(ctx, pr, "standard", img_dir)
            sets, grid = token_sets(pr["h"], (pr["n_a"] % 12) * 5, (pr["n_b"] % 12) * 5, int(imask.sum()), tmp)
            sizes.append({k: int(v.sum()) for k, v in sets.items()})
            ok_all &= 0 < sets["minute_hand"].sum() < 0.5 * len(imask.nonzero()) and sets["hour_hand"].sum() > 0
            save_overlay(os.path.join(ov_dir, f"overlay_{i:02d}_A{pr['n_a']}_B{pr['n_b']}_H{pr['h']}.png"), pa, sets, grid)
        check("V4_region_masks_nonempty_and_small", ok_all, example_sizes=sizes[:3], grid=grid)
        check("V4b_overlays_written_MANUAL_INSPECTION_REQUIRED", True, gating=False, folder=ov_dir,
              note="red = minute-hand tokens, blue = hour-hand tokens (A and B drawn strokes); look at them before trusting any region result")
    except Exception as e:  # noqa: BLE001
        check("V4_region_masks_nonempty_and_small", False, error=E3._err(e))

    res["pass"] = all(c["pass"] for c in res["checks"].values() if c.get("gating", True))
    json.dump(res, open(os.path.join(ctx.out, "verify.json"), "w"), indent=1)
    print(f"E2 VERIFY GATE {'PASSED' if res['pass'] else 'FAILED'} for {ctx.short}", flush=True)
    return res["pass"]


# ---------------------------------------------------------------------------- run
def part_run(ctx):
    L = ctx.L
    layers_k = e2_layers(L)
    if ctx.smoke:
        layers_k = [0, L // 2]
    variants = [v for v in ctx.args.variants.split(",") if v]
    n_pairs = 3 if ctx.smoke else ctx.args.pairs
    pairs = build_pairs(n_pairs)
    img_dir = os.path.join("e2_work", ctx.short, "images"); os.makedirs(img_dir, exist_ok=True)
    tmp = os.path.join("e2_work", ctx.short, "mask_renders"); os.makedirs(tmp, exist_ok=True)
    path = os.path.join(ctx.out, "e2_trials.csv")
    frames, done = [], set()
    if os.path.exists(path):
        old = pd.read_csv(path)
        frames.append(old)
        done = set(zip(old.variant, old.pair))
    pd.DataFrame(pairs).to_csv(os.path.join(ctx.out, "e2_pairs.csv"), index_label="pair")
    rng = np.random.RandomState(SEED + 17)
    t_start = time.time()
    for variant in variants:
        for pi, pr in enumerate(pairs):
            if (variant, pi) in done:
                continue
            t0 = time.time()
            ins, imask, _ = pair_inputs(ctx, pr, variant, img_dir)
            pos_all = imask.nonzero().flatten().numpy()
            sets, grid = token_sets(pr["h"], (pr["n_a"] % 12) * 5, (pr["n_b"] % 12) * 5, len(pos_all), tmp)
            ids = pair_ids(ctx, pr)
            store_b, store_c, store_a = capture(ctx, ins["B"]), capture(ctx, ins["C"]), capture(ctx, ins["A"])
            ma, mb = (pr["n_a"] % 12) * 5, (pr["n_b"] % 12) * 5
            rows = []

            def rec(layer, sname, cond, raw, d, ntok):
                pm = parse_minute(raw)
                rows.append({"variant": variant, "pair": pi, "n_a": pr["n_a"], "n_b": pr["n_b"], "h": pr["h"], "layer": layer, "set": sname,
                             "cond": cond, "n_tokens": ntok, "raw": raw, "pred_minute": pm, "follow_B": pm == mb, "stays_A": pm == ma,
                             "tf_B_minus_A": d})
            raw0, d0 = evaluate(ctx, ins["A"], ids)
            rec(-1, "none", "baseline", raw0, d0, 0)
            for k in layers_k:
                for sname in SETS:
                    sel = sets[sname]
                    if sel.sum() == 0:
                        continue
                    pos = pos_all[sel]
                    for cond, src in (("real_B", store_b[k][0, pos]), ("same_minute_C", store_c[k][0, pos])):
                        with ML.patch_hidden(ctx.layers, k, pos, src):
                            raw, d = evaluate(ctx, ins["A"], ids)
                        rec(k, sname, cond, raw, d, int(sel.sum()))
                    if sname in NOISE_SETS:
                        noise = ML.make_matched_noise(store_a[k][0, pos].detach().cpu(), rng)
                        with ML.patch_hidden(ctx.layers, k, pos, noise):
                            raw, d = evaluate(ctx, ins["A"], ids)
                        rec(k, sname, "noise", raw, d, int(sel.sum()))
            frames.append(pd.DataFrame(rows))
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            print(f"  [{variant}] pair {pi + 1}/{len(pairs)} done in {time.time() - t0:.0f}s (total {(time.time() - t_start) / 60:.1f} min)", flush=True)
            if pi == 0 and not ctx.smoke:
                print(f"  >>> GUESS for this variant: {len(pairs) * (time.time() - t0) / 3600:.1f} h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--parts", default="verify,run")
    ap.add_argument("--out_dir", default=os.path.join("results", "mech_e2"))
    ap.add_argument("--prior_dir", default="prior")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--variants", default="standard")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--rep_penalty", type=float, default=None)
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--v3_informational", action="store_true", help="V3 logged but non-gating (descriptive near-ceiling models; deviations log 2026-11-08)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    from adapters import get_adapter
    adapter = get_adapter(model_id=args.model_id, adapter_name=args.adapter)
    adapter.load(args.model_id)
    ctx = E3.Ctx(adapter, args)
    print(f"mech_e2: model={ctx.short} layers={ctx.L} k-layers={e2_layers(ctx.L)} parts={parts} variants={args.variants} pairs={args.pairs} "
          f"smoke={args.smoke} rep_penalty={args.rep_penalty}", flush=True)
    with open(os.path.join(ctx.out, "decoding_rep_penalty.txt"), "a") as f:
        f.write(f"parts={parts} rep_penalty_arg={args.rep_penalty} checkpoint_default="
                f"{getattr(getattr(adapter.model, 'generation_config', None), 'repetition_penalty', None)}" + chr(10))
    if "verify" in parts:
        if not part_verify(ctx):
            print("GATE FAILED -> E2 run not started (see verify.json).", flush=True)
            return 3
    if "run" in parts:
        vj = os.path.join(ctx.out, "verify.json")
        if not args.no_gate and not (os.path.exists(vj) and json.load(open(vj)).get("pass")):
            print("run: skipped, no passing verify.json", flush=True)
            return 3
        part_run(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
