"""mech_e1.py -- E1: windowed attention knockout text -> image-token keys (PREREG_MECHANISM.md Section 5 + deviations log 2026-11-07, third entry).

ONE model per invocation. Parts (comma separated):
  verify  machinery gate (REQUIRED before `run` unless --no_gate)
  run     all E1 cells (resumable: finished cells are skipped when the output CSV already holds them)
  followup  H-E1-2: run the best matching_clock window (selected on matching_clock only) on mismatched_clock

Design (all as pre-registered / logged):
  * trials = the 432 conversion-test groups (lead-in, N, H); prompt = conversion_test.prompt_for; images = conversion_test.ensure_image
    (matching_clock, blank, mismatched_clock), identical to the conversion test.
  * query set B (primary) = every text position after the image (question text + generated tokens); query set A = the last prompt token +
    generated tokens. Image tokens are never blocked as queries. Keys blocked = the image-token positions. Blocking is applied in prefill
    AND in every cached decode step, on decoder layers [start, end).
  * cells: sliding 4-wide windows (28 layers -> 7, 36 -> 9), cumulative-from-L-to-end, all layers, early half, late half, final 4.
    matching_clock: all cells. blank / mismatched_clock: all, early, late, final4.   --scope all_only (7B/8B): intact, all, final4 only.
  * eager attention with fp32 accumulation inside the block context (mech_lib.eager_attention); intact cells use the model's default
    attention and an extra `intact_eager` cell measures the eager-vs-default difference.
  * greedy cached generation, --rep_penalty (1.0 headline), max_new_tokens 32.
Outputs: <out_dir>/<short>/verify.json, e1_trials.csv (long), e1_cell_summary.csv, e1_machinery.json.
USAGE (tiny model smoke test): python mech_e1.py --model_id <path> --adapter internvl3-2b --parts verify,run --smoke
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
COND_ORDER = ["matching_clock", "blank", "mismatched_clock"]
GATE_ABS = 0.5          # G8c form of the eager-vs-default check (deviations log 2026-11-07, third entry)


def layer_cells(L, cond, scope):
    """OrderedDict name -> (start, end) for one condition."""
    starts = list(range(0, L - 3, 4))
    cells = {}
    if scope == "all_only":
        cells["all"] = (0, L)
        cells["final4"] = (L - 4, L)
        return cells
    if cond == "matching_clock":
        for s in starts:
            cells[f"w{s}-{s + 4}"] = (s, s + 4)
        for s in starts:
            cells[f"cum{s}-end"] = (s, L)
    cells["all"] = (0, L)
    cells["early"] = (0, L // 2)
    cells["late"] = (L // 2, L)
    cells["final4"] = (L - 4, L)
    return cells


def plan_cells(L, scope, qsets, smoke):
    """Ordered list of (cond, qset, cell_name, layer_range or None) -- most important first (query set B, matching clock)."""
    plan = [("matching_clock", "-", "intact", None), ("matching_clock", "-", "intact_eager", "eager_only"),
            ("blank", "-", "intact", None), ("mismatched_clock", "-", "intact", None)]
    for q in qsets:
        for cond in COND_ORDER:
            if scope == "all_only" and cond != "matching_clock":
                cells = {"all": (0, L)}
            else:
                cells = layer_cells(L, cond, scope)
            for name, rng in cells.items():
                plan.append((cond, q, name, rng))
    if smoke:
        keep = {"intact", "intact_eager", "all", "final4"}
        plan = [p for p in plan if p[2] in keep]
    return plan


def get_trials(ctx):
    t = ctx.CT.build_groups()
    if ctx.smoke:
        t = t[::max(1, len(t) // 6)][:6]
    return t


def images_dir(ctx):
    d = "e1_images_smoke" if ctx.smoke else "e1_images"
    os.makedirs(d, exist_ok=True)
    return d


def one_trial(ctx, st, cond, qset, rng, lead, n, h):
    CT = ctx.CT
    path, ih, im = CT.ensure_image(cond, n, h, images_dir(ctx), CT.IMAGE_SEED)
    inputs = ctx.adapter.build_inputs(path, CT.prompt_for(lead, n, h))
    imask = ctx.adapter.image_token_positions(inputs)
    if rng is None:
        raw = ctx.gen_inputs(inputs, ctx.args.max_new_tokens)
    elif rng == "eager_only":
        with ML.eager_attention(ctx.adapter):
            raw = ctx.gen_inputs(inputs, ctx.args.max_new_tokens)
    else:
        st.set(imask, qset, active=True)
        with ML.block_image_attention(ctx.adapter, rng, st):
            raw = ctx.gen_inputs(inputs, ctx.args.max_new_tokens)
        st.active = False
    truth = (n % 12) * 5
    lab, ph, pm = ML.label_answer(raw, h, truth, image_minute=im)
    return {"lead": lead, "N": n, "H": h, "raw": raw, "label": lab, "pred_hour": ph, "pred_minute": pm,
            "minute_correct": lab == "correct", "follows_image": lab == "follows_image", "leak": lab == "leak",
            "unparsed": lab == "unparsed", "image_hour": ih, "image_minute": im, "n_image_tokens": int(imask.sum())}


# ----------------------------------------------------------------------------------------------------------
# verify gate
# ----------------------------------------------------------------------------------------------------------

def part_verify(ctx):
    ad, L = ctx.adapter, ctx.L
    st = ML.ImageBlockState()
    res = {"model": ctx.short, "n_layers": L, "checks": {}, "rep_penalty_arg": getattr(ctx.args, "rep_penalty", None),
           "checkpoint_default_repetition_penalty": getattr(getattr(ad.model, "generation_config", None), "repetition_penalty", None)}
    trials = get_trials(ctx)
    rng = np.random.default_rng(SEED)
    pick = [trials[i] for i in rng.choice(len(trials), size=min(20, len(trials)), replace=False)]

    def check(name, ok, **info):
        res["checks"][name] = {"pass": bool(ok), "gating": True, **info}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

    def prep(trial, cond="matching_clock"):
        lead, n, h = trial
        path, ih, im = ctx.CT.ensure_image(cond, n, h, images_dir(ctx), ctx.CT.IMAGE_SEED)
        inputs = ad.build_inputs(path, ctx.CT.prompt_for(lead, n, h))
        return inputs, ad.image_token_positions(inputs)

    try:   # G1 inputs / image mask sane
        inputs, imask = prep(pick[0])
        P = inputs["input_ids"].shape[1]
        idx = imask.nonzero().flatten()
        check("G1_image_mask_sane", int(imask.sum()) > 0 and imask.shape[0] == P and bool((idx.diff() == 1).all()) and int(imask.sum()) < P,
              n_image_tokens=int(imask.sum()), prompt_len=P, contiguous=bool((idx.diff() == 1).all()))
    except Exception as e:  # noqa: BLE001
        check("G1_image_mask_sane", False, error=E3._err(e))

    try:   # G2 hook reaches every decoder layer in prefill, and fires in cached decode steps
        inputs, imask = prep(pick[0])
        st.set(imask, "B", active=True)
        with ML.block_image_attention(ad, (0, L), st):
            with torch.no_grad():
                ad.model(**inputs, use_cache=False)
            prefill_calls = st.calls
            st.calls = st.decode_calls = 0
            ctx.gen_inputs(inputs, 8)
            dec = st.decode_calls
        check("G2_hook_reaches_every_layer_prefill_and_decode", prefill_calls >= L and dec > 0, prefill_calls=prefill_calls, n_layers=L, decode_calls=dec)
    except Exception as e:  # noqa: BLE001
        check("G2_hook_reaches_every_layer_prefill_and_decode", False, error=E3._err(e))

    try:   # G3 empty block set == no hook (same eager path), exact generations
        same, n3 = 0, 0
        for t in pick[:10]:
            inputs, imask = prep(t)
            with ML.eager_attention(ad):
                a = ctx.gen_inputs(inputs, ctx.args.max_new_tokens)
            st.set(imask, "B", active=False)
            with ML.block_image_attention(ad, (0, L), st):
                b = ctx.gen_inputs(inputs, ctx.args.max_new_tokens)
            same += int(a == b); n3 += 1
        check("G3_empty_block_set_reproduces_intact_generations", same == n3, matches=f"{same}/{n3}")
    except Exception as e:  # noqa: BLE001
        check("G3_empty_block_set_reproduces_intact_generations", False, error=E3._err(e))

    try:   # G4 eager (fp32-accumulating) vs default attention: G8c form (replaces the 1e-3 rule, deviations log)
        fin, worst, agree, n4 = True, 0.0, 0, 0
        for t in pick[:10]:
            inputs, _ = prep(t)
            with torch.no_grad():
                ld = ad.model(**inputs, use_cache=False).logits[0, -1].float().cpu()
                with ML.eager_attention(ad):
                    le = ad.model(**inputs, use_cache=False).logits[0, -1].float().cpu()
            ok = bool(torch.isfinite(ld).all() and torch.isfinite(le).all())
            fin &= ok
            if ok:
                worst = max(worst, float((torch.log_softmax(ld, -1) - torch.log_softmax(le, -1)).abs().max()))
                agree += int(ld.argmax() == le.argmax())
            n4 += 1
        check("G4_eager_vs_default_finite_close_argmax", fin and worst < GATE_ABS and agree >= 0.9 * n4, finite=fin,
              max_abs_logprob_diff=round(worst, 4), argmax_equal=f"{agree}/{n4}", bound_nat=GATE_ABS)
    except Exception as e:  # noqa: BLE001
        check("G4_eager_vs_default_finite_close_argmax", False, error=E3._err(e))

    try:   # G5 the block does something: all-layer block changes generations (both query sets)
        for q in ("B", "A"):
            ch, n5 = 0, 0
            for t in pick[:12]:
                inputs, imask = prep(t)
                with ML.eager_attention(ad):
                    a = ctx.gen_inputs(inputs, ctx.args.max_new_tokens)
                st.set(imask, q, active=True)
                with ML.block_image_attention(ad, (0, L), st):
                    b = ctx.gen_inputs(inputs, ctx.args.max_new_tokens)
                st.active = False
                ch += int(a != b); n5 += 1
            check(f"G5_block_changes_generations_query_set_{q}", ch / n5 >= 0.3 or ctx.smoke, changed=f"{ch}/{n5}",
                  note="needs >= 30% (a no-op block would silently make every window look null); smoke only logs it")
    except Exception as e:  # noqa: BLE001
        check("G5_block_changes_generations", False, error=E3._err(e))

    res["pass"] = all(c["pass"] for c in res["checks"].values())
    json.dump(res, open(os.path.join(ctx.out, "verify.json"), "w"), indent=1)
    print(f"E1 VERIFY GATE {'PASSED' if res['pass'] else 'FAILED'} for {ctx.short}", flush=True)
    return res["pass"]


# ----------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------

def summarize(ctx):
    p = os.path.join(ctx.out, "e1_trials.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    rows = []
    for (cond, q, cell), g in d.groupby(["cond", "qset", "cell"], sort=False):
        rows.append({"cond": cond, "qset": q, "cell": cell, "start": g.start.iloc[0], "end": g["end"].iloc[0], "n": len(g),
                     "minute_acc": g.minute_correct.mean(), "follows_image_rate": g.follows_image.mean(), "leak_rate": g.leak.mean(),
                     "unparsed_rate": g.unparsed.mean()})
    s = pd.DataFrame(rows)
    ctx.save(s, "e1_cell_summary.csv")
    # machinery check (NOT a hypothesis test): blocked-all matching_clock vs the model's own text-only run, same 432 trials
    mc = {}
    prior = ctx.prior
    for q in sorted(d.qset.unique()):
        if q == "-":
            continue
        a = d[(d.cond == "matching_clock") & (d.qset == q) & (d.cell == "all")]
        if len(a) == 0:
            continue
        if prior is not None:
            t = prior.copy()
            keys = set(zip(a.lead, a.N, a.H))
            t = t[[(int(l), int(n), int(h)) in keys for l, n, h in zip(t.lead_in_id, t.position, t.hour)]]
            text_acc = float(t.minute_correct.astype(bool).mean()) if len(t) else float("nan")
        else:
            text_acc = float("nan")
        blocked = float(a.minute_correct.mean())
        mc[q] = {"n": int(len(a)), "blocked_all_matching_acc": blocked, "text_only_acc": text_acc,
                 "abs_diff_points": abs(blocked - text_acc) * 100 if text_acc == text_acc else None,
                 "knockout_valid": bool(text_acc == text_acc and abs(blocked - text_acc) * 100 <= 5.0)}
    json.dump(mc, open(os.path.join(ctx.out, "e1_machinery.json"), "w"), indent=1)
    print("  machinery check (blocked-all vs text-only, tolerance 5 pts):", mc, flush=True)


def part_run(ctx):
    L = ctx.L
    st = ML.ImageBlockState()
    qsets = [q for q in ctx.args.qsets.split(",") if q]
    plan = plan_cells(L, ctx.args.scope, qsets, ctx.smoke)
    trials = get_trials(ctx)
    path = os.path.join(ctx.out, "e1_trials.csv")
    done = set()
    frames = []
    if os.path.exists(path):
        old = pd.read_csv(path)
        frames.append(old)
        done = set(zip(old.cond, old.qset, old.cell))
    print(f"E1 run: {len(plan)} cells x {len(trials)} trials; {len(done)} cells already done", flush=True)
    t_start = time.time()
    for k, (cond, q, name, rng) in enumerate(plan):
        if (cond, q, name) in done:
            continue
        t0 = time.time()
        rows = []
        for (lead, n, h) in trials:
            r = one_trial(ctx, st, cond, q, rng, lead, n, h)
            r.update({"cond": cond, "qset": q, "cell": name,
                      "start": -1 if rng in (None, "eager_only") else rng[0], "end": -1 if rng in (None, "eager_only") else rng[1]})
            rows.append(r)
        new = pd.DataFrame(rows)
        frames.append(new)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"  [{k + 1}/{len(plan)}] {cond:<17} qset {q} cell {name:<12} acc {new.minute_correct.mean():.3f} "
              f"follows_image {new.follows_image.mean():.3f}  ({time.time() - t0:.0f}s; total {(time.time() - t_start) / 60:.1f} min)", flush=True)
        if k == 0 and not ctx.smoke:
            print(f"  >>> time per cell ~{time.time() - t0:.0f}s (first cell; blocked cells are slower). GUESS for all {len(plan)} cells: "
                  f"{len(plan) * (time.time() - t0) * 1.4 / 3600:.1f} h", flush=True)
    summarize(ctx)


def part_followup(ctx):
    """H-E1-2 needs the window with the largest matching_clock drop (selected on matching_clock ONLY) to be run on mismatched_clock (different
    images/trials). Those cells are not in the main plan because the window is data-dependent: select it here and run it. Resumable."""
    path = os.path.join(ctx.out, "e1_trials.csv")
    d = pd.read_csv(path)
    st = ML.ImageBlockState()
    trials = get_trials(ctx)
    sel = {}
    for q in sorted(x for x in d.qset.unique() if x != "-"):
        base = d[(d.cond == "matching_clock") & (d.cell == "intact")]
        acc0 = base.minute_correct.mean()
        best = None
        for name, g in d[(d.cond == "matching_clock") & (d.qset == q) & d.cell.str.startswith("w")].groupby("cell"):
            drop = acc0 - g.minute_correct.mean()
            if best is None or drop > best[1]:
                best = (name, drop, int(g.start.iloc[0]), int(g["end"].iloc[0]))
        if best is None:
            continue
        sel[q] = {"window": best[0], "matching_drop": float(best[1]), "start": best[2], "end": best[3]}
        done = set(zip(d.cond, d.qset, d.cell))
        if ("mismatched_clock", q, best[0]) in done:
            continue
        rows = []
        for (lead, n, h) in trials:
            r = one_trial(ctx, st, "mismatched_clock", q, (best[2], best[3]), lead, n, h)
            r.update({"cond": "mismatched_clock", "qset": q, "cell": best[0], "start": best[2], "end": best[3]})
            rows.append(r)
        d = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)
        d.to_csv(path, index=False)
        print(f"  follow-up qset {q}: best matching window {best[0]} (drop {best[1] * 100:.1f} pts) -> mismatched_clock follows_image "
              f"{pd.DataFrame(rows).follows_image.mean():.3f}", flush=True)
    json.dump(sel, open(os.path.join(ctx.out, "e1_followup.json"), "w"), indent=1)
    summarize(ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--parts", default="verify,run")
    ap.add_argument("--out_dir", default=os.path.join("results", "mech_e1"))
    ap.add_argument("--prior_dir", default="prior")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--qsets", default="B,A")
    ap.add_argument("--scope", default="full", choices=["full", "all_only"])
    ap.add_argument("--rep_penalty", type=float, default=None)
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    from adapters import get_adapter
    adapter = get_adapter(model_id=args.model_id, adapter_name=args.adapter)
    adapter.load(args.model_id)
    ctx = E3.Ctx(adapter, args)
    print(f"mech_e1: model={ctx.short} layers={ctx.L} parts={parts} scope={args.scope} qsets={args.qsets} smoke={args.smoke} "
          f"rep_penalty={args.rep_penalty}", flush=True)
    with open(os.path.join(ctx.out, "decoding_rep_penalty.txt"), "a") as f:
        f.write(f"parts={parts} rep_penalty_arg={args.rep_penalty} checkpoint_default="
                f"{getattr(getattr(adapter.model, 'generation_config', None), 'repetition_penalty', None)}" + chr(10))
    if "verify" in parts:
        if not part_verify(ctx):
            print("GATE FAILED -> E1 run not started (see verify.json).", flush=True)
            return 3
    if "run" in parts:
        vj = os.path.join(ctx.out, "verify.json")
        if not args.no_gate and not (os.path.exists(vj) and json.load(open(vj)).get("pass")):
            print("run: skipped, no passing verify.json", flush=True)
            return 3
        part_run(ctx)
    if "followup" in parts:
        part_followup(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
