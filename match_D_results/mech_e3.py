"""mech_e3.py -- E3 (text-side analysis) + text-light pilot, as pre-registered in PREREG_MECHANISM.md.

ONE model per invocation (one load). Parts (comma separated, run in this order):
  verify  machinery gate (REQUIRED before any other part unless --no_gate; a failed gate stops the run, exit code 3)
  a       E3a logit lens (432 trials, 12 candidate minutes, teacher-forced gold hour prefix "HH:")
  b       E3b swap patching  P(N,H) <- residuals of P(H,N), equal-token-length pairs, per layer x position group
  c       E3c restoration pairs (same N, other H; right run patched into wrong run), uses prior free-generation outputs
  d       E3d edge knockout  (minute-slot queries -> H-token keys); --d_mode full (3B, 2B) | min (7B: intact, all layers, random span)
  e       E3e description transplant (text-only, the model's own correct description values)
  f       E3f H=12: word / numeral / o'clock forms of H and N, hour-slot vs minute-slot error types
  p       text-light pilot: hour in text, minute only in the image (matching clock) vs blank image
  analyze re-compute every summary table from the saved CSVs (no model needed; use --parts analyze alone)

Outputs: <out_dir>/<short_name>/*.csv (+ verify.json). Prior outputs needed by c, a(label), e: --prior_dir/<short>/
conversion_test.csv and describe.csv (fallback: results/<short>/).
Everything is greedy / deterministic; seeds fixed (SEED=0).

USAGE (smoke test on a tiny model):
  python mech_e3.py --model_id <path> --adapter internvl3-2b --parts verify,a,b,c,d,e,f,p --smoke
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import mech_lib as ML

SEED = 0
LAYER_STEP = 3
TIMEOUT_NOTE = "all timings in the log are measurements of THIS run; planning numbers elsewhere are guesses"


# ----------------------------------------------------------------------------------------------------------
# context
# ----------------------------------------------------------------------------------------------------------

class Ctx:
    def __init__(self, adapter, args):
        import conversion_test as CT
        import text_only_v2 as TO
        self.CT, self.TO = CT, TO
        self.adapter, self.args = adapter, args
        self.tok = adapter.processor.tokenizer
        self.layers = adapter.decoder_layers()
        self.L = len(self.layers)
        self.short = adapter.short_name
        self.out = os.path.join(args.out_dir, self.short)
        os.makedirs(self.out, exist_ok=True)
        self._pc = {}
        self.smoke = args.smoke
        self.layers_sub = [0, self.L // 2] if self.smoke else list(range(0, self.L - 1, LAYER_STEP))
        self.prior = self._load_prior("conversion_test.csv")
        self.desc = self._load_prior("describe.csv", full=True)

    def _load_prior(self, fname, full=False):
        for d in (os.path.join(self.args.prior_dir, self.short), os.path.join("results", self.short),
                  os.path.join("qwen_prior", "results", self.short)):
            p = os.path.join(d, fname)
            if os.path.exists(p):
                df = pd.read_csv(p)
                if not full:
                    df = df[df.condition == "text_only"].copy()
                    df["parse_ok"] = df.parse_ok.astype(bool)
                    df["minute_correct"] = df.minute_correct.astype(bool)
                return df
        return None

    def trials(self):
        t = self.CT.build_groups()                       # (lead, n, h)
        if self.smoke:
            t = t[::max(1, len(t) // 12)][:12]
        return t

    def prompt(self, lead, n, h):
        k = (lead, n, h)
        if k not in self._pc:
            self._pc[k] = ML.TextPrompt(self.adapter, self.CT.prompt_for(lead, n, h))
        return self._pc[k]

    def raw_prompt(self, text, spans=False):
        if text not in self._pc:
            self._pc[text] = ML.TextPrompt(self.adapter, text, find_spans=spans)
        return self._pc[text]

    def ntok(self, n):
        return len(self.tok(" " + str(n), add_special_tokens=False)["input_ids"])

    def gen(self, text, max_new=None):
        inputs = self.adapter.build_inputs(None, text)
        return self.adapter.generate_answer(inputs, max_new_tokens=max_new or self.args.max_new_tokens)

    def save(self, df, name):
        p = os.path.join(self.out, name)
        df.to_csv(p, index=False)
        print(f"  wrote {p} ({len(df)} rows)", flush=True)


def cands_for(ctx, h, minutes):
    pre = None
    cs = []
    for m in minutes:
        p, c = ML.cand_tokens(ctx.tok, ML.hour_prefix(h), m)
        pre = p
        cs.append(c)
    return pre, cs


ALL_MINUTES = [f"{m:02d}" for m in range(0, 60, 5)]


# ----------------------------------------------------------------------------------------------------------
# verify gate
# ----------------------------------------------------------------------------------------------------------

def part_verify(ctx):
    ad, tok, L = ctx.adapter, ctx.tok, ctx.L
    rng = np.random.default_rng(SEED)
    res = {"model": ctx.short, "n_layers": L, "checks": {}}
    trials = ctx.CT.build_groups()
    pick = [trials[i] for i in rng.choice(len(trials), size=min(100, len(trials)), replace=False)]
    if ctx.smoke:
        pick = pick[:12]

    def check(name, ok, **info):
        res["checks"][name] = {"pass": bool(ok), **info}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

    # G1 tokenisation / span detection
    try:
        for h in range(1, 13):
            for m in ALL_MINUTES:
                ML.cand_tokens(tok, ML.hour_prefix(h), m)
        for (lead, n, h) in pick[:30] + [(0, 12, 12), (1, 10, 11), (2, 9, 12)]:
            sp = ctx.prompt(lead, n, h).spans
            assert sp and all(len(sp[k]) > 0 for k in ("N", "H", "between", "last")), sp
        check("G1_tokens_and_spans", True)
    except Exception as e:  # noqa: BLE001
        check("G1_tokens_and_spans", False, error=str(e)[:200])

    # G2 teacher-forced argmax vs free generation (correctness agreement)
    agree, n_tf, n_free = 0, 0, 0
    for (lead, n, h) in pick:
        tp = ctx.prompt(lead, n, h)
        txt = ctx.gen(tp.prompt, 32)
        _, m = ML.parse_hhmm(txt)
        free_ok = (m == (n % 12) * 5)
        pre, cs = cands_for(ctx, h, ALL_MINUTES)
        lp = ML.score_candidates(ad, tp.ids, pre, cs)
        tf_ok = (int(lp.argmax()) == (n % 12))
        agree += int(free_ok == tf_ok); n_tf += int(tf_ok); n_free += int(free_ok)
    check("G2_teacher_forced_vs_free_generation", (agree / len(pick) >= 0.95) or ctx.smoke, agreement=round(agree / len(pick), 3),
          tf_acc=round(n_tf / len(pick), 3), free_acc=round(n_free / len(pick), 3), n=len(pick),
          note="threshold 0.95; --smoke (random-weight test model) only logs it" if ctx.smoke else "threshold 0.95")

    # G3 identity patch + G4 logit lens vs real log-probs
    max_id, max_lens = 0.0, 0.0
    for (lead, n, h) in pick[:10]:
        tp = ctx.prompt(lead, n, h)
        pre, cs = cands_for(ctx, h, [ML.minute_str(n), ML.minute_str(h)])
        lp0, store, q = ML.score_candidates(ad, tp.ids, pre, cs, return_resid=True)
        src = [s[0] for s in store]
        for li in sorted({0, L // 2, max(L - 2, 0)}):
            with ML.patch_residuals(ctx.layers, [li], list(range(tp.P)), src):
                lp1 = ML.score_candidates(ad, tp.ids, pre, cs)
            max_id = max(max_id, float((lp1 - lp0).abs().max()))
        lens = ML.lens_logprobs(ad, [s[0] for s in store], q, cs[0])
        max_lens = max(max_lens, abs(float(lens[-1]) - float(lp0[0])))
    check("G3_identity_patch_no_change", max_id < 1e-3, max_abs_diff_nats=max_id)
    check("G4_logit_lens_final_layer_matches_model", max_lens < 0.05, max_abs_diff_nats=max_lens)

    # G5 empty block == no hook (same eager path), G8 eager vs default, adapter generate agreement
    st = ML.BlockState()
    exact, eag_default, vs_adapter, n5 = 0, 0, 0, 0
    sub = pick[:50] if not ctx.smoke else pick[:6]
    for (lead, n, h) in sub:
        tp = ctx.prompt(lead, n, h)
        with ML.eager_attention(ad):
            a = ML.greedy_generate_blocked(ad, tp.ids, st, key_pos=None)
        with ML.block_attention(ad, (0, L), st):
            b = ML.greedy_generate_blocked(ad, tp.ids, st, key_pos=[])
        exact += int(a == b)
        # default attention path (no cache loop) vs eager
        c = ML.greedy_generate_blocked(ad, tp.ids, st, key_pos=None)
        eag_default += int(ML.parse_hhmm(a) == ML.parse_hhmm(c))
        d = ctx.gen(tp.prompt, 32)
        vs_adapter += int(ML.parse_hhmm(c) == ML.parse_hhmm(d))
        n5 += 1
    check("G5_empty_block_equals_no_hook", exact == n5, matches=f"{exact}/{n5}")
    check("G8_eager_vs_default_attention_agree", eag_default / n5 >= 0.95, agreement=round(eag_default / n5, 3))
    check("G8b_nocache_loop_vs_adapter_generate_agree", vs_adapter / n5 >= 0.95, agreement=round(vs_adapter / n5, 3),
          note="parsed HH:MM equality; fp16 + different code path, so a small disagreement is expected")

    # G6 guard
    try:
        tp = ctx.prompt(*pick[0])
        ML.greedy_generate_blocked(ad, tp.ids, st, key_pos=list(range(tp.P)), max_new=2)
        check("G6_guard_rejects_full_prompt_block", False)
    except ValueError:
        check("G6_guard_rejects_full_prompt_block", True)

    # G7 hook reach and effect (teacher-forced, so it does not depend on what the model would generate):
    # blocking the N-span from the colon position on, at all layers, must (a) fire once per layer and (b) change the
    # candidate log-probs; blocking an EMPTY span must change nothing.
    changed, fired, n7, empty_diff = 0, 0, 0, 0.0
    for (lead, n, h) in pick[:20]:
        tp = ctx.prompt(lead, n, h)
        pre, cs = cands_for(ctx, h, [ML.minute_str(n), ML.minute_str(h)])
        with ML.eager_attention(ad):
            lp0 = ML.score_candidates(ad, tp.ids, pre, cs)
        q = tp.P + len(pre) - 1
        for tag, key in (("N", tp.spans["N"]), ("empty", [])):
            st.q_start, st.key_pos, st.calls = q, torch.as_tensor(key, dtype=torch.long), 0
            with ML.block_attention(ad, (0, L), st):
                lp1 = ML.score_candidates(ad, tp.ids, pre, cs)
            if tag == "N":
                fired = max(fired, st.calls)
                changed += int(float((lp1 - lp0).abs().max()) > 0.01)
            else:
                empty_diff = max(empty_diff, float((lp1 - lp0).abs().max()))
        st.q_start = None
        n7 += 1
    check("G7_hooks_fire_on_every_layer", fired >= L, max_calls_per_forward=fired, n_layers=L)
    check("G7b_blocking_N_span_changes_logprobs", changed / n7 >= 0.5, changed=f"{changed}/{n7}")
    check("G7c_empty_block_changes_nothing", empty_diff < 1e-3, max_abs_diff_nats=empty_diff)

    # G9 the block really activates inside free generation (needs the model to produce ':'), H-span, all layers
    act, n9 = 0, 0
    for (lead, n, h) in pick[:20]:
        tp = ctx.prompt(lead, n, h)
        st.calls = 0
        with ML.block_attention(ad, (0, L), st):
            ML.greedy_generate_blocked(ad, tp.ids, st, key_pos=tp.spans["H"], max_new=24)
        act += int(st.calls > 0)
        n9 += 1
    check("G9_block_activates_during_generation", (act / n9 >= 0.9) or ctx.smoke, activated=f"{act}/{n9}",
          note="needs >=90% (a model that never writes ':' would silently get no knockout); smoke only logs it")

    res["pass"] = all(c["pass"] for c in res["checks"].values())
    with open(os.path.join(ctx.out, "verify.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"VERIFY GATE {'PASSED' if res['pass'] else 'FAILED'} for {ctx.short}", flush=True)
    return res["pass"]


def gate_ok(ctx):
    p = os.path.join(ctx.out, "verify.json")
    return os.path.exists(p) and json.load(open(p)).get("pass", False)


# ----------------------------------------------------------------------------------------------------------
# E3a logit lens
# ----------------------------------------------------------------------------------------------------------

def prior_label(ctx, lead, n, h):
    """'right' / 'leak' / 'wrong_other' / None from the prior free-generation text-only output."""
    if ctx.prior is None:
        return None
    r = ctx.prior[(ctx.prior.lead_in_id == lead) & (ctx.prior.position == n) & (ctx.prior.hour == h)]
    if len(r) == 0:
        return None
    r = r.iloc[0]
    if r.minute_correct:
        return "right"
    if not r.parse_ok:
        return "unparsed"
    lab, _, _ = ML.label_answer(r.raw_output, h, (n % 12) * 5)
    return "wrong_leak" if lab == "leak" else "wrong_other"


def part_a(ctx):
    ad = ctx.adapter
    rows = []
    t0 = time.time()
    for k, (lead, n, h) in enumerate(ctx.trials()):
        tp = ctx.prompt(lead, n, h)
        pre, cs = cands_for(ctx, h, ALL_MINUTES)
        lp, store, q = ML.score_candidates(ad, tp.ids, pre, cs, return_resid=True)
        gi, li = n % 12, h % 12                      # candidate index = minute // 5
        row = {"lead": lead, "N": n, "H": h, "label": prior_label(ctx, lead, n, h), "tf_argmax_idx": int(lp.argmax()),
               "tf_correct": int(lp.argmax()) == gi, "gold_lp": float(lp[gi]), "leak_lp": float(lp[li]) if li != gi else np.nan}
        lg = ML.lens_logprobs(ad, [s[gi] for s in store], q, cs[gi])
        ll = ML.lens_logprobs(ad, [s[li] for s in store], q, cs[li]) if li != gi else np.full(ctx.L, np.nan, np.float32)
        for l in range(ctx.L):
            row[f"gold_L{l}"], row[f"leak_L{l}"] = float(lg[l]), float(ll[l])
        rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"  E3a {k + 1}/{len(ctx.trials())}  {time.time() - t0:.0f}s", flush=True)
    ctx.save(pd.DataFrame(rows), "e3a_lens.csv")


def summarize_a(ctx):
    p = os.path.join(ctx.out, "e3a_lens.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d.N != d.H]
    d["cell"] = d.N * 100 + d.H
    groups = {"all": d, "right": d[d.label == "right"], "wrong_leak": d[d.label == "wrong_leak"],
              "wrong_other": d[d.label == "wrong_other"], "H12": d[d.H == 12], "H_not12": d[d.H != 12]}
    out = []
    for g, dd in groups.items():
        if len(dd) < 3:
            continue
        for l in range(ctx.L):
            diff = (dd[f"gold_L{l}"] - dd[f"leak_L{l}"]).values
            m, lo, hi = ML.cluster_bootstrap_ci(diff, dd.cell.values, n_boot=2000 if ctx.smoke else 10000)
            out.append({"group": g, "layer": l, "n": len(dd), "mean_gold_minus_leak": m, "ci_lo": lo, "ci_hi": hi,
                        "frac_gold_ahead": float((diff > 0).mean())})
    ctx.save(pd.DataFrame(out), "e3a_summary.csv")


# ----------------------------------------------------------------------------------------------------------
# E3b swap patching / E3c restoration
# ----------------------------------------------------------------------------------------------------------

def rand_positions(spans, n, seed):
    pool = spans["question"]
    if len(pool) < n:
        return None
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def source_residuals(ctx, tp):
    with ML.capture_residuals(ctx.layers) as st:
        with torch.no_grad():
            ML.forward_logits(ctx.adapter, tp.ids)
    return [s[0] for s in st]


def part_b(ctx):
    ad = ctx.adapter
    pairs = [(lead, n, h) for lead in range(ctx.CT.N_LEAD_INS) for n in range(1, 13) for h in range(1, 13)
             if n != h and ctx.ntok(n) == ctx.ntok(h)]
    if ctx.smoke:
        pairs = pairs[::max(1, len(pairs) // 6)][:6]
    rows = []
    t0 = time.time()
    for k, (lead, n, h) in enumerate(pairs):
        tgt, src = ctx.prompt(lead, n, h), ctx.prompt(lead, h, n)
        assert tgt.P == src.P and all(len(tgt.spans[g]) == len(src.spans[g]) for g in ("N", "H", "between"))
        pre, cs = cands_for_pair(ctx, h, n)
        lp0 = ML.score_candidates(ad, tgt.ids, pre, cs)
        d0 = float(lp0[1] - lp0[0])
        src_res = source_residuals(ctx, src)
        tgt_res = source_residuals(ctx, tgt)
        sp = tgt.spans
        groups = {"N": sp["N"], "H": sp["H"], "between": sp["between"], "last": sp["last"]}
        union = sorted(set(sp["H"]) | set(sp["between"]) | set(sp["last"]))

        def rec(kind, layer, group, lp):
            rows.append({"lead": lead, "N": n, "H": h, "kind": kind, "layer": layer, "group": group,
                         "lp_tgt": float(lp[0]), "lp_src": float(lp[1]), "d": float(lp[1] - lp[0]), "d0": d0,
                         "shift": float(lp[1] - lp[0]) - d0, "dtgt": float(lp[0] - lp0[0])})

        for l in ctx.layers_sub:
            for g, pos in groups.items():
                with ML.patch_residuals(ctx.layers, [l], pos, src_res):
                    rec("swap", l, g, ML.score_candidates(ad, tgt.ids, pre, cs))
            rp = rand_positions(sp, len(sp["N"]), hash((lead, n, h, l)) % (2 ** 31))
            if rp is not None:
                with ML.patch_residuals(ctx.layers, [l], rp, src_res):
                    rec("random", l, "random", ML.score_candidates(ad, tgt.ids, pre, cs))
        for l in ctx.layers_sub[::4] or ctx.layers_sub[:1]:
            with ML.patch_residuals(ctx.layers, [l], list(range(tgt.P)), tgt_res):
                rec("identity", l, "all", ML.score_candidates(ad, tgt.ids, pre, cs))
        # same-answer source: P(N,H2), H2 != H, N with equal token length -> same minute answer
        h2 = next((x for x in range(1, 13) if x not in (h, n) and ctx.ntok(x) == ctx.ntok(h)), None)
        if h2 is not None:
            s2 = ctx.prompt(lead, n, h2)
            if s2.P == tgt.P:
                s2_res = source_residuals(ctx, s2)
                for l in ctx.layers_sub:
                    with ML.patch_residuals(ctx.layers, [l], union, s2_res):
                        rec("same_answer", l, "H+between+last", ML.score_candidates(ad, tgt.ids, pre, cs))
        if (k + 1) % 25 == 0:
            print(f"  E3b {k + 1}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    ctx.save(pd.DataFrame(rows), "e3b_swap.csv")


def cands_for_pair(ctx, h, n):
    pre, cs = cands_for(ctx, h, [ML.minute_str(n), ML.minute_str(h)])
    return pre, cs


def summarize_b(ctx):
    p = os.path.join(ctx.out, "e3b_swap.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d["cell"] = d.N * 100 + d.H
    d["is12"] = (d.N >= 10) | (d.H >= 10)
    out = []
    for (kind, layer, group), g in d.groupby(["kind", "layer", "group"]):
        for tag, gg in (("all_pairs", g), ("pairs_with_10_11_12", g[g.is12]), ("single_digit_pairs", g[~g.is12])):
            if len(gg) < 3:
                continue
            m, lo, hi = ML.cluster_bootstrap_ci(gg["shift"].values, gg.cell.values, n_boot=2000 if ctx.smoke else 10000)
            mt, lot, hit = ML.cluster_bootstrap_ci(gg.dtgt.values, gg.cell.values, n_boot=2000 if ctx.smoke else 10000)
            out.append({"kind": kind, "layer": layer, "group": group, "subset": tag, "n": len(gg), "mean_shift": m,
                        "ci_lo": lo, "ci_hi": hi, "mean_dtgt": mt, "dtgt_lo": lot, "dtgt_hi": hit})
    s = pd.DataFrame(out)
    ctx.save(s, "e3b_summary.csv")
    sw = s[(s.kind == "swap") & (s.subset == "all_pairs")]
    ctrl = s[(s.kind.isin(["random", "same_answer"])) & (s.subset == "all_pairs")]
    cmax = ctrl.groupby("layer").mean_shift.apply(lambda x: x.abs().max()) if len(ctrl) else pd.Series(dtype=float)
    rule = []
    for (layer, group), g in sw.groupby(["layer", "group"]):
        r = g.iloc[0]
        c = float(cmax.get(layer, 0.0))
        rule.append({"layer": layer, "group": group, "mean_shift": r.mean_shift, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                     "best_control_abs_shift_same_layer": c,
                     "PREREG_supported": bool(r.ci_lo > 0 and r.mean_shift - c >= 1.0)})
    ctx.save(pd.DataFrame(rule), "e3b_prereg_rule.csv")


def part_c(ctx):
    ad = ctx.adapter
    if ctx.prior is None:
        raise FileNotFoundError("E3c needs prior text_only free-generation outputs (conversion_test.csv)")
    rng = np.random.default_rng(SEED)
    pr = ctx.prior
    wrong = pr[(~pr.minute_correct) & pr.parse_ok]
    right = pr[pr.minute_correct]
    cap = 3 if ctx.smoke else ctx.args.c_pairs

    def pick_source(tr, pool, need_same_n, rng):
        c = pool[(pool.lead_in_id == tr.lead_in_id)]
        if need_same_n:
            c = c[(c.position == tr.position) & (c.hour != tr.hour)]
        c = c[np.array([ctx.ntok(x) == ctx.ntok(tr.hour) for x in c.hour], dtype=bool)]
        c = c[np.array([ctx.ntok(x) == ctx.ntok(tr.position) for x in c.position], dtype=bool)]
        c = c[~((c.position == tr.position) & (c.hour == tr.hour)).values]
        return None if len(c) == 0 else c.iloc[int(rng.integers(len(c)))]

    jobs = []
    for _, tr in wrong.sample(frac=1.0, random_state=SEED).iterrows():
        s_r = pick_source(tr, right, True, rng)
        if s_r is not None:
            jobs.append(("right2wrong", tr, s_r))
        s_w = pick_source(tr, wrong, False, rng)
        if s_w is not None:
            jobs.append(("wrong2wrong", tr, s_w))
    jobs = [j for j in jobs if j[0] == "right2wrong"][:cap] + [j for j in jobs if j[0] == "wrong2wrong"][:cap]
    for _, tr in right.sample(frac=1.0, random_state=SEED + 1).iterrows():
        if sum(j[0] == "right2right" for j in jobs) >= (2 if ctx.smoke else cap // 2):
            break
        s = pick_source(tr, right, False, rng)
        if s is not None:
            jobs.append(("right2right", tr, s))
    rows, t0 = [], time.time()
    for k, (kind, tr, sr) in enumerate(jobs):
        n, h, lead = int(tr.position), int(tr.hour), int(tr.lead_in_id)
        tgt, src = ctx.prompt(lead, n, h), ctx.prompt(int(sr.lead_in_id), int(sr.position), int(sr.hour))
        if tgt.P != src.P:
            continue
        pre, cs = cands_for(ctx, h, ALL_MINUTES)
        gi = n % 12
        base_ok = int(ML.score_candidates(ad, tgt.ids, pre, cs).argmax()) == gi
        src_res, tgt_res = source_residuals(ctx, src), source_residuals(ctx, tgt)
        sp = tgt.spans
        groups = {"N": sp["N"], "H": sp["H"], "between": sp["between"], "last": sp["last"]}

        def rec(variant, layer, group, lp):
            rows.append({"kind": kind, "variant": variant, "lead": lead, "N": n, "H": h, "layer": layer, "group": group,
                         "baseline_ok": base_ok, "ok": int(lp.argmax()) == gi, "gold_lp": float(lp[gi])})

        for l in ctx.layers_sub:
            for g, pos in groups.items():
                with ML.patch_residuals(ctx.layers, [l], pos, src_res):
                    rec("source", l, g, ML.score_candidates(ad, tgt.ids, pre, cs))
            if kind == "right2wrong":
                rp = rand_positions(sp, len(sp["N"]), hash((lead, n, h, l)) % (2 ** 31))
                if rp is not None:
                    with ML.patch_residuals(ctx.layers, [l], rp, src_res):
                        rec("random_position", l, "random", ML.score_candidates(ad, tgt.ids, pre, cs))
        if kind == "right2wrong":
            for l in ctx.layers_sub[::4] or ctx.layers_sub[:1]:
                with ML.patch_residuals(ctx.layers, [l], list(range(tgt.P)), tgt_res):
                    rec("identity", l, "all", ML.score_candidates(ad, tgt.ids, pre, cs))
        if (k + 1) % 20 == 0:
            print(f"  E3c {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    ctx.save(pd.DataFrame(rows), "e3c_restoration.csv")


def summarize_c(ctx):
    p = os.path.join(ctx.out, "e3c_restoration.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    out = []
    for (kind, variant, layer, group), g in d.groupby(["kind", "variant", "layer", "group"]):
        out.append({"kind": kind, "variant": variant, "layer": layer, "group": group, "n": len(g),
                    "baseline_ok_rate": g.baseline_ok.mean(), "patched_ok_rate": g.ok.mean(),
                    "restored_rate_among_wrong": (g.ok & ~g.baseline_ok.astype(bool)).sum() / max(1, (~g.baseline_ok.astype(bool)).sum())
                    if kind != "right2right" else np.nan,
                    "damage_rate_among_right": ((~g.ok.astype(bool)) & g.baseline_ok.astype(bool)).sum() / max(1, g.baseline_ok.sum())
                    if kind == "right2right" else np.nan})
    s = pd.DataFrame(out)
    ctx.save(s, "e3c_summary.csv")
    a = s[(s.kind == "right2wrong") & (s.variant == "source")]
    ctrl = s[((s.kind == "wrong2wrong") & (s.variant == "source")) | (s.variant == "random_position")]
    best_ctrl = float(ctrl.restored_rate_among_wrong.max()) if len(ctrl) else 0.0
    rule = a.assign(best_control_restored=best_ctrl,
                    PREREG_supported=lambda x: (x.restored_rate_among_wrong >= 0.5) & (x.restored_rate_among_wrong - best_ctrl >= 0.25))
    ctx.save(rule, "e3c_prereg_rule.csv")


# ----------------------------------------------------------------------------------------------------------
# E3d edge knockout
# ----------------------------------------------------------------------------------------------------------

def d_cells(ctx):
    L = ctx.L
    if ctx.smoke:
        return [("intact", "none", None), ("H_all", "H", (0, L)), ("H_late", "H", (L // 2, L)),
                ("random_all", "random", (0, L)), ("N_all", "N", (0, L))]
    cells = [("intact", "none", None), ("H_all", "H", (0, L)), ("random_all", "random", (0, L))]
    if ctx.args.d_mode == "full":
        cells += [("N_all", "N", (0, L))]
        cells += [(f"H_win{s}-{min(s + 4, L)}", "H", (s, min(s + 4, L))) for s in range(0, L, 4)]
        cells += [(f"H_cum{s}-end", "H", (s, L)) for s in range(4, L, 4)]
    return cells


def part_d(ctx):
    ad = ctx.adapter
    st = ML.BlockState()
    rows, t0 = [], time.time()
    trials = ctx.trials()[:4] if ctx.smoke else ctx.trials()
    max_new = 6 if ctx.smoke else 24
    for ci, (cell, span_kind, rng_layers) in enumerate(d_cells(ctx)):
        manager = (ML.block_attention(ad, rng_layers, st) if rng_layers else ML.eager_attention(ad))
        with manager:
            for (lead, n, h) in trials:
                tp = ctx.prompt(lead, n, h)
                if span_kind == "none":
                    key = None
                elif span_kind == "random":
                    key = rand_positions(tp.spans, len(tp.spans["H"]), hash((lead, n, h)) % (2 ** 31))
                else:
                    key = tp.spans[span_kind]
                raw = ML.greedy_generate_blocked(ad, tp.ids, st, key_pos=key, max_new=max_new)
                lab, ph, pm = ML.label_answer(raw, h, (n % 12) * 5)
                rows.append({"cell": cell, "span": span_kind, "start": rng_layers[0] if rng_layers else -1,
                             "end": rng_layers[1] if rng_layers else -1, "lead": lead, "N": n, "H": h, "raw": raw,
                             "label": lab, "pred_hour": ph, "pred_minute": pm, "minute_correct": lab == "correct",
                             "hour_correct": (ph is not None and ML.hour12(ph) == ML.hour12(h))})
        print(f"  E3d cell {ci + 1}/{len(d_cells(ctx))} {cell}: acc {np.mean([r['minute_correct'] for r in rows if r['cell'] == cell]):.3f}"
              f"  elapsed {time.time() - t0:.0f}s", flush=True)
        ctx.save(pd.DataFrame(rows), "e3d_knockout.csv")


def summarize_d(ctx):
    p = os.path.join(ctx.out, "e3d_knockout.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    key = ["lead", "N", "H"]
    intact = d[d.cell == "intact"][key + ["minute_correct", "label"]].rename(
        columns={"minute_correct": "mc0", "label": "lab0"})
    nb = 2000 if ctx.smoke else 10000
    out = []
    for cell, g in d.groupby("cell", sort=False):
        m = g.merge(intact, on=key)
        cl = (m.N * 100 + m.H).values
        row = {"cell": cell, "n": len(m), "acc": m.minute_correct.mean(), "acc_intact": m.mc0.mean(),
               "leak_n": int((m.label == "leak").sum()), "leak_n_intact": int((m.lab0 == "leak").sum())}
        if cell != "intact" and len(m) >= 3:
            acc_d = m.minute_correct.astype(float).values - m.mc0.astype(float).values
            leak_d = (m.label == "leak").astype(float).values - (m.lab0 == "leak").astype(float).values
            mm, lo, hi = ML.cluster_bootstrap_ci(acc_d, cl, n_boot=nb)
            ml, lol, hil = ML.cluster_bootstrap_ci(leak_d, cl, n_boot=nb)
            row.update({"acc_diff": mm, "acc_lo": lo, "acc_hi": hi, "acc_p": ML.sign_flip_p(acc_d, cl, n_perm=nb),
                        "leak_diff_per_trial": ml, "leak_lo": lol, "leak_hi": hil,
                        "leak_p": ML.sign_flip_p(leak_d, cl, n_perm=nb)})
        out.append(row)
    s = pd.DataFrame(out)
    ctx.save(s, "e3d_summary.csv")
    # PREREG rule (all layers, H-span): leak count down >=30% relative AND accuracy up >=4 points, both CIs exclude 0,
    # random-span control changes accuracy by <2 points. 'weak conversion' = leaks fall but accuracy gain < half the leak drop.
    try:
        h, r = s[s.cell == "H_all"].iloc[0], s[s.cell == "random_all"].iloc[0]
        rel = (h.leak_n_intact - h.leak_n) / max(1, h.leak_n_intact)
        leak_drop_frac = (h.leak_n_intact - h.leak_n) / h.n
        v = {"model": ctx.short, "leak_intact": int(h.leak_n_intact), "leak_blocked": int(h.leak_n), "leak_rel_drop": rel,
             "acc_diff_points": 100 * h.acc_diff, "acc_ci_points": [100 * h.acc_lo, 100 * h.acc_hi],
             "leak_ci_excludes0": bool(h.leak_hi < 0), "acc_ci_excludes0": bool(h.acc_lo > 0),
             "random_span_acc_diff_points": 100 * r.acc_diff}
        v["M_bind_supported"] = bool(rel >= 0.30 and 100 * h.acc_diff >= 4 and h.acc_lo > 0 and h.leak_hi < 0
                                     and abs(100 * r.acc_diff) < 2)
        v["weak_conversion_account"] = bool((not v["M_bind_supported"]) and rel > 0 and h.acc_diff < 0.5 * leak_drop_frac)
        with open(os.path.join(ctx.out, "e3d_prereg_verdict.json"), "w") as f:
            json.dump(v, f, indent=1)
        print("  E3d verdict:", v)
    except Exception as e:  # noqa: BLE001
        print("  E3d verdict not computable:", e)


# ----------------------------------------------------------------------------------------------------------
# E3e description transplant
# ----------------------------------------------------------------------------------------------------------

DESC_SENT = "The long minute hand is pointing at the number {N}, and the short hour hand is pointing at the number {H}."
CONV_Q = "What time does the clock show? Answer only in HH:MM format."
DESC_Q = "Finally give the time as HH:MM."


def part_e(ctx):
    d = ctx.desc
    if d is None:
        raise FileNotFoundError("E3e needs describe.csv")
    d = d.copy()
    d["both_exact"] = (d.said_minute_number == d.true_minute_number) & d.hour_desc_ok.astype(bool)
    sel = d[d.both_exact].reset_index(drop=True)
    if ctx.smoke:
        sel = sel.head(4)
    rows = []
    for k, r in sel.iterrows():
        N, H = int(r.said_minute_number), int(r.said_hour_number)
        verb = str(r.raw_answer)
        cut = None
        import re as _re
        mm = _re.search(r"(?i)\b(therefore|finally|thus|so,? the time|the time (?:shown|is))", verb)
        desc_verbatim = verb[:mm.start()].strip() if mm else None
        # PRIMARY (deviations log 2026-11-05): the model's own verbatim description, no image.
        # Clocks whose reply has no extractable description-before-answer are absent from the primary variant.
        # SECONDARY: template sentence + conversion question; EXPLORATORY: template + describe-style question.
        variants = {}
        if desc_verbatim and len(desc_verbatim) > 20:
            variants["verbatim_description"] = f"{desc_verbatim} {CONV_Q}"
        variants["conv_question"] = f"{DESC_SENT.format(N=N, H=H)} {CONV_Q}"
        variants["describe_style"] = f"{DESC_SENT.format(N=N, H=H)} {DESC_Q}"
        for name, text in variants.items():
            raw = ctx.gen(text, ctx.args.max_new_tokens)
            ph, pm = ML.parse_hhmm(raw)
            ok = ph is not None and ML.hour12(ph) == ML.hour12(int(r.true_hour)) and pm == int(r.true_minute)
            rows.append({"filename": r.filename, "variant": name, "N": N, "H": H, "true_hour": int(r.true_hour),
                         "true_minute": int(r.true_minute), "raw": raw, "ok": bool(ok), "with_image_ok": bool(r.answer_ok)})
        if (k + 1) % 50 == 0:
            print(f"  E3e {k + 1}/{len(sel)}", flush=True)
    ctx.save(pd.DataFrame(rows), "e3e_transplant.csv")


def summarize_e(ctx):
    p = os.path.join(ctx.out, "e3e_transplant.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    out = []
    for v, g in d.groupby("variant"):
        w_img = (~g.with_image_ok.astype(bool)).astype(float).values
        w_txt = (~g.ok.astype(bool)).astype(float).values
        diff = w_txt - w_img
        cl = np.arange(len(g))
        m, lo, hi = ML.cluster_bootstrap_ci(diff, cl, n_boot=2000 if ctx.smoke else 10000)
        wr = float(w_txt.mean())
        verdict = ("language_side" if (wr >= 0.40 and lo >= -0.15 and hi <= 0.15) else
                   "image_interference" if (wr <= 0.25 and (hi < 0 or lo > 0)) else "mixed")
        out.append({"variant": v, "n_clocks": len(g), "wrong_with_image": w_img.mean(), "wrong_transplant": wr,
                    "paired_diff": m, "diff_lo": lo, "diff_hi": hi, "sign_flip_p": ML.sign_flip_p(diff, cl, n_perm=2000 if ctx.smoke else 10000),
                    "is_primary": v == "verbatim_description",
                    "PREREG_verdict_if_primary": verdict})
    ctx.save(pd.DataFrame(out), "e3e_summary.csv")


# ----------------------------------------------------------------------------------------------------------
# E3f H=12 forms
# ----------------------------------------------------------------------------------------------------------

def form(n, kind):
    return {"num": f"the number {n}", "word": f"the number {ML.NUM_WORDS[n]}", "oclock": f"the {n} o'clock position"}[kind]


FORM_CONDS = [("num", "num"), ("num", "word"), ("num", "oclock"), ("word", "num"), ("oclock", "num")]


def part_f(ctx):
    cells = sorted({(n, h) for n in range(1, 13) for h in range(1, 13) if h >= 10 or n >= 10})
    rows = []
    jobs = [(lead, n, h, fn, fh) for lead in range(ctx.CT.N_LEAD_INS) for (n, h) in cells for (fn, fh) in FORM_CONDS]
    if ctx.smoke:
        jobs = jobs[::max(1, len(jobs) // 20)][:20]
    t0 = time.time()
    for k, (lead, n, h, fn, fh) in enumerate(jobs):
        base = (f"The long minute hand is pointing at {form(n, fn)}, and the short hour hand is pointing at {form(h, fh)}. "
                f"{CONV_Q}")
        text = ctx.TO.with_lead_in(ctx.TO.LEAD_INS[lead], base)
        raw = ctx.gen(text, ctx.args.max_new_tokens)
        lab, ph, pm = ML.label_answer(raw, h, (n % 12) * 5)
        rows.append({"lead": lead, "N": n, "H": h, "formN": fn, "formH": fh, "raw": raw, "label": lab, "pred_hour": ph,
                     "pred_minute": pm, "minute_correct": lab == "correct",
                     "hour_correct": ph is not None and ML.hour12(ph) == ML.hour12(h),
                     "hour_copied_from_N": ph is not None and ML.hour12(ph) == ML.hour12(n) and ML.hour12(n) != ML.hour12(h),
                     "hour_is_01_for_H12": ph is not None and h == 12 and ph == 1, "minute_is_00": pm == 0})
        if (k + 1) % 100 == 0:
            print(f"  E3f {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    ctx.save(pd.DataFrame(rows), "e3f_forms.csv")


def summarize_f(ctx):
    p = os.path.join(ctx.out, "e3f_forms.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d["cond"] = d.formN + "/" + d.formH
    d["Hgrp"] = np.where(d.H == 12, "H=12", np.where(d.H >= 10, "H=10,11", "H<10"))
    out = []
    for (c, g), x in d.groupby(["cond", "Hgrp"]):
        out.append({"cond(formN/formH)": c, "H_group": g, "n": len(x), "minute_acc": x.minute_correct.mean(),
                    "hour_acc": x.hour_correct.mean(), "hour_copied_from_N": x.hour_copied_from_N.mean(),
                    "hour_01_at_H12": x.hour_is_01_for_H12.mean(), "minute_00": x.minute_is_00.mean(),
                    "leak": (x.label == "leak").mean(), "unparsed": (x.label == "unparsed").mean()})
    ctx.save(pd.DataFrame(out), "e3f_summary.csv")


# ----------------------------------------------------------------------------------------------------------
# text-light pilot
# ----------------------------------------------------------------------------------------------------------

TEXT_LIGHT = "The short hour hand is pointing at the number {H}. What time does the clock show? Answer only in HH:MM format."


def part_p(ctx):
    CT, TO = ctx.CT, ctx.TO
    groups = CT.build_groups()
    if ctx.smoke:
        groups = groups[::max(1, len(groups) // 6)][:6]
    images_dir = "conversion_images_smoke" if ctx.smoke else "conversion_images"
    os.makedirs(images_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for k, (lead, n, h) in enumerate(groups):
        prompt = TO.with_lead_in(TO.LEAD_INS[lead], TEXT_LIGHT.format(H=h))
        for cond in ("matching_clock", "blank"):
            path, ih, im = CT.ensure_image(cond, n, h, images_dir, CT.IMAGE_SEED)
            inputs = ctx.adapter.build_inputs(path, prompt)
            raw = ctx.adapter.generate_answer(inputs, max_new_tokens=ctx.args.max_new_tokens)
            truth = (n % 12) * 5
            lab, ph, pm = ML.label_answer(raw, h, truth)
            rows.append({"cond": cond, "N": n, "H": h, "lead": lead, "raw": raw, "label": lab, "pred_hour": ph,
                         "pred_minute": pm, "minute_correct": lab == "correct",
                         "hour_correct": ph is not None and ML.hour12(ph) == ML.hour12(h),
                         "unparsed": lab == "unparsed"})
        if (k + 1) % 100 == 0:
            print(f"  pilot {k + 1}/{len(groups)}  {time.time() - t0:.0f}s", flush=True)
    ctx.save(pd.DataFrame(rows), "text_light_pilot.csv")


def summarize_p(ctx):
    p = os.path.join(ctx.out, "text_light_pilot.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    out = []
    for cond, g in d.groupby("cond"):
        out.append({"model": ctx.short, "cond": cond, "n": len(g), "minute_acc": g.minute_correct.mean(),
                    "hour_acc": g.hour_correct.mean(), "unparsed": g.unparsed.mean(),
                    "chance_12_positions": 1 / 12})
    s = pd.DataFrame(out)
    m = s[s.cond == "matching_clock"].minute_acc.iloc[0]
    s["E2_floor_rule_pass_(>=0.15)"] = bool(m >= 0.15)
    ctx.save(s, "text_light_pilot_summary.csv")
    print(f"  TEXT-LIGHT PILOT {ctx.short}: matching-clock minute accuracy {m:.3f} -> floor rule "
          f"{'PASSED' if m >= 0.15 else 'NOT passed (E2 text-light test uninformative for this model)'}", flush=True)


# ----------------------------------------------------------------------------------------------------------

PARTS = {"a": part_a, "b": part_b, "c": part_c, "d": part_d, "e": part_e, "f": part_f, "p": part_p}
SUMMARIES = {"a": summarize_a, "b": summarize_b, "c": summarize_c, "d": summarize_d, "e": summarize_e, "f": summarize_f,
             "p": summarize_p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--parts", default="verify,a,b,c,d,e,f,p")
    ap.add_argument("--out_dir", default=os.path.join("results", "mech_e3"))
    ap.add_argument("--prior_dir", default="prior")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--d_mode", default="full", choices=["full", "min"])
    ap.add_argument("--c_pairs", type=int, default=100)
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    if parts == ["analyze"]:
        from adapters import get_adapter
        short = get_adapter(model_id=args.model_id, adapter_name=args.adapter).short_name
        class _A:  # minimal stand-in: summaries only need CSVs
            short_name = short
        ctx = Ctx.__new__(Ctx)
        ctx.args, ctx.smoke, ctx.short = args, args.smoke, short
        ctx.out = os.path.join(args.out_dir, short)
        ctx.L = int(max((int(c[6:]) for c in pd.read_csv(os.path.join(ctx.out, "e3a_lens.csv"), nrows=1).columns
                         if c.startswith("gold_L")), default=0) + 1) if os.path.exists(os.path.join(ctx.out, "e3a_lens.csv")) else 0
        for f in SUMMARIES.values():
            f(ctx)
        return 0

    from adapters import get_adapter
    adapter = get_adapter(model_id=args.model_id, adapter_name=args.adapter)
    adapter.load(args.model_id)
    ctx = Ctx(adapter, args)
    print(f"mech_e3: model={ctx.short} layers={ctx.L} parts={parts} smoke={args.smoke}  [{TIMEOUT_NOTE}]", flush=True)

    if "verify" in parts:
        t0 = time.time()
        ok = part_verify(ctx)
        print(f"verify took {time.time() - t0:.0f}s", flush=True)
        if not ok:
            print("GATE FAILED -> not running any experiment part for this model (see verify.json).", flush=True)
            return 3
    for p in [x for x in parts if x != "verify"]:
        if p not in ("p",) and not args.no_gate and not gate_ok(ctx):
            print(f"part {p}: skipped, no passing verify.json for {ctx.short}", flush=True)
            return 3
        t0 = time.time()
        print(f"\n===== part {p} =====", flush=True)
        PARTS[p](ctx)
        SUMMARIES[p](ctx)
        print(f"part {p} took {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
