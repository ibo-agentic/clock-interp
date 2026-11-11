"""R1 (styled clocks S1 thin hands, S2 ticks only; Qwen-3B, InternVL3-2B, Qwen-7B S2) vs the standard style, and the pre-registered predictions that R1 covers
(PREREG_MECHANISM.md ninth entry: P1, P2, P4, P5; P3 waits for R2, P6 for R3). CPU only.
usage: python analysis/analyze_R1.py -> results/styles/R1_report.txt"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mech_lib as ML  # noqa: E402
from describe_check import add_tolerant_columns  # noqa: E402

out = []
P = lambda s="": (print(s), out.append(s))
STYLES = ["standard", "s1_thin", "s2_ticks"]
MODELS = ["qwen2.5-vl-3b", "internvl3-2b", "qwen2.5-vl-7b"]
KEY = ["position", "hour", "lead_in_id"]


def wilson(k, n):
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt(k, n):
    lo, hi = wilson(k, n)
    lo, hi = max(0.0, lo), min(1.0, hi)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f},{100 * hi:.1f}] ({k}/{n})"


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def folder(m, st):
    return m if st == "standard" else f"{m}_{st}"


def beh(m, st):
    p = os.path.join(ROOT, "results", folder(m, st), "results.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    hr = (d.pred_hour == d.true_hour).fillna(False).astype(bool)      # parse failure (NaN) never equals the truth -> wrong, over ALL 500 (P1 definition)
    mn = (d.pred_minute == d.true_minute).fillna(False).astype(bool)
    return {"n": len(d), "exact": int((hr & mn).sum()), "hour": int(hr.sum()), "minute": int(mn.sum()), "parsed": int(d.parse_success.astype(bool).sum())}


def desc(m, st):
    p = os.path.join(ROOT, "results", folder(m, st), "describe.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if not {"said_minute_number_tolerant", "hour_desc_ok_tolerant"} <= set(d.columns):
        d = add_tolerant_columns(d[[c for c in d.columns if not c.endswith("_tolerant")]])
    mx = (d.said_minute_number_tolerant == d.true_minute_number).fillna(False).astype(bool)
    ho = d.hour_desc_ok_tolerant.astype(bool)
    both = mx & ho
    ao = d.answer_ok.astype(bool)
    return {"n": len(d), "minute_exact": int(mx.sum()), "hour": int(ho.sum()), "both": int(both.sum()), "wrong_given_both": int((both & ~ao).sum()), "answer_ok": int(ao.sum())}


def conv(m, st):
    f = "conversion_test.csv" if st == "standard" else f"conversion_test_{st}.csv"
    p = os.path.join(ROOT, "results", m, f)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.model == m].copy()
    d["mc"] = d.minute_correct.astype(bool)
    return d


def labels(d, with_image=False):
    return pd.Series([ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute), (None if pd.isna(r.image_minute) else int(r.image_minute)) if with_image else None)[0]
                      for r in d.itertuples()])


def leak_count(d):
    return int((labels(d) == "leak").sum())


P("=== 1. STATUS and generator gate ===")
sj = json.load(open(os.path.join(ROOT, "results/styles/R1_status.json")))
for s in sj:
    P(f"  {s['step']:<40} {s['status']:<4} {s['minutes']:>6.1f} min")
P(f"  total {sum(s['minutes'] for s in sj) / 60:.2f} h (guess ~3.5 h)")
P("  generator gate (log 00_style_check): default-style re-render identical to data/ (500/500) and data_positions/ (480/480), 0 images differing outside the numeral ring, ring difference 0 px -> PASS")

P("\n=== 4a. BEHAVIOUR on the 500 clocks (exact HH:MM / hour / minute; over ALL 500, parse failure = wrong) ===")
rows, b_all = [], {}
for m in MODELS:
    for st in STYLES:
        r = beh(m, st)
        if r is None:
            continue
        b_all[(m, st)] = r
        rows.append({"model": m, "style": st, "exact": fmt(r["exact"], r["n"]), "hour": fmt(r["hour"], r["n"]), "minute": fmt(r["minute"], r["n"]), "parsed": f"{r['parsed']}/{r['n']}"})
P(pd.DataFrame(rows).to_string(index=False))

P("\n=== 4b. DESCRIBE (tolerant parse = primary; 480 clocks) ===")
rows, d_all = [], {}
for m in MODELS:
    for st in STYLES:
        r = desc(m, st)
        if r is None:
            continue
        d_all[(m, st)] = r
        rows.append({"model": m, "style": st, "minute exact": fmt(r["minute_exact"], r["n"]), "hour": fmt(r["hour"], r["n"]), "both exact": fmt(r["both"], r["n"]),
                     "wrong | both exact": fmt(r["wrong_given_both"], r["both"]) if r["both"] else "n/a (0 clocks)", "final answer right": fmt(r["answer_ok"], r["n"])})
P(pd.DataFrame(rows).to_string(index=False))

P("\n=== 4c. CONVERSION (432 keys): minute accuracy matching / mismatched; leaks in the matching condition; D vs the standard-style text-only ===")
rows, Dres = [], {}
for m in ("qwen2.5-vl-3b", "internvl3-2b"):
    std = conv(m, "standard")
    base = std[std.condition == "text_only"].set_index(KEY).mc
    for st in STYLES:
        d = conv(m, st)
        mt = d[d.condition == "matching_clock"].set_index(KEY)
        mm = d[d.condition == "mismatched_clock"].set_index(KEY)
        assert len(mt) == 432 and len(mm) == 432 and mt.index.isin(base.index).all()
        o = mt.mc.reindex(base.index)
        ot, oi = int((base & ~o).sum()), int((~base & o).sum())
        Dres[(m, st)] = (ot, oi, oi - ot, mcnemar(ot, oi))
        fol = int((labels(mm.reset_index(), with_image=True) == "follows_image").sum())
        rows.append({"model": m, "style": st, "text-only (standard, ref)": fmt(int(base.sum()), 432), "matching": fmt(int(mt.mc.sum()), 432), "mismatched": fmt(int(mm.mc.sum()), 432),
                     "mismatched follows image": f"{fol}/432", "leaks (matching)": leak_count(mt.reset_index()), "only-text / only-image": f"{ot} / {oi}", "D": oi - ot, "exact McNemar p": f"{mcnemar(ot, oi):.2g}"})
P(pd.DataFrame(rows).to_string(index=False))

P("\n=== 3. PRE-REGISTERED PREDICTIONS covered by R1 (ninth entry) ===")
P("P1 (confirmatory): behaviour exact HH:MM <= 20% in every cell. R1 cells: Qwen-3B S1, S2; InternVL3-2B S1, S2; Qwen-7B S2 (5 of the 7; Gemma-4B S1/S2 wait for R2).")
ok1 = True
for m, st in [("qwen2.5-vl-3b", "s1_thin"), ("qwen2.5-vl-3b", "s2_ticks"), ("internvl3-2b", "s1_thin"), ("internvl3-2b", "s2_ticks"), ("qwen2.5-vl-7b", "s2_ticks")]:
    r = b_all[(m, st)]
    lo, hi = wilson(r["exact"], r["n"])
    ok = r["exact"] / r["n"] <= 0.20
    ok1 &= ok
    P(f"   {m} {st}: exact {fmt(r['exact'], r['n'])}; threshold <= 20% (upper CI {100 * hi:.1f}%) -> {'holds' if ok else 'FAILS'}")
P(f"   P1 on the 5 R1 cells: {'all hold' if ok1 else 'NOT all hold'}")

P("\nP2 (confirmatory): D = (only styled matching clock right) - (only text-only right) has the standard sign (Qwen-3B D<0, InternVL3-2B D>0). R1 cells: 4 of 6; exact McNemar, Holm over these 4 (the pre-registered Holm over 6 needs the two Gemma cells from R2).")
cells = [("qwen2.5-vl-3b", "s1_thin", -1), ("qwen2.5-vl-3b", "s2_ticks", -1), ("internvl3-2b", "s1_thin", 1), ("internvl3-2b", "s2_ticks", 1)]
adj = holm(np.array([Dres[(m, s)][3] for m, s, _ in cells]))
ok2 = True
for (m, s, sign), a in zip(cells, adj):
    ot, oi, D, p = Dres[(m, s)]
    ok = (D * sign) > 0
    ok2 &= ok
    P(f"   {m} {s}: only-text {ot}, only-image {oi}, D = {D:+d} (standard D = {Dres[(m, 'standard')][2]:+d}); p = {p:.2g}, Holm(4) = {a:.2g} -> sign {'matches' if ok else 'DOES NOT match'}")
P(f"   P2 on the 4 R1 cells: {'all signs match' if ok2 else 'NOT all signs match'}")

P("\nP5 (confirmatory): styled-matching leak count >= 70% of the standard-style matching count (Qwen-3B >= 11 of 15; InternVL3-2B >= 24 of 34; Gemma-4B waits for R2).")
ok5 = True
for m in ("qwen2.5-vl-3b", "internvl3-2b"):
    ref = leak_count(conv(m, "standard").query("condition == 'matching_clock'"))
    for st in ("s1_thin", "s2_ticks"):
        n = leak_count(conv(m, st).query("condition == 'matching_clock'"))
        thr = math.ceil(0.7 * ref)
        ok = n >= thr
        ok5 &= ok
        P(f"   {m} {st}: leaks {n} (standard {ref}; threshold >= {thr}; ratio {n / ref:.0%}) -> {'holds' if ok else 'FAILS'}")
P(f"   P5 on the 4 R1 cells: {'all hold' if ok5 else 'NOT all hold'}")

P("\nP4 (exploratory): ticks-only describe minute-exact for Qwen-7B <= standard minus 10 points (standard 84.4% -> threshold <= 74.4%).")
s7, t7 = desc("qwen2.5-vl-7b", "standard"), d_all[("qwen2.5-vl-7b", "s2_ticks")]
P(f"   Qwen-7B standard {fmt(s7['minute_exact'], s7['n'])} -> S2 {fmt(t7['minute_exact'], t7['n'])}: change {100 * (t7['minute_exact'] / t7['n'] - s7['minute_exact'] / s7['n']):+.1f} points -> "
  f"{'supported' if t7['minute_exact'] / t7['n'] <= 0.744 else 'NOT supported'} (threshold 74.4%)")
P("   Descriptive, minute-exact standard -> S1 -> S2: " + "; ".join(f"{m}: " + " -> ".join(f"{100 * d_all[(m, st)]['minute_exact'] / d_all[(m, st)]['n']:.1f}%" for st in STYLES if (m, st) in d_all) for m in MODELS))
open(os.path.join(ROOT, "results/styles/R1_report.txt"), "w", encoding="utf8").write("\n".join(out) + "\n")
