"""R2 (styled clocks, Gemma-4B) and R3 (real photos, five models): completes the pre-registered predictions P1-P6 (PREREG_MECHANISM.md ninth entry), now that every cell exists.
P1 over all 7 cells, P2 with the pre-registered Holm over all 6 cells, P3, P5 over all 6 cells, P6 on the photos. CPU only.
usage: python analysis/analyze_R2_R3.py -> results/styles/R2_R3_report.txt"""
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
KEY = ["position", "hour", "lead_in_id"]
ALL = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl3-2b", "internvl3-8b", "gemma-3-4b-it"]


def wilson(k, n):
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    lo, hi = wilson(k, n)
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


def beh(path):
    p = os.path.join(ROOT, "results", path, "results.csv")
    d = pd.read_csv(p)
    hr = (d.pred_hour == d.true_hour).fillna(False).astype(bool)      # parse failure = wrong, over ALL rows (P1/P6 definition)
    mn = (d.pred_minute == d.true_minute).fillna(False).astype(bool)
    return {"n": len(d), "exact": int((hr & mn).sum()), "hour": int(hr.sum()), "minute": int(mn.sum()), "parsed": int(d.parse_success.astype(bool).sum())}


def desc(path):
    d = pd.read_csv(os.path.join(ROOT, "results", path, "describe.csv"))
    if not {"said_minute_number_tolerant", "hour_desc_ok_tolerant"} <= set(d.columns):
        d = add_tolerant_columns(d[[c for c in d.columns if not c.endswith("_tolerant")]])
    mx = (d.said_minute_number_tolerant == d.true_minute_number).fillna(False).astype(bool)
    ho = d.hour_desc_ok_tolerant.astype(bool)
    both = mx & ho
    ao = d.answer_ok.astype(bool)
    return {"n": len(d), "minute_exact": int(mx.sum()), "hour": int(ho.sum()), "both": int(both.sum()), "wrong_given_both": int((both & ~ao).sum()), "answer_ok": int(ao.sum())}


def conv(m, st):
    f = "conversion_test.csv" if st == "standard" else f"conversion_test_{st}.csv"
    d = pd.read_csv(os.path.join(ROOT, "results", m, f))
    d = d[d.model == m].copy()
    d["mc"] = d.minute_correct.astype(bool)
    return d


def labels(d, with_image=False):
    return pd.Series([ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute), (None if pd.isna(r.image_minute) else int(r.image_minute)) if with_image else None)[0]
                      for r in d.itertuples()])


leak_count = lambda d: int((labels(d) == "leak").sum())

P("=== 1. STATUS and gates ===")
for tag, f in (("R2", "results/styles/R2_status.json"), ("R3", "results/photos/R3_status.json")):
    sj = json.load(open(os.path.join(ROOT, f)))
    P(f"{tag}: " + "; ".join(f"{s['step']} {s['status']} {s['minutes']:.1f}" for s in sj) + f"  => total {sum(s['minutes'] for s in sj) / 60:.2f} h")
P("R2 generator gate: default-style re-render identical to data/ (500/500) and data_positions/ (480/480), 0 differing pixels -> PASS")
man = pd.read_csv(os.path.join(ROOT, "results/photos/data_photos_tt500/manifest_provenance.csv"))
P(f"R3 photo selection: {len(man)} rows, licence values {man.license.value_counts().to_dict()}, sources {man.source.value_counts().to_dict()}, unique sample_id {man.sample_id.nunique()}; "
  f"width {man.width.min()}-{man.width.max()} px, height {man.height.min()}-{man.height.max()} px (median {int(man.width.median())} x {int(man.height.median())}); no image files in the zip or in results/")

P("\n=== 4a. GEMMA-4B: standard vs S1 vs S2 ===")
g = "gemma-3-4b-it"
rows = []
gb, gd = {}, {}
for st in STYLES:
    b, dd = beh(folder(g, st)), desc(folder(g, st))
    gb[st], gd[st] = b, dd
    rows.append({"style": st, "exact": fmt(b["exact"], b["n"]), "hour": fmt(b["hour"], b["n"]), "minute": fmt(b["minute"], b["n"]), "parsed": f"{b['parsed']}/{b['n']}",
                 "describe minute exact": fmt(dd["minute_exact"], dd["n"]), "describe hour": fmt(dd["hour"], dd["n"]), "both exact": fmt(dd["both"], dd["n"]),
                 "wrong|both": fmt(dd["wrong_given_both"], dd["both"]) if dd["both"] else "n/a (0 clocks)"})
P(pd.DataFrame(rows).T.to_string(header=False))
std = conv(g, "standard")
base = std[std.condition == "text_only"].set_index(KEY).mc
blank = std[std.condition == "blank"].set_index(KEY).mc
Dres, P3res, rows = {}, {}, []
for st in STYLES:
    d = conv(g, st)
    mt = d[d.condition == "matching_clock"].set_index(KEY)
    mm = d[d.condition == "mismatched_clock"].set_index(KEY)
    assert len(mt) == 432 and len(mm) == 432
    o = mt.mc.reindex(base.index)
    ot, oi = int((base & ~o).sum()), int((~base & o).sum())
    Dres[st] = (ot, oi, oi - ot, mcnemar(ot, oi))
    bl = blank.reindex(base.index)
    ob, om = int((bl & ~o).sum()), int((~bl & o).sum())             # only-blank-right, only-matching-right
    diff = (om - ob) / 432
    se = math.sqrt(max((om + ob) / 432 ** 2 - (om - ob) ** 2 / 432 ** 3, 0))
    P3res[st] = (100 * (o.mean() - bl.mean()), 100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se), ob, om, mcnemar(ob, om))
    fol = int((labels(mm.reset_index(), with_image=True) == "follows_image").sum())
    rows.append({"style": st, "text-only (std, ref)": fmt(int(base.sum()), 432), "blank (std, ref)": fmt(int(bl.sum()), 432), "matching": fmt(int(mt.mc.sum()), 432), "mismatched": fmt(int(mm.mc.sum()), 432),
                 "mismatched follows image": f"{fol}/432", "leaks (matching)": leak_count(mt.reset_index()), "only-text/only-image": f"{ot}/{oi}", "D": oi - ot, "McNemar p": f"{mcnemar(ot, oi):.2g}"})
P("\nConversion (32-token headline setting; 432 keys):")
P(pd.DataFrame(rows).T.to_string(header=False))

P("\n=== 3. PRE-REGISTERED PREDICTIONS, complete (ninth entry) ===")
# P1 ------------------------------------------------------------------
P("P1 (confirmatory): behaviour exact HH:MM <= 20% in all 7 cells (parse failures counted wrong).")
cells1 = [("qwen2.5-vl-3b", "s1_thin"), ("qwen2.5-vl-3b", "s2_ticks"), ("internvl3-2b", "s1_thin"), ("internvl3-2b", "s2_ticks"), ("qwen2.5-vl-7b", "s2_ticks"), (g, "s1_thin"), (g, "s2_ticks")]
ok1 = True
for m, st in cells1:
    r = beh(folder(m, st))
    ok = r["exact"] / r["n"] <= 0.20
    ok1 &= ok
    P(f"   {m} {st}: exact {fmt(r['exact'], r['n'])} (upper CI {100 * wilson(r['exact'], r['n'])[1]:.1f}%) -> {'holds' if ok else 'FAILS'}")
P(f"   P1: {'SUPPORTED (all 7 cells <= 20%)' if ok1 else 'NOT supported'}")
# P2 ------------------------------------------------------------------
P("\nP2 (confirmatory): D = only-styled-matching-right minus only-text-right has the standard sign in all 6 cells (Qwen-3B D<0; InternVL3-2B D>0; Gemma-4B D>0); exact McNemar, Holm over the 6.")
cells2 = [("qwen2.5-vl-3b", "s1_thin", -1), ("qwen2.5-vl-3b", "s2_ticks", -1), ("internvl3-2b", "s1_thin", 1), ("internvl3-2b", "s2_ticks", 1), (g, "s1_thin", 1), (g, "s2_ticks", 1)]
DD = {}
for m in ("qwen2.5-vl-3b", "internvl3-2b", g):
    s0 = conv(m, "standard")
    b0 = s0[s0.condition == "text_only"].set_index(KEY).mc
    for st in ("standard", "s1_thin", "s2_ticks"):
        mt = conv(m, st).query("condition == 'matching_clock'").set_index(KEY).mc.reindex(b0.index)
        ot, oi = int((b0 & ~mt).sum()), int((~b0 & mt).sum())
        DD[(m, st)] = (ot, oi, oi - ot, mcnemar(ot, oi))
adj = holm(np.array([DD[(m, s)][3] for m, s, _ in cells2]))
ok2 = True
for (m, s, sign), a in zip(cells2, adj):
    ot, oi, D, p = DD[(m, s)]
    ok = D * sign > 0
    ok2 &= ok
    P(f"   {m} {s}: only-text {ot}, only-image {oi}, D = {D:+d} (standard {DD[(m, 'standard')][2]:+d}); p = {p:.2g}, Holm(6) = {a:.2g} -> sign {'matches' if ok else 'DOES NOT match'}")
P(f"   P2: {'SUPPORTED (all 6 signs match)' if ok2 else 'NOT supported'}; all Holm(6) p < 0.05: {bool((adj < 0.05).all())}")
# P3 ------------------------------------------------------------------
P("\nP3 (confirmatory): Gemma-4B |minute accuracy(styled matching clock) - minute accuracy(blank)| < 5 points in BOTH styles (paired on the same 432 keys; blank = standard-style blank condition, style independent).")
ok3 = True
for st in ("s1_thin", "s2_ticks"):
    dpt, lo, hi, ob, om, p = P3res[st]
    ok = abs(dpt) < 5
    ok3 &= ok
    P(f"   {st}: matching - blank = {dpt:+.1f} points (paired 95% CI [{lo:+.1f},{hi:+.1f}]; only-blank-right {ob}, only-matching-right {om}, exact McNemar p = {p:.2g}) -> {'within 5' if ok else 'NOT within 5'}")
P(f"   (standard style for reference: matching - blank = {100 * (std[std.condition == 'matching_clock'].mc.mean() - std[std.condition == 'blank'].mc.mean()):+.1f} points)")
P(f"   P3: {'SUPPORTED' if ok3 else 'NOT supported'}")
# P5 ------------------------------------------------------------------
P("\nP5 (confirmatory): styled-matching leak count >= 70% of the standard matching count (Qwen-3B >= 11/15; InternVL3-2B >= 24/34; Gemma-4B >= 54/77), 6 cells.")
ok5 = True
for m in ("qwen2.5-vl-3b", "internvl3-2b", g):
    ref = leak_count(conv(m, "standard").query("condition == 'matching_clock'"))
    for st in ("s1_thin", "s2_ticks"):
        n = leak_count(conv(m, st).query("condition == 'matching_clock'"))
        thr = math.ceil(0.7 * ref)
        ok = n >= thr
        ok5 &= ok
        P(f"   {m} {st}: leaks {n} (standard {ref}; bar >= {thr}; ratio {n / ref:.0%}) -> {'holds' if ok else 'FAILS'}")
P(f"   P5: {'SUPPORTED (all 6 cells)' if ok5 else 'NOT supported'}")
# P4 (from R1) --------------------------------------------------------
s7, t7 = desc("qwen2.5-vl-7b"), desc("qwen2.5-vl-7b_s2_ticks")
P(f"\nP4 (exploratory, reported in R1): Qwen-7B describe minute-exact {fmt(s7['minute_exact'], s7['n'])} -> S2 {fmt(t7['minute_exact'], t7['n'])}; threshold <= 74.4% -> {'supported' if t7['minute_exact'] / t7['n'] <= 0.744 else 'NOT supported'}")

# photos ---------------------------------------------------------------
P("\n=== 4b. REAL PHOTOS (500 TickTockVQA CC BY 4.0 crops) vs the synthetic standard clocks ===")
rows = []
b_ph, d_ph = {}, {}
for m in ALL:
    bp, dp = beh(m + "_photos"), desc(m + "_photos")
    bs, ds = beh(m), desc(m)
    b_ph[m], d_ph[m] = bp, dp
    rows.append({"model": m, "photo exact": fmt(bp["exact"], bp["n"]), "photo hour": fmt(bp["hour"], bp["n"]), "photo minute": fmt(bp["minute"], bp["n"]), "photo parsed": f"{bp['parsed']}/{bp['n']}",
                 "synthetic exact": fmt(bs["exact"], bs["n"]), "synthetic hour": fmt(bs["hour"], bs["n"]), "synthetic minute": fmt(bs["minute"], bs["n"])})
P(pd.DataFrame(rows).T.to_string(header=False))
rows = []
for m in ALL:
    dp, ds = d_ph[m], desc(m)
    rows.append({"model": m, "photo minute exact": fmt(dp["minute_exact"], dp["n"]), "photo hour": fmt(dp["hour"], dp["n"]), "photo both exact": fmt(dp["both"], dp["n"]),
                 "photo wrong|both": fmt(dp["wrong_given_both"], dp["both"]) if dp["both"] else "n/a (0 photos)", "photo final answer right": fmt(dp["answer_ok"], dp["n"]),
                 "synthetic minute exact": fmt(ds["minute_exact"], ds["n"]), "synthetic both exact": fmt(ds["both"], ds["n"])})
P("\nDescribe, tolerant parse (primary; 500 photos vs 480 synthetic positions clocks):")
P(pd.DataFrame(rows).T.to_string(header=False))
P("   Chance levels: exact HH:MM 1/720 = 0.14%; hour 1/12 = 8.3%; minute 1/60 = 1.7%; described minute number 1/12 = 8.3%.")

P("\nP6 (confirmatory): photo exact HH:MM <= 10% for all five models; photo hour accuracy >= 25% for Qwen-7B and InternVL3-8B.")
ok6a = ok6b = True
for m in ALL:
    r = b_ph[m]
    ok = r["exact"] / r["n"] <= 0.10
    ok6a &= ok
    P(f"   {m}: exact {fmt(r['exact'], r['n'])} (bar <= 10%; upper CI {100 * wilson(r['exact'], r['n'])[1]:.1f}%) -> {'holds' if ok else 'FAILS'}")
for m in ("qwen2.5-vl-7b", "internvl3-8b"):
    r = b_ph[m]
    ok = r["hour"] / r["n"] >= 0.25
    ok6b &= ok
    P(f"   {m}: hour {fmt(r['hour'], r['n'])} (bar >= 25%; lower CI {100 * wilson(r['hour'], r['n'])[0]:.1f}%) -> {'holds' if ok else 'FAILS'}")
P(f"   P6: {'SUPPORTED' if ok6a and ok6b else 'NOT supported'} (exact part {ok6a}, hour part {ok6b})")

P("\n=== FINAL SUMMARY P1-P6 ===")
P(f"P1 {'SUPPORTED' if ok1 else 'NOT supported'} | P2 {'SUPPORTED' if ok2 else 'NOT supported'} | P3 {'SUPPORTED' if ok3 else 'NOT supported'} | P4 (exploratory) "
  f"{'supported' if t7['minute_exact'] / t7['n'] <= 0.744 else 'not supported'} | P5 {'SUPPORTED' if ok5 else 'NOT supported'} | P6 {'SUPPORTED' if ok6a and ok6b else 'NOT supported'}")
open(os.path.join(ROOT, "results/styles/R2_R3_report.txt"), "w", encoding="utf8").write("\n".join(out) + "\n")
