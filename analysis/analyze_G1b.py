"""G1b (Gemma-3-4B, bf16, max_new_tokens 128) vs G1 (32): cap rule check, rule validation (replies that finished early unchanged), text-only accuracy and hour->minute leaks,
text-only vs matching vs blank vs mismatched (exact McNemar, Holm over 3) on all trials and on sensitivity subsets.  Reads match_G1b_results/ (new) and results/gemma-3-4b-it/ (32-token G1).
usage: python analysis/analyze_G1b.py [--new match_G1b_results/results/gemma-3-4b-it] [--old results/gemma-3-4b-it_mnt32]  -> results/gemma/G1b_report.txt"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd  # noqa: E402

import mech_lib as ML  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--new", default="match_G1b_results/results/gemma-3-4b-it")
ap.add_argument("--old", default="results/gemma-3-4b-it")
ap.add_argument("--out", default="results/gemma/G1b_report.txt")
a = ap.parse_args()
out = []
P = lambda s="": (print(s), out.append(s))


def binom_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k, n):
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


rd = lambda base, f: pd.read_csv(os.path.join(ROOT, base, f))
tn, cn_ = rd(a.new, "text_only_v2.csv"), rd(a.new, "conversion_test.csv")
to_, co_ = rd(a.old, "text_only_v2.csv"), rd(a.old, "conversion_test.csv")
cap = rd(a.new, "cap_check_128.csv")

P("=== 1. SUCCESS RULE (logged before the run): cap_check at 128 -> 0 hits in conversion_test AND <= 5 in text_only_v2 ===")
P(cap.to_string(index=False))
h_t, h_c = int(cap.hit_cap.iloc[0]), int(cap.hit_cap.iloc[1])
P(f"text_only_v2 hits {h_t} (rule <= 5): {'MET' if h_t <= 5 else 'NOT MET'};  conversion_test hits {h_c} (rule 0): {'MET' if h_c == 0 else 'NOT MET'}  ->  RULE {'MET' if (h_t <= 5 and h_c == 0) else 'NOT MET'}")
Ln = cn_.raw_output.astype(str).str.len()
P(f"conversion_test: replies >= 200 characters (long chain-of-thought): {int((Ln >= 200).sum())} (condition: {cn_[Ln >= 200].condition.value_counts().to_dict()}; lead_in_id {cn_[Ln >= 200].lead_in_id.value_counts().to_dict()}); unparsed {int((~cn_.parse_ok.astype(bool)).sum())}")
P(f"the 6 cap hits (>= 430 characters, proxy; tokenizer not cached locally): all parsed = {bool(cn_[Ln >= 430].parse_ok.astype(bool).all())}; minute_correct among them {int(cn_[Ln >= 430].minute_correct.astype(bool).sum())}/{int((Ln >= 430).sum())}")

P("\n=== 2. RULE CHECK: trials whose 32-token reply had already finished (< 100 chars) -- identical at 128? ===")
for nm, o, n in (("text_only_v2", to_, tn), ("conversion_test", co_, cn_)):
    key = [k for k in ["task", "phrasing", "position", "hour", "lead_in_id", "condition"] if k in o.columns]
    m = o.merge(n, on=key, suffixes=("_32", "_128"))
    fin = m.raw_output_32.astype(str).str.len() < 100
    same = (m.raw_output_32.astype(str) == m.raw_output_128.astype(str))
    P(f"  {nm}: {len(m)} matched trials; finished at 32: {int(fin.sum())}, identical reply at 128: {int((fin & same).sum())}; truncated at 32: {int((~fin).sum())}, of which now different: {int((~fin & ~same).sum())}")

P("\n=== 3. TEXT-ONLY two-hand conversion at 128 vs 32 (conversion_test, condition text_only, n=432) ===")
rows = []
for tag, d in (("32 tokens (G1)", co_), ("128 tokens (G1b)", cn_)):
    t = d[d.condition == "text_only"]
    labs = pd.Series([ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute))[0] for r in t.itertuples()])
    k = int((labs == "leak").sum()); lo, hi = wilson(k, len(t)); kc = int(t.minute_correct.astype(bool).sum()); clo, chi = wilson(kc, len(t))
    rows.append({"run": tag, "n": len(t), "minute_correct": f"{kc} = {kc / len(t):.1%} [{clo:.1%},{chi:.1%}]", "leaks": f"{k} = {k / len(t):.1%} [{lo:.1%},{hi:.1%}]",
                 "unparsed": int((labs == "unparsed").sum()), "adjacent": int((labs == "adjacent").sum()), "default_0_30": int((labs == "default_0_30").sum()), "other": int((labs == "other").sum())})
P(pd.DataFrame(rows).to_string(index=False))
t_old, t_new = co_[co_.condition == "text_only"], cn_[cn_.condition == "text_only"]
key = ["position", "hour", "lead_in_id"]
mm = t_old.merge(t_new, on=key, suffixes=("_32", "_128"))
trunc = mm.raw_output_32.astype(str).str.len() >= 100
P(f"  the {int(trunc.sum())} trials truncated at 32: minute correct at 128 = {int(mm[trunc].minute_correct_128.astype(bool).sum())}/{int(trunc.sum())}; hour->minute leaks at 128 = "
  f"{sum(ML.label_answer(str(r.raw_output_128), int(r.hour), int(r.truth_minute_128))[0] == 'leak' for r in mm[trunc].itertuples())}; the other {int((~trunc).sum())}: correct {int(mm[~trunc].minute_correct_128.astype(bool).sum())}")

P("\n=== 4. TEXT-ONLY vs MATCHING vs BLANK vs MISMATCHED at 128 (exact McNemar; Holm over the 3 comparisons with text-only) ===")
conds = ["text_only", "matching_clock", "blank", "mismatched_clock"]
w = cn_.pivot_table(index=key, columns="condition", values="minute_correct", aggfunc="first").astype(bool)
long_key = cn_[(cn_.condition == "text_only") & (Ln >= 200)].set_index(key).index
hit_key = cn_[(cn_.condition == "text_only") & (Ln >= 430)].set_index(key).index
subsets = [("ALL 432 keys (every trial parsed)", pd.Series(True, index=w.index)),
           ("excluding the 48 keys whose text-only reply is a long chain-of-thought (>= 200 chars; these are the keys truncated at 32)", ~w.index.isin(long_key)),
           ("excluding the 6 keys that still hit the 128 cap", ~w.index.isin(hit_key))]
for nm, mask in subsets:
    sub = w[mask]
    P(f"\n--- {nm}: {len(sub)} keys ---")
    P("minute accuracy: " + "  ".join(f"{c} {sub[c].mean():.1%}" for c in conds))
    ps, rows = [], []
    for c in conds[1:]:
        b, cc = int((sub.text_only & ~sub[c]).sum()), int((~sub.text_only & sub[c]).sum())
        ps.append(binom_p(b, cc)); rows.append((c, b, cc, int((sub.text_only & sub[c]).sum()), int((~sub.text_only & ~sub[c]).sum())))
    order = sorted(range(3), key=lambda i: ps[i]); holm = [0] * 3; run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (3 - r))); holm[i] = run
    for (c, b, cc, both, nei), p, h in zip(rows, ps, holm):
        P(f"  text_only vs {c:<17} only-text-correct {b:>3}  only-image-correct {cc:>3}  both {both:>3}  neither {nei:>3}   p = {p:.3g}  Holm p = {h:.3g}")
    bm, bb = int((sub.matching_clock & ~sub.blank).sum()), int((~sub.matching_clock & sub.blank).sum())
    P(f"  matching vs blank: only-matching {bm}, only-blank {bb}  p = {binom_p(bm, bb):.3g}")
P("\nleaks by condition at 128: " + ", ".join(f"{cnd} {sum(ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute), None if pd.isna(r.image_minute) else int(r.image_minute))[0] == 'leak' for r in d.itertuples())}" for cnd, d in cn_.groupby('condition')))
P("leaks by condition at 32:  " + ", ".join(f"{cnd} {sum(ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute), None if pd.isna(r.image_minute) else int(r.image_minute))[0] == 'leak' for r in d.itertuples())}" for cnd, d in co_.groupby('condition')))
open(os.path.join(ROOT, a.out), "w", encoding="utf8").write("\n".join(out) + "\n")
