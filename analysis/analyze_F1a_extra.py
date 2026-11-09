"""Extra E2 analysis for InternVL3-2B (F1a): (1) minute-hand vs other-region contrast with CIs (the pre-registered 'exceeds other region' clause, shown explicitly),
(2) per-token efficiency, (3) where do answers go (A / B / other / unparsed; circular minute distance to B vs A), (4) share of the all-token effect carried by the minute-hand tokens.
usage: python analysis/analyze_F1a_extra.py   -> results/mech_e2/F1a_extra_report.txt   (descriptive; the confirmatory rules are in analyze_E2.py)"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import mech_lib as ML
OUT = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)
d = pd.read_csv("results/mech_e2/internvl3-2b/e2_trials.csv")
def w(v, k, s, c, col):
    x = v[(v.layer == k) & (v.set == s) & (v.cond == c)].sort_values("pair")
    return x.set_index("pair")[col].astype(float)
def contrast(a, b):
    idx = a.index.intersection(b.index); v = (a.loc[idx] - b.loc[idx]).values; cl = np.asarray(idx)
    m, lo, hi = ML.cluster_bootstrap_ci(v, cl, n_boot=5000, seed=0)
    return 100 * m, 100 * lo, 100 * hi, ML.sign_flip_p(v, cl, n_perm=5000, seed=0)
for variant in ("standard", "text_light"):
    v = d[d.variant == variant]
    p(f"\n=== {variant} ===")
    p("(1) minute_hand effect minus other-region effect (each = follow-B real_B minus same-minute control), pts [CI over pairs], and (4) minute-hand share of all-token effect")
    rows = []
    for k in sorted(l for l in v.layer.unique() if l >= 0):
        eff = {s: w(v, k, s, "real_B", "follow_B") - w(v, k, s, "same_minute_C", "follow_B") for s in ("all", "minute_hand", "hour_hand", "other")}
        m, lo, hi, pv = contrast(eff["minute_hand"], eff["other"])
        mh = 100 * eff["minute_hand"].mean(); al = 100 * eff["all"].mean()
        nt = v[(v.layer == k) & (v.cond == "real_B")].groupby("set").n_tokens.mean()
        rows.append({"layer": k, "min_minus_other": round(m, 1), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1), "p": round(pv, 4), "minute_share_of_all": round(mh / al, 2) if al > 0 else np.nan,
                     "pts_per_token_minute": round(mh / nt["minute_hand"], 3), "pts_per_token_other": round(100 * eff["other"].mean() / nt["other"], 3)})
    p(pd.DataFrame(rows).to_string(index=False))
    p("(3) where do answers go (real_B, mean over pairs): share A / B / other / unparsed; mean circular minute distance (minutes) to B and to A; unpatched baseline first")
    rows = []
    pairs = v[v.cond == "baseline"].set_index("pair")
    for (k, s, c), g in [((-1, "none", "baseline"), v[v.cond == "baseline"])] + [((k, s, "real_B"), v[(v.layer == k) & (v.set == s) & (v.cond == "real_B")]) for k in (0, 10, 17, 21) for s in ("all", "minute_hand", "other")]:
        pm = g.pred_minute.values.astype(float); na = g.n_a.values * 5.0; nb = g.n_b.values * 5.0
        ok = ~np.isnan(pm)
        circ = lambda x, y: np.minimum(np.abs(x - y) % 60, 60 - np.abs(x - y) % 60)
        isA = g.stays_A.mean(); isB = g.follow_B.mean()
        rows.append({"layer": k, "set": s, "cond": c, "stays_A": round(isA, 2), "follow_B": round(isB, 2), "other": round(1 - isA - isB - (1 - ok.mean()), 2), "unparsed": round(1 - ok.mean(), 2),
                     "dist_to_B": round(float(circ(pm[ok], nb[ok]).mean()), 1), "dist_to_A": round(float(circ(pm[ok], na[ok]).mean()), 1)})
    p(pd.DataFrame(rows).to_string(index=False))
open("results/mech_e2/F1a_extra_report.txt", "w", encoding="utf-8").write("\n".join(OUT))
