"""Analysis of E1 (windowed text->image attention knockout) exactly as pre-registered (PREREG_MECHANISM.md Section 5 + deviations log).
usage (repo root):  python analysis/analyze_E1.py [--root results/mech_e1] [--models internvl3-2b,...]
Reads  <root>/<model>/e1_trials.csv, e1_machinery.json, e1_followup.json  and  results/<model>/conversion_test.csv (the model's own text-only run).
Writes <root>/E1_report.txt and <root>/<model>/e1_windows.csv.
Status labels: CONFIRMATORY only for InternVL3-2B and only when the machinery check passes for that query set; everything else is EXPLORATORY/DESCRIPTIVE."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import mech_lib as ML  # noqa: E402

OUT = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def cell(d, cond, q, name):
    x = d[(d.cond == cond) & (d.qset == q) & (d.cell == name)].sort_values(["lead", "N", "H"]).reset_index(drop=True)
    return x


def paired(base, test, col):
    """per-trial (base - test) for a boolean column, aligned on (lead,N,H); returns mean, lo, hi, p, n."""
    m = base.merge(test, on=["lead", "N", "H"], suffixes=("_b", "_t"))
    v = m[col + "_b"].astype(float).values - m[col + "_t"].astype(float).values
    cl = (m.N.astype(str) + "_" + m.H.astype(str)).values
    mean, lo, hi = ML.cluster_bootstrap_ci(v, cl, n_boot=5000, seed=0)
    return mean, lo, hi, ML.sign_flip_p(v, cl, n_perm=5000, seed=0), len(m)


def analyze(model, root):
    d = pd.read_csv(os.path.join(root, model, "e1_trials.csv"))
    d["qset"] = d.qset.astype(str)
    mach = json.load(open(os.path.join(root, model, "e1_machinery.json"))) if os.path.exists(os.path.join(root, model, "e1_machinery.json")) else {}
    p("\n" + "=" * 100 + f"\nE1  {model}   ({'CONFIRMATORY candidate' if model == 'internvl3-2b' else 'EXPLORATORY/DESCRIPTIVE (see prereg Section 5)'})\n" + "=" * 100)
    # intact numbers
    ia = cell(d, "matching_clock", "-", "intact")
    ie = cell(d, "matching_clock", "-", "intact_eager")
    ib = cell(d, "blank", "-", "intact")
    im = cell(d, "mismatched_clock", "-", "intact")
    to = pd.read_csv(os.path.join("results", model, "conversion_test.csv"))
    to = to[to.condition == "text_only"]
    keys = set(zip(ia.lead, ia.N, ia.H))
    to = to[[(int(l), int(n), int(h)) in keys for l, n, h in zip(to.lead_in_id, to.position, to.hour)]]
    T = ia.minute_correct.mean() - to.minute_correct.astype(bool).mean()
    p(f"intact matching_clock minute acc {ia.minute_correct.mean():.3f} (n={len(ia)}); eager intact {ie.minute_correct.mean() if len(ie) else float('nan'):.3f}; "
      f"text-only {to.minute_correct.astype(bool).mean():.3f}; blank {ib.minute_correct.mean() if len(ib) else float('nan'):.3f}; "
      f"total image effect T = {100 * T:+.1f} pts; mismatched intact follows_image {int(im.follows_image.sum()) if len(im) else 'NA'} of {len(im)}")
    valid = {}
    for q, v in mach.items():
        valid[q] = bool(v["knockout_valid"])
        p(f"MACHINERY CHECK query set {q}: blocked-all matching {v['blocked_all_matching_acc']:.3f} vs text-only {v['text_only_acc']:.3f} -> "
          f"|diff| {v['abs_diff_points']:.1f} pts (<= 5 required) => {'VALID' if v['knockout_valid'] else 'INVALID: all E1 results of this model/query set are EXPLORATORY ONLY'}")
    rows = []
    for q in sorted(x for x in d.qset.unique() if x != "-"):
        sub = d[(d.qset == q) & (d.cond == "matching_clock")]
        for name in sub.cell.unique():
            c = cell(d, "matching_clock", q, name)
            m, lo, hi, pv, n = paired(ia, c, "minute_correct")
            rows.append({"model": model, "qset": q, "cell": name, "start": int(c.start.iloc[0]), "end": int(c["end"].iloc[0]), "n": n, "acc": c.minute_correct.mean(),
                         "drop_pts": 100 * m, "ci_lo": 100 * lo, "ci_hi": 100 * hi, "p": pv, "frac_of_T": (m / T) if T else np.nan})
    w = pd.DataFrame(rows)
    if len(w):
        sl = w.cell.str.startswith("w")
        for q in w.qset.unique():
            idx = w.index[(w.qset == q) & sl]
            if len(idx):
                w.loc[idx, "p_holm"] = ML.holm(w.loc[idx, "p"].values)
        w.to_csv(os.path.join(root, model, "e1_windows.csv"), index=False)
        show = w.copy()
        for c in ("acc", "drop_pts", "ci_lo", "ci_hi", "frac_of_T", "p", "p_holm"):
            if c in show:
                show[c] = show[c].astype(float).round(3)
        p("\nmatching_clock: drop in minute accuracy vs the intact run (positive = blocking hurts); cluster bootstrap over (N,H); Holm over the sliding windows")
        p(show.to_string(index=False))
    # H-E1-1
    for q in ("B", "A"):
        if q not in valid and q not in w.qset.values:
            continue
        sw = w[(w.qset == q) & w.cell.str.startswith("w")] if len(w) else pd.DataFrame()
        if not len(sw):
            continue
        lit = sw[(sw.drop_pts >= 11) & (sw.ci_lo > 0) & (sw.p_holm < 0.05)]
        rel = sw[(sw.frac_of_T >= 0.5) & (sw.ci_lo > 0) & (sw.p_holm < 0.05)]
        tag = "PRIMARY" if q == "B" else "reported alongside, not used for decisions"
        ok = valid.get(q, False) and model == "internvl3-2b"
        p(f"\nH-E1-1 (localised evidence), query set {q} [{tag}]; status {'CONFIRMATORY' if ok else 'EXPLORATORY'}: windows with drop >= 11 pts (CI excl 0, Holm p<.05): "
          f"{lit.cell.tolist() or 'none'}; windows removing >= 50% of T={100 * T:.1f}: {rel.cell.tolist() or 'none'}  ->  "
          f"{'SUPPORTED' if len(lit) else 'NOT supported: evidence DISTRIBUTED across layers (a result, not a failure)'}")
    # H-E1-2
    fu = os.path.join(root, model, "e1_followup.json")
    if os.path.exists(fu):
        sel = json.load(open(fu))
        for q, s in sel.items():
            cm = cell(d, "mismatched_clock", q, s["window"])
            if len(cm) and len(im):
                m, lo, hi, pv, n = paired(im, cm, "follows_image")
                base_n, blk_n = int(im.follows_image.sum()), int(cm.follows_image.sum())
                rel = (base_n - blk_n) / base_n if base_n else float("nan")
                ok = valid.get(q, False) and model == "internvl3-2b"
                p(f"H-E1-2 (replication), query set {q}: best matching window {s['window']} (selected on matching_clock; drop {100 * s['matching_drop']:.1f} pts); mismatched follows_image "
                  f"{base_n} -> {blk_n} ({100 * rel:.0f}% relative drop; paired diff {100 * m:+.1f} pts CI [{100 * lo:+.1f}, {100 * hi:+.1f}]) => "
                  f"{'SUPPORTED' if (rel >= 0.5 and lo > 0) else 'not supported'}; status {'CONFIRMATORY' if ok else 'EXPLORATORY'}")
    # H-E1-3
    for q in ("B", "A"):
        f4 = cell(d, "matching_clock", q, "final4")
        al = cell(d, "matching_clock", q, "all")
        if len(f4):
            m, lo, hi, pv, n = paired(ia, f4, "minute_correct")
            p(f"H-E1-3 (specificity), query set {q}: final-4-layers window changes matching_clock accuracy by {100 * m:+.1f} pts (CI [{100 * lo:+.1f}, {100 * hi:+.1f}]) -> "
              f"{'within 3 pts' if abs(100 * m) < 3 else 'NOT within 3 pts'}")
        if len(al) and model != "internvl3-2b":
            m, lo, hi, pv, n = paired(ia, al, "minute_correct")
            p(f"   all-layers block on {model}: {100 * m:+.1f} pts -> {'< 3 pts (image-robust statement holds)' if abs(100 * m) < 3 else 'NOT < 3 pts (weakens the image-robust statement)'}")
    # cumulative curve (descriptive)
    cum = w[(w.qset == "B") & w.cell.str.startswith("cum")][["cell", "start", "acc", "drop_pts"]] if len(w) else pd.DataFrame()
    if len(cum):
        p("\ncumulative-from-L-to-end, query set B (descriptive): " + "; ".join(f"L{int(r.start)}: acc {r.acc:.3f}" for r in cum.itertuples()))
    # other conditions
    oth = d[(d.cond != "matching_clock") & (d.qset != "-")].groupby(["cond", "qset", "cell"], sort=False).agg(acc=("minute_correct", "mean"), follows_image=("follows_image", "mean"),
                                                                                                         n=("minute_correct", "size")).round(3)
    if len(oth):
        p("\nblank / mismatched_clock cells (descriptive):"); p(oth.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "mech_e1"))
    ap.add_argument("--models", default=",".join(m for m in ("internvl3-2b", "qwen2.5-vl-7b", "internvl3-8b", "qwen2.5-vl-3b")))
    a = ap.parse_args()
    for m in a.models.split(","):
        if os.path.exists(os.path.join(a.root, m, "e1_trials.csv")):
            analyze(m, a.root)
    open(os.path.join(a.root, "E1_report.txt"), "w", encoding="utf-8").write("\n".join(OUT))
    print("\nwrote", os.path.join(a.root, "E1_report.txt"))
