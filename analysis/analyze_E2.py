"""Analysis of E2 (cross-clock image-token patching) exactly as pre-registered (PREREG_MECHANISM.md Section 6 + deviations log).
usage (repo root):  python analysis/analyze_E2.py [--root results/mech_e2] [--models ...]
Reads <root>/<model>/e2_trials.csv ; writes <root>/E2_report.txt and <root>/<model>/e2_cells.csv.
Unit of analysis = the pair (100 per model); cluster bootstrap / sign-flip over pairs; Holm over layers x regions within model and variant.
Status: InternVL3-2B standard = CONFIRMATORY (its decision rules); Qwen-3B = EXPLORATORY (text-light floor 0.093 < 0.15); Qwen-7B / InternVL3-8B = DESCRIPTIVE (no rule);
the text_light variant of InternVL3-2B is the positive control for the Qwen-3B text-light rule."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import mech_lib as ML  # noqa: E402

OUT = []
STATUS = {"internvl3-2b": "CONFIRMATORY (standard variant); text_light = positive control", "qwen2.5-vl-3b": "EXPLORATORY (text-light floor fails)",
          "qwen2.5-vl-7b": "DESCRIPTIVE (no pre-registered rule)", "internvl3-8b": "DESCRIPTIVE (no pre-registered rule)"}


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def wide(d, variant, layer, sname, cond, col):
    x = d[(d.variant == variant) & (d.layer == layer) & (d.set == sname) & (d.cond == cond)].sort_values("pair")
    return x.set_index("pair")[col].astype(float)


def diff_stats(a, b):
    idx = a.index.intersection(b.index)
    v = (a.loc[idx] - b.loc[idx]).values
    cl = np.asarray(idx)
    m, lo, hi = ML.cluster_bootstrap_ci(v, cl, n_boot=5000, seed=0)
    return m, lo, hi, ML.sign_flip_p(v, cl, n_perm=5000, seed=0), len(idx)


def analyze(model, root):
    d = pd.read_csv(os.path.join(root, model, "e2_trials.csv"))
    p("\n" + "=" * 100 + f"\nE2  {model}   status: {STATUS.get(model, 'DESCRIPTIVE')}\n" + "=" * 100)
    for variant in sorted(d.variant.unique()):
        v = d[d.variant == variant]
        base = v[v.cond == "baseline"]
        p(f"\n--- variant {variant}: {v.pair.nunique()} pairs; unpatched A: stays_A {base.stays_A.mean():.3f}, follow_B {base.follow_B.mean():.3f}")
        rows = []
        layers = sorted(l for l in v.layer.unique() if l >= 0)
        for k in layers:
            for s in ("all", "minute_hand", "hour_hand", "other"):
                rb = wide(v, variant, k, s, "real_B", "follow_B")
                if not len(rb):
                    continue
                rc = wide(v, variant, k, s, "same_minute_C", "follow_B")
                m, lo, hi, pv, n = diff_stats(rb, rc)
                tb, tc = wide(v, variant, k, s, "real_B", "tf_B_minus_A"), wide(v, variant, k, s, "same_minute_C", "tf_B_minus_A")
                tm, tlo, thi, tp, _ = diff_stats(tb, tc)
                nz = wide(v, variant, k, s, "noise", "follow_B")
                rows.append({"layer": k, "set": s, "n_tokens_mean": v[(v.layer == k) & (v.set == s) & (v.cond == "real_B")].n_tokens.mean(), "follow_B_real": rb.mean(),
                             "follow_B_sameminute": rc.mean(), "follow_B_noise": nz.mean() if len(nz) else np.nan, "diff_pts": 100 * m, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                             "p": pv, "tf_diff_nats": tm, "tf_lo": tlo, "tf_hi": thi, "stays_A_real": wide(v, variant, k, s, "real_B", "stays_A").mean()})
        t = pd.DataFrame(rows)
        t["p_holm"] = ML.holm(t.p.values)
        t.insert(0, "variant", variant)
        t.to_csv(os.path.join(root, model, f"e2_cells_{variant}.csv"), index=False)
        show = t.copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].round(3)
        p("follow-B rate when the patch comes from clock B, minus the same-minute control (pts), CI over pairs, Holm over layers x regions:")
        p(show[["layer", "set", "n_tokens_mean", "follow_B_real", "follow_B_sameminute", "follow_B_noise", "diff_pts", "ci_lo", "ci_hi", "p_holm", "tf_diff_nats", "stays_A_real"]].to_string(index=False))
        # decision rules
        mh, oth, al = t[t.set == "minute_hand"], t[t.set == "other"], t[t.set == "all"]
        if model == "internvl3-2b" and variant == "standard":
            hits = []
            for r in mh.itertuples():
                o = oth[oth.layer == r.layer]
                if len(o) and r.diff_pts >= 5 and r.ci_lo > 0 and r.p_holm < 0.05:
                    # must also exceed the other-region patch at that layer
                    a = wide(v, variant, r.layer, "minute_hand", "real_B", "follow_B") - wide(v, variant, r.layer, "minute_hand", "same_minute_C", "follow_B")
                    b = wide(v, variant, r.layer, "other", "real_B", "follow_B") - wide(v, variant, r.layer, "other", "same_minute_C", "follow_B")
                    m, lo, hi, pv, n = diff_stats(a, b)
                    if lo > 0:
                        hits.append(int(r.layer))
            p(f"\nRULE (2B reads minute evidence from the minute-hand region): layers where minute-hand patching raises follow-B by >= 5 pts over the same-minute control (CI excl 0, Holm) "
              f"AND exceeds the other-region patch: {hits or 'none'} => {'SUPPORTED' if hits else 'NOT supported'}")
            ub = float(mh.ci_hi.max()) if len(mh) else np.nan
            only_all = bool(len(al) and (al.diff_pts.max() >= 5) and (ub < 3))
            p(f"RULE (cue/consistency account): only all-token (or non-hand) patching moves the answer, minute-hand upper bound < 3 pts: max minute-hand CI upper {ub:.1f} pts, "
              f"max all-token diff {al.diff_pts.max():.1f} pts => {'SUPPORTED' if only_all else 'not supported'}")
        if variant == "text_light":
            ub_all = float(al.ci_hi.max()) if len(al) else np.nan
            best = mh.sort_values("diff_pts", ascending=False).head(1)
            rule10 = bool(len(best) and (best.diff_pts.iloc[0] >= 10) and (best.ci_lo.iloc[0] > 0) and (best.p_holm.iloc[0] < 0.05))
            p(f"\nTEXT-LIGHT: strongest minute-hand cell: layer {int(best.layer.iloc[0])} diff {best.diff_pts.iloc[0]:+.1f} pts CI [{best.ci_lo.iloc[0]:+.1f}, {best.ci_hi.iloc[0]:+.1f}], Holm p {best.p_holm.iloc[0]:.3g}; "
              f"rule (>= +10 pts, CI excl 0, Holm) => {'MET' if rule10 else 'not met'}; all-token upper bound (max over layers) {ub_all:.1f} pts "
              f"({'< 5: a claim of no image-driven shift would need this AND the 2B positive control >= 20 pts' if ub_all < 5 else '>= 5'})")
            if model == "qwen2.5-vl-3b":
                p("   (Qwen-3B text-light: the pre-declared floor rule is NOT passed (pilot 0.093 < 0.15), so this is reported as EXPLORATORY regardless of the numbers.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "mech_e2"))
    ap.add_argument("--models", default="internvl3-2b,qwen2.5-vl-7b,internvl3-8b,qwen2.5-vl-3b")
    a = ap.parse_args()
    for m in a.models.split(","):
        if os.path.exists(os.path.join(a.root, m, "e2_trials.csv")):
            analyze(m, a.root)
    open(os.path.join(a.root, "E2_report.txt"), "w", encoding="utf-8").write("\n".join(OUT))
    print("\nwrote", os.path.join(a.root, "E2_report.txt"))
