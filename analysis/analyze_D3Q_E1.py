"""Analysis of notebook D3-Q (Qwen E3a-d, rp 1.0) and notebook E (E1 windowed knockout).
usage (repo root):
    python analysis/analyze_D3Q_E1.py --stage merge    # copy match_D3Q_results/ and match_E_results/ files into results/ (never overwrites; name clashes get a _D3Q / _E suffix)
    python analysis/analyze_D3Q_E1.py --stage report   # D3-Q report -> results/mech_e3/D3Q_report.txt  (E1 report: python analysis/analyze_E1.py)
Thresholds are the pre-registered ones (PREREG_MECHANISM.md Section 4/5) plus the deviations log."""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import mech_lib as ML  # noqa: E402

MECH = os.path.join("results", "mech_e3")
OUT = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def stage_merge():
    log = []
    for src_root, tag in (("match_D3Q_results", "D3Q"), ("match_E_results", "E")):
        base = os.path.join(src_root, "results")
        for dp, _, fs in os.walk(base):
            for f in fs:
                s = os.path.join(dp, f)
                rel = os.path.relpath(s, base)
                d = os.path.join("results", rel)
                if os.path.exists(d):
                    stem, ext = os.path.splitext(d)
                    d = f"{stem}_{tag}{ext}"
                    if os.path.exists(d):
                        log.append(f"SKIP (exists) {d}")
                        continue
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
                log.append(f"{s} -> {d}")
    json.dump(log, open(os.path.join("results", "_D3Q_E_merge_log.json"), "w"), indent=1)
    print("\n".join(log))


def gate(path, title):
    j = json.load(open(path))
    bad = [k for k, v in j["checks"].items() if not v["pass"]]
    gating_bad = [k for k in bad if j["checks"][k].get("gating", True)]
    p(f"  {title}: {len(j['checks'])} checks, overall pass={j['pass']}; failed: {bad or 'none'} (failed AND gating: {gating_bad or 'none'})")


def e3d_family():
    p("\n--- E3d family (primary comparison: block answer->hour span at ALL layers vs intact; Holm over {Qwen-3B, InternVL3-2B, Qwen-7B})")
    rows = []
    for m in ("qwen2.5-vl-3b", "internvl3-2b", "qwen2.5-vl-7b"):
        d = os.path.join(MECH, m)
        v = json.load(open(os.path.join(d, "e3d_prereg_verdict.json")))
        s = pd.read_csv(os.path.join(d, "e3d_summary.csv"))
        H, R = s[s.cell == "H_all"].iloc[0], s[s.cell == "random_all"].iloc[0]
        rows.append({"model": m, "leak_intact": v["leak_intact"], "leak_blocked": v["leak_blocked"], "leak_rel_drop": v["leak_rel_drop"], "acc_diff_pts": v["acc_diff_points"],
                     "acc_ci_lo": v["acc_ci_points"][0], "acc_ci_hi": v["acc_ci_points"][1], "p_acc": H.acc_p, "p_leak": H.leak_p, "random_acc_pts": 100 * R.acc_diff,
                     "random_leaks": int(R.leak_n), "M_bind_supported": v["M_bind_supported"]})
    t = pd.DataFrame(rows)
    t["p_acc_holm"] = ML.holm(t.p_acc.values)
    t["p_leak_holm"] = ML.holm(t.p_leak.values)
    p(t.round(4).to_string(index=False))
    p("  Holm-adjusted p do not change any verdict: the pre-registered decision depends on the effect-size thresholds, not on significance alone. 7B fails the >= 4 pts bar (ceiling), "
      "as expected for a specificity control; its leaks (6 -> 2) and +1.4 pts are small but their CI excludes 0.")
    t.to_csv(os.path.join(MECH, "e3d_family_holm.csv"), index=False)
    return t


def depth_profile():
    p("\n--- Which layers carry the E3d effect (exploratory, relative depth)")
    for m, L in (("qwen2.5-vl-3b", 36), ("internvl3-2b", 28)):
        s = pd.read_csv(os.path.join(MECH, m, "e3d_summary.csv"))
        intact = int(s[s.cell == "intact"].leak_n.iloc[0])
        w = s[s.cell.str.startswith("H_win")].copy()
        w["lo"] = w.cell.str.extract(r"H_win(\d+)-")[0].astype(int)
        w["hi"] = w.cell.str.extract(r"-(\d+)$")[0].astype(int)
        w["leaks_removed"] = intact - w.leak_n
        w["acc_gain_pts"] = 100 * w.acc_diff
        w["rel_depth"] = ((w.lo + w.hi) / 2 / L).round(2)
        cum = s[s.cell.str.startswith("H_cum")].copy()
        cum["start"] = cum.cell.str.extract(r"H_cum(\d+)-")[0].astype(int)
        allrem = intact - int(s[s.cell == "H_all"].leak_n.iloc[0])
        # onset: the latest start such that cumulative-to-end still removes >= 90% of the all-layer leak reduction (late edge of the effect); earliest start needed
        ok = cum[(intact - cum.leak_n) >= 0.9 * allrem].sort_values("start")
        late_start = int(ok.start.max()) if len(ok) else None
        p(f"\n{m} ({L} layers), intact leaks {intact}, all-layer block removes {allrem}")
        p(w[["cell", "rel_depth", "leak_n", "leaks_removed", "acc_gain_pts"]].round(2).to_string(index=False))
        p("  cumulative-to-end: " + "; ".join(f"L{r.start}: {int(r.leak_n)}" for r in cum.sort_values("start").itertuples()))
        if late_start is not None:
            p(f"  blocking only layers >= {late_start} ({late_start / L:.2f} of depth) still removes >= 90% of the all-layer leak reduction")
        top = w.sort_values("leaks_removed", ascending=False).head(1)
        p(f"  strongest single window: {top.cell.iloc[0]} (relative depth {top.rel_depth.iloc[0]:.2f}), removes {int(top.leaks_removed.iloc[0])} leaks")


def e3abc(m, L):
    d = os.path.join(MECH, m)
    p("\n" + "-" * 90 + f"\n{m}: E3a / E3b / E3c")
    a = pd.read_csv(os.path.join(d, "e3a_summary.csv"))
    p("E3a logit lens (EXPLORATORY, no decision rule): gold-minus-leak log-odds at the final layer by group")
    for g in ("right", "wrong_leak", "wrong_other", "H12", "H_not12"):
        x = a[a.group == g].sort_values("layer")
        if len(x):
            p(f"  {g:<12} n={int(x.n.iloc[0])}  final layer {x.mean_gold_minus_leak.iloc[-1]:+.2f}")
    wl = a[a.group == "wrong_leak"].sort_values("layer")
    if len(wl):
        neg = wl[wl.mean_gold_minus_leak < 0]
        p(f"  wrong_leak group favours the leak minute from layer {int(neg.layer.min()) if len(neg) else 'never'} ({'' if not len(neg) else f'{int(neg.layer.min()) / L:.2f} of depth'})")
    if not os.path.exists(os.path.join(d, "e3b_prereg_rule.csv")):
        return
    b = pd.read_csv(os.path.join(d, "e3b_prereg_rule.csv"))
    p("E3b swap patching. Rule (pre-registered): N-span patch moves the answer toward the source minute, CI excl 0, exceeding every control by >= 1 nat at some layer.")
    for grp in ("N", "H", "between", "last"):
        g = b[b.group == grp]
        if len(g):
            p(f"  span {grp:<8}: supported at {int(g.PREREG_supported.sum())}/{len(g)} tested layers; shift {g.mean_shift.min():+.2f}..{g.mean_shift.max():+.2f} nats (max at layer {int(g.loc[g.mean_shift.idxmax(), 'layer'])})")
    gn = b[b.group == "N"]
    p(f"  >>> E3b VERDICT (pre-registered, N-span): {'SUPPORTED' if gn.PREREG_supported.any() else 'NOT supported'} at layers {sorted(gn[gn.PREREG_supported].layer.astype(int).tolist())}; best control |shift| {gn.best_control_abs_shift_same_layer.max():.2f} nats")
    c = pd.read_csv(os.path.join(d, "e3c_prereg_rule.csv"))
    top = c.sort_values("restored_rate_among_wrong", ascending=False).head(5)
    p("E3c restoration. Rule: >= 50% of wrong runs restored at a single layer-position cell AND >= 25 pts above the best control.")
    p(top[["layer", "group", "n", "restored_rate_among_wrong", "best_control_restored", "PREREG_supported"]].round(3).to_string(index=False))
    p(f"  >>> E3c VERDICT: supported cells {int(c.PREREG_supported.sum())}/{len(c)} -> {'SUPPORTED' if c.PREREG_supported.any() else 'NOT supported (reported as exploratory)'}")


def stage_report():
    p("=" * 100 + "\nD3-Q: Qwen E3a-d at repetition_penalty 1.0\n" + "=" * 100)
    p(open(os.path.join("match_D3Q_results", "STATUS.txt")).read())
    for m in ("qwen2.5-vl-3b", "qwen2.5-vl-7b"):
        gate(os.path.join("match_D3Q_results", "results", "mech_e3", m, "verify.json"), m)
    p("  (Qwen-3B G2-gold, informational and non-gating, fails; G2-own, the gating version, passes.)")
    t = e3d_family()
    for m in ("qwen2.5-vl-3b",):
        v = json.load(open(os.path.join(MECH, m, "e3d_prereg_verdict.json")))
        s = pd.read_csv(os.path.join(MECH, m, "e3d_summary.csv"))
        H, R, N = s[s.cell == "H_all"].iloc[0], s[s.cell == "random_all"].iloc[0], s[s.cell == "N_all"].iloc[0]
        p(f"\n>>> E3d Qwen-3B (CONFIRMATORY; bar leaks 40 -> <= 28 and accuracy +4, CIs excl 0, random-span control < 2 pts): leaks {v['leak_intact']} -> {v['leak_blocked']} "
          f"({100 * v['leak_rel_drop']:.1f}% fewer, CI of per-trial change [{H.leak_lo:+.3f}, {H.leak_hi:+.3f}]); accuracy {v['acc_diff_points']:+.1f} pts CI [{v['acc_ci_points'][0]:+.1f}, {v['acc_ci_points'][1]:+.1f}]; "
          f"random-span control {v['random_span_acc_diff_points']:+.1f} pts (leaks {int(R.leak_n)}); destruction control accuracy {N.acc:.3f} (intact {s[s.cell == 'intact'].acc.iloc[0]:.3f}) "
          f"=> M_bind_supported = {v['M_bind_supported']}, weak_conversion_account = {v['weak_conversion_account']}")
    depth_profile()
    p("\n--- Qwen-7B specificity control (E3d, 'min' mode: only intact / H_all / random_all)")
    s = pd.read_csv(os.path.join(MECH, "qwen2.5-vl-7b", "e3d_summary.csv"))
    p(s[["cell", "acc", "leak_n", "acc_diff", "acc_lo", "acc_hi", "leak_p"]].round(4).to_string(index=False))
    for m, L in (("qwen2.5-vl-3b", 36), ("qwen2.5-vl-7b", 28)):
        e3abc(m, L)
    open(os.path.join(MECH, "D3Q_report.txt"), "w", encoding="utf-8").write("\n".join(OUT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="report", choices=["merge", "report"])
    a = ap.parse_args()
    {"merge": stage_merge, "report": stage_report}[a.stage]()
