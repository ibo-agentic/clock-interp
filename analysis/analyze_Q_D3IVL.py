"""Reproducible analysis of the D2'' gate (Qwen at repetition_penalty 1.0), notebook Q (Qwen regenerated at 1.0) and D3-IVL (InternVL3-2B E3a-d).

usage (repo root):
    python analysis/analyze_Q_D3IVL.py --stage merge       # backup, snapshot the rp1.05 data, rename old Qwen files *_rp105, merge new files (never overwrites)
    python analysis/analyze_Q_D3IVL.py --stage aggregate   # text_only_v2 / conversion_test aggregates, compare_families, selection_bias_check at rp 1.0 (+ rp1.05 sensitivity table)
    python analysis/analyze_Q_D3IVL.py --stage report      # all tables / verdicts -> results/mech_e3/Q_D3IVL_report.txt
    python analysis/analyze_Q_D3IVL.py                     # all three stages in order
Inputs: match_D2pp_results/, match_Q_results/, match_D3IVL_results/ (the unzipped Kaggle outputs).
Thresholds are the pre-registered ones (PREREG_MECHANISM.md Section 4) and the 2026-11-06 deviations-log entries; nothing here changes them."""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import mech_lib as ML  # noqa: E402

QW = ["qwen2.5-vl-3b", "qwen2.5-vl-7b"]
IV = ["internvl3-2b", "internvl3-8b"]
ALL4 = QW + IV
ROOTS = ["results", "results_v2/results", "conv_results/results", "qwen_prior/results"]       # compare_families lookup order
FILES = ["results.csv", "describe.csv", "text_only_v2.csv", "conversion_test.csv"]
SENS = os.path.join("results", "_sens_rp105")
BACKUP = os.path.join("results", "_backup_pre_Q")
MECH = os.path.join("results", "mech_e3")
OUT = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def rp105_name(f):
    stem, ext = os.path.splitext(f)
    return f"{stem}_rp105{ext}"


# ------------------------------------------------------------------------------------------------ stage: merge
def stage_merge():
    log = []
    # 0. backups of generated aggregates (they will be regenerated)
    os.makedirs(BACKUP, exist_ok=True)
    for d in ("text_only_v2", "conversion_test", "comparison"):
        src, dst = os.path.join("results", d), os.path.join(BACKUP, d)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst); log.append(f"backup {src} -> {dst}")
    # 1. snapshot of the rp 1.05 state under NORMAL file names (so compare_families can be run on it for the sensitivity table)
    if not os.path.isdir(SENS):
        for m in QW:
            os.makedirs(os.path.join(SENS, m), exist_ok=True)
            for f in FILES:
                for r in ROOTS:
                    s = os.path.join(r, m, f)
                    if os.path.exists(s):
                        shutil.copy2(s, os.path.join(SENS, m, f)); log.append(f"snapshot {s} -> {SENS}/{m}/{f}")
                        break
        for m in IV:
            os.makedirs(os.path.join(SENS, m), exist_ok=True)
            for f in FILES:
                for r in ROOTS:
                    s = os.path.join(r, m, f)
                    if os.path.exists(s):
                        shutil.copy2(s, os.path.join(SENS, m, f)); break
        # 3B legacy describe (positions) is not needed: results/qwen2.5-vl-3b/describe.csv exists
    # 2. rename every existing Qwen per-model file in the pipeline roots to *_rp105*
    n_ren = 0
    for r in ROOTS:
        for m in QW:
            d = os.path.join(r, m)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if "_rp105" in f or not os.path.isfile(os.path.join(d, f)):
                    continue
                t = os.path.join(d, rp105_name(f))
                if os.path.exists(t):
                    raise SystemExit(f"refusing to overwrite {t}")
                os.rename(os.path.join(d, f), t); n_ren += 1
    # same for the D4 (rp 1.05) mechanism hook-free outputs
    for m in QW:
        d = os.path.join(MECH, m)
        for f in ("e3e_summary.csv", "e3e_transplant.csv", "e3f_forms.csv", "e3f_summary.csv", "text_light_pilot.csv", "text_light_pilot_summary.csv"):
            s = os.path.join(d, f)
            if os.path.exists(s):
                t = os.path.join(d, rp105_name(f))
                if os.path.exists(t):
                    raise SystemExit(f"refusing to overwrite {t}")
                os.rename(s, t); n_ren += 1
    log.append(f"renamed {n_ren} existing Qwen files to *_rp105*")
    # 3. merge the new files (no clobber)
    n_new, n_skip = 0, 0
    for m in QW:
        for sub, dst in ((os.path.join("match_Q_results", "results", m), os.path.join("results", m)),
                         (os.path.join("match_Q_results", "results", "mech_e3", m), os.path.join(MECH, m))):
            os.makedirs(dst, exist_ok=True)
            for f in sorted(os.listdir(sub)):
                t = os.path.join(dst, f)
                if os.path.exists(t):
                    n_skip += 1; log.append(f"SKIP (exists) {t}")
                else:
                    shutil.copy2(os.path.join(sub, f), t); n_new += 1
        for f, new in (("verify.json", "verify_D2pp_rp1.json"), ("verify_g2_trials.csv", "verify_g2_trials_D2pp_rp1.csv")):
            s, t = os.path.join("match_D2pp_results", "results", "mech_e3", m, f), os.path.join(MECH, m, new)
            if os.path.exists(s) and not os.path.exists(t):
                shutil.copy2(s, t); n_new += 1
    m = "internvl3-2b"
    sub = os.path.join("match_D3IVL_results", "results", "mech_e3", m)
    for f in sorted(os.listdir(sub)):
        t = os.path.join(MECH, m, f)
        if os.path.exists(t):
            n_skip += 1; log.append(f"SKIP (exists) {t}")
        else:
            shutil.copy2(os.path.join(sub, f), t); n_new += 1
    log.append(f"merged {n_new} new files, skipped {n_skip} that already existed")
    json.dump(log, open(os.path.join("results", "_Q_merge_log.json"), "w"), indent=1)
    print("\n".join(log[-6:]))
    print("full log: results/_Q_merge_log.json")


# ------------------------------------------------------------------------------------------------ stage: aggregate
def run(cmd, out=None):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if out:
        open(out, "w", encoding="utf-8").write(r.stdout + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def stage_aggregate():
    py = sys.executable
    run([py, "text_only_v2.py", "--aggregate"])
    run([py, "conversion_test.py", "--aggregate"])
    run([py, "compare_families.py"])
    run([py, os.path.join("analysis", "selection_bias_check.py")], out=os.path.join("results", "comparison", "selection_bias_check_output.txt"))
    # sensitivity (rp 1.05) table from the snapshot
    run([py, "compare_families.py", "--roots", SENS, "--out_dir", os.path.join("results", "comparison_rp105")])
    print("aggregates done: results/text_only_v2, results/conversion_test, results/comparison (rp 1.0), results/comparison_rp105 (sensitivity)")


# ------------------------------------------------------------------------------------------------ report helpers
def gate_table(path, title):
    v = json.load(open(path))
    p(f"\n{title}: GATE {'PASSED' if v['pass'] else 'FAILED'}   (rep_penalty arg {v.get('rep_penalty_arg')}, checkpoint default {v.get('checkpoint_default_repetition_penalty')})")
    for k, c in v["checks"].items():
        tag = ("PASS" if c["pass"] else "FAIL") if c.get("gating", True) else ("info:pass" if c["pass"] else "info:FAIL")
        info = {a: b for a, b in c.items() if a not in ("pass", "gating", "note")}
        p(f"  [{tag:<9}] {k}  {str(info)[:420]}")


def sens_table():
    new = pd.read_csv(os.path.join("results", "comparison", "four_model_long.csv"))
    old = pd.read_csv(os.path.join("results", "comparison_rp105", "four_model_long.csv"))
    rows = []
    for m in QW:
        for metric in new.metric.unique():
            a = new[(new.model == m) & (new.metric == metric)].iloc[0]
            b = old[(old.model == m) & (old.metric == metric)].iloc[0]
            is_p = metric.startswith("conv_mcnemar")
            d = None if (pd.isna(a.value) or pd.isna(b.value)) else float(a.value - b.value)
            flag = ""
            if is_p:
                if not (pd.isna(a.value) or pd.isna(b.value)) and ((a.value < 0.05) != (b.value < 0.05)):
                    flag = "SIGNIFICANCE FLIPS"
            elif d is not None and abs(d) * 100 > 3:
                flag = ">3 pts"
            rows.append({"model": m, "metric": metric, "rp1.0": a.value, "n_1.0": a.n, "k_1.0": a.k, "rp1.05": b.value, "n_1.05": b.n, "k_1.05": b.k,
                         "delta": d, "flag": flag})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join("results", "comparison", "sensitivity_rp1.0_vs_rp1.05.csv"), index=False)
    return t


def prereg_counts():
    """both-exact clock sets (describe) and conversion-test text-only leak counts, at rp 1.0 vs the rp 1.05 snapshot."""
    rows = []
    for tag, rootdir in (("rp1.0", "results"), ("rp1.05", SENS)):
        for m in ALL4:
            if tag == "rp1.05" and m in IV:
                continue
            d = pd.read_csv(os.path.join(rootdir, m, "describe.csv"))
            both_x = int(((d.said_minute_number == d.true_minute_number).fillna(False).astype(bool) & d.hour_desc_ok.astype(bool)).sum())
            c = pd.read_csv(os.path.join(rootdir, m, "conversion_test.csv"))
            to = c[c.condition == "text_only"]
            labs = [ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute))[0] for r in to.itertuples()]
            leaks = int(sum(1 for l in labs if l == "leak"))
            rows.append({"decoding": tag, "model": m, "both_exact_clocks": both_x, "n_describe": len(d), "text_only_trials": len(to),
                         "text_only_leaks": leaks, "text_only_minute_acc": round(float(to.minute_correct.astype(bool).mean()), 3)})
    return pd.DataFrame(rows)


def e3e_block():
    p("\n--- E3e description transplant at rp 1.0 (HEADLINE; CONFIRMATORY for Qwen-3B primary = verbatim; everything else descriptive)")
    for m in ALL4:
        f = os.path.join(MECH, m, "e3e_summary.csv")
        if not os.path.exists(f):
            continue
        s = pd.read_csv(f)
        s["status"] = np.where(s.is_primary, "PRIMARY", "secondary/expl.")
        p(f"\n{m}" + ("   <- confirmatory claim = verbatim row" if m == QW[0] else "   (descriptive)"))
        p(s[["variant", "status", "n_clocks", "wrong_with_image", "wrong_transplant", "paired_diff", "diff_lo", "diff_hi", "sign_flip_p", "PREREG_verdict_if_primary"]].round(3).to_string(index=False))
    s = pd.read_csv(os.path.join(MECH, QW[0], "e3e_summary.csv"))
    r = s[s.is_primary].iloc[0]
    lang = (r.wrong_transplant >= 0.40) and (r.diff_lo >= -0.15) and (r.diff_hi <= 0.15)
    img = (r.wrong_transplant <= 0.25) and (r.diff_lo > 0 or r.diff_hi < 0)
    p(f"\nQwen-3B verbatim @rp1.0, pre-registered rule: n_clocks={int(r.n_clocks)}; wrong WITHOUT image {r.wrong_transplant:.1%} (>=40%: {r.wrong_transplant >= 0.40}); "
      f"wrong WITH image {r.wrong_with_image:.1%}; paired diff {r.paired_diff:+.1%} CI [{r.diff_lo:+.1%}, {r.diff_hi:+.1%}] (inside +/-15 pts: {(r.diff_lo >= -0.15) and (r.diff_hi <= 0.15)})"
      f"  ->  LANGUAGE-SIDE {'SUPPORTED' if lang else 'not supported'}; IMAGE-INTERFERENCE {'supported' if img else 'not supported'}"
      f"{'' if (lang or img) else '  => MIXED / neither rule met'}")
    o = os.path.join(MECH, QW[0], "e3e_summary_rp105.csv")
    if os.path.exists(o):
        q = pd.read_csv(o)
        q = q[q.is_primary].iloc[0]
        p(f"SENSITIVITY (old D4, rp1.05): n_clocks={int(q.n_clocks)}, wrong without image {q.wrong_transplant:.1%}, with image {q.wrong_with_image:.1%}, paired diff {q.paired_diff:+.1%} CI [{q.diff_lo:+.1%}, {q.diff_hi:+.1%}] -> {q.PREREG_verdict_if_primary}")


def e3f_block(suffix=""):
    p(f"\n--- E3f H=12 at {'rp 1.05 (sensitivity)' if suffix else 'rp 1.0 (HEADLINE)'}  (rule: word form within 10 pts of the H!=12 accuracy while the numeral keeps the gap -> numeral/tokenisation; gap for all forms -> semantic 12=00)")
    for m in QW:
        f = os.path.join(MECH, m, f"e3f_summary{suffix}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f); d["cond"] = d["cond(formN/formH)"]
        rows = []
        for c, g in d.groupby("cond", sort=False):
            g = g.set_index("H_group"); nn = g.loc[["H<10", "H=10,11"], "n"]
            not12 = float((g.loc[["H<10", "H=10,11"], "minute_acc"] * nn).sum() / nn.sum()); h12 = float(g.loc["H=12", "minute_acc"])
            rows.append({"cond(N/H)": c, "acc_H!=12": not12, "acc_H=12": h12, "gap_pts": 100 * (not12 - h12), "H12_minute_00": float(g.loc["H=12", "minute_00"]),
                         "H12_leak": float(g.loc["H=12", "leak"]), "H12_hour_acc": float(g.loc["H=12", "hour_acc"])})
        t = pd.DataFrame(rows); p(f"\n{m}"); p(t.round(3).to_string(index=False))
        b = t[t["cond(N/H)"] == "num/num"].iloc[0]; w = t[t["cond(N/H)"] == "num/word"]
        if len(w):
            w = w.iloc[0]
            p(f"  rule: word-H gap {w['gap_pts']:.1f} pts (<=10: {w['gap_pts'] <= 10}); numeral gap {b['gap_pts']:.1f} pts -> "
              + ("numeral/tokenisation form SUPPORTED" if (w['gap_pts'] <= 10 and b['gap_pts'] > 10) else "semantic 12=00 (gap remains for all forms)" if w['gap_pts'] > 10 else "no gap at baseline"))


def pilot_block():
    p("\n--- Text-light pilot (E2 floor rule: matching_clock minute accuracy >= 0.15), rp 1.0 vs rp 1.05")
    rows = []
    for m in QW + ["internvl3-2b", "internvl3-8b"]:
        for tag, f in (("rp1.0", "text_light_pilot_summary.csv"), ("rp1.05", "text_light_pilot_summary_rp105.csv")):
            q = os.path.join(MECH, m, f)
            if os.path.exists(q):
                d = pd.read_csv(q); d = d[d.cond == "matching_clock"]
                rows.append({"model": m, "decoding": tag if m in QW else "(no penalty)", "minute_acc": float(d.minute_acc.iloc[0]), "hour_acc": float(d.hour_acc.iloc[0]),
                             "clears_0.15_floor": bool(d.minute_acc.iloc[0] >= 0.15)})
    p(pd.DataFrame(rows).round(3).to_string(index=False))


def d3ivl_block():
    d = os.path.join(MECH, "internvl3-2b")
    p("\n" + "=" * 100 + "\nPART 3  D3-IVL  InternVL3-2B mechanism (E3a-d)\n" + "=" * 100)
    p(open(os.path.join("match_D3IVL_results", "STATUS.txt")).read())
    gate_table(os.path.join("match_D3IVL_results", "results", "mech_e3", "internvl3-2b", "verify.json"), "internvl3-2b (D3-IVL run)")
    # E3d  (CONFIRMATORY)
    v = json.load(open(os.path.join(d, "e3d_prereg_verdict.json")))
    s = pd.read_csv(os.path.join(d, "e3d_summary.csv"))
    H, R = s[s.cell == "H_all"].iloc[0], s[s.cell == "random_all"].iloc[0]
    p("\n--- E3d leak blocking (CONFIRMATORY test). Rule: leaks fall >= 30% AND accuracy rises >= 4 pts, both CIs exclude 0, AND the random-span control changes accuracy by < 2 pts.")
    p(f"intact: leaks {v['leak_intact']} / 432, accuracy {s[s.cell == 'intact'].acc.iloc[0]:.3f}")
    p(f"block H-span, all layers: leaks {v['leak_blocked']} ({100 * v['leak_rel_drop']:.1f}% fewer), leak change per trial {H.leak_diff_per_trial:+.3f} CI [{H.leak_lo:+.3f}, {H.leak_hi:+.3f}]; "
      f"accuracy {H.acc:.3f} ({v['acc_diff_points']:+.1f} pts, CI [{v['acc_ci_points'][0]:+.1f}, {v['acc_ci_points'][1]:+.1f}])")
    p(f"random-span control: accuracy {R.acc_diff * 100:+.1f} pts (CI [{R.acc_lo * 100:+.1f}, {R.acc_hi * 100:+.1f}]), leaks {int(R.leak_n)} -> the pre-registered control clause needs < 2 pts")
    p(f"destruction control (block N-span): accuracy {s[s.cell == 'N_all'].acc.iloc[0]:.3f} (must collapse)")
    p(f">>> PRE-REGISTERED VERDICT: M_bind_supported = {v['M_bind_supported']};  weak_conversion_account = {v['weak_conversion_account']}.")
    thr = (v["leak_rel_drop"] >= 0.30 and v["acc_diff_points"] >= 4 and v["leak_ci_excludes0"] and v["acc_ci_excludes0"])
    p(f"    leak-drop >=30%: {v['leak_rel_drop'] >= 0.30}; accuracy >= +4 pts: {v['acc_diff_points'] >= 4}; both CIs exclude 0: {v['leak_ci_excludes0'] and v['acc_ci_excludes0']}; "
      f"random control < 2 pts: {v['random_span_acc_diff_points'] < 2}   (primary-effect criteria all met: {thr})")
    p("    Layer windows (exploratory description, 4 layers wide):")
    w = s[s.cell.str.startswith("H_win")][["cell", "acc", "leak_n", "acc_diff", "leak_diff_per_trial"]]
    p(w.round(3).to_string(index=False))
    p("    Cumulative-to-end blocks: " + "; ".join(f"{r.cell}: leaks {int(r.leak_n)}, acc {r.acc:.3f}" for r in s[s.cell.str.startswith("H_cum")].itertuples()))
    # E3a  (exploratory)
    a = pd.read_csv(os.path.join(d, "e3a_summary.csv"))
    p("\n--- E3a logit lens (EXPLORATORY, no decision rule): mean [log p(gold minute) - log p(leak minute)] by layer")
    t = a[a.group.isin(["right", "wrong_leak", "wrong_other", "H12", "H_not12"])].pivot(index="layer", columns="group", values="mean_gold_minus_leak").round(2)
    p(t.iloc[::3].to_string())
    for g in ("right", "wrong_leak", "wrong_other", "H12", "H_not12"):
        x = a[a.group == g].sort_values("layer")
        p(f"  {g:<12} n={int(x.n.iloc[0])}  final layer: {x.mean_gold_minus_leak.iloc[-1]:+.2f}")
    # E3b
    b = pd.read_csv(os.path.join(d, "e3b_prereg_rule.csv"))
    p("\n--- E3b swap patching. Rule: 'minute value is formed from the N-token' supported if, for some layer window, patching the N-span moves the answer toward the source minute with a CI excluding 0 AND exceeding every control by >= 1 nat.")
    for grp in ("N", "H", "between", "last"):
        g = b[b.group == grp]
        p(f"  span {grp:<8}: supported at {int(g.PREREG_supported.sum())}/{len(g)} tested layers; max shift {g.mean_shift.max():+.2f} nats at layer {int(g.loc[g.mean_shift.idxmax(), 'layer'])}")
    gn = b[b.group == "N"]
    p(f"  N-span: shift {gn.mean_shift.min():+.2f}..{gn.mean_shift.max():+.2f} nats; best control {gn.best_control_abs_shift_same_layer.max():.2f}; supported layers {sorted(gn[gn.PREREG_supported].layer.astype(int).tolist())}")
    p(f"  >>> VERDICT (pre-registered, N-span): {'SUPPORTED' if gn.PREREG_supported.any() else 'not supported'}   (note: it is a statement about the N-token, NOT about the H->minute leak)")
    gh = b[b.group == "H"]
    p(f"  H-span (descriptive): shift {gh.mean_shift.min():+.2f}..{gh.mean_shift.max():+.2f} nats, the sign is toward the TARGET's own minute for layers 6-21 (negative = away from the source minute)")
    # E3c
    c = pd.read_csv(os.path.join(d, "e3c_prereg_rule.csv"))
    p("\n--- E3c restoration. Rule: restoration >= 50% of wrong runs at a single layer-position cell AND >= 25 pts above the best control (else exploratory).")
    top = c.sort_values("restored_rate_among_wrong", ascending=False).head(5)
    p(top[["layer", "group", "n", "restored_rate_among_wrong", "best_control_restored", "PREREG_supported"]].round(3).to_string(index=False))
    p(f"  >>> VERDICT: supported cells: {int(c.PREREG_supported.sum())}/{len(c)}  -> {'SUPPORTED' if c.PREREG_supported.any() else 'NOT supported (best cell below the 50% bar; reported as exploratory)'}")
    o = os.path.join(d, "e3c_summary_own_hour_eq_gold.csv")
    if os.path.exists(o):
        cc = pd.read_csv(o); cc = cc[(cc.kind == "right2wrong") & (cc.variant == "source")].sort_values("restored_rate_among_wrong", ascending=False).head(3)
        p("  robustness, restricted to trials where the model's own hour == gold hour:"); p(cc[["layer", "group", "n", "restored_rate_among_wrong"]].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ stage: report
def stage_report():
    p("=" * 100, "\nPART 1  D2'' GATE (Qwen, --rep_penalty 1.0)\n" + "=" * 100)
    p(open(os.path.join("match_D2pp_results", "STATUS.txt")).read())
    for m in QW:
        gate_table(os.path.join("match_D2pp_results", "results", "mech_e3", m, "verify.json"), m)
    p("\nCompared with D2' (rep penalty 1.05 in the free-generation reference):")
    for m in QW:
        a = json.load(open(os.path.join("match_D2_results", "results", "mech_e3", m, "verify.json")))["checks"]
        b = json.load(open(os.path.join("match_D2pp_results", "results", "mech_e3", m, "verify.json")))["checks"]
        p(f"  {m}: G2-own {a['G2_own_prefix_teacher_forced_vs_free_generation']['agreement']} -> {b['G2_own_prefix_teacher_forced_vs_free_generation']['agreement']};"
          f"  G8b {a['G8b_nocache_loop_vs_adapter_generate_agree']['agreement']} -> {b['G8b_nocache_loop_vs_adapter_generate_agree']['agreement']}")

    p("\n" + "=" * 100, "\nPART 2  NOTEBOOK Q (Qwen regenerated at repetition_penalty 1.0)\n" + "=" * 100)
    p(open(os.path.join("match_Q_results", "STATUS.txt")).read())
    p("merge log:", "; ".join(json.load(open(os.path.join("results", "_Q_merge_log.json")))[-2:]))
    t = sens_table()
    p("\n--- SENSITIVITY TABLE: rp 1.0 (headline) vs rp 1.05 (old).  delta in absolute units (rates = proportion; McNemar rows = p-values). Flags: '>3 pts' / 'SIGNIFICANCE FLIPS'")
    show = t.copy()
    for c in ("rp1.0", "rp1.05", "delta"):
        show[c] = show[c].astype(float).round(3)
    p(show[["model", "metric", "rp1.0", "n_1.0", "rp1.05", "n_1.05", "delta", "flag"]].to_string(index=False))
    fl = t[t.flag != ""]
    p(f"\nFLAGGED ({len(fl)}):")
    p(fl[["model", "metric", "rp1.0", "rp1.05", "delta", "flag"]].astype({"rp1.0": float, "rp1.05": float}).round(3).to_string(index=False) if len(fl) else "none")
    p("\n--- re-derived pre-registration counts")
    p(prereg_counts().to_string(index=False))
    e3e_block()
    e3f_block("")
    e3f_block("_rp105")
    pilot_block()
    d3ivl_block()
    open(os.path.join(MECH, "Q_D3IVL_report.txt"), "w", encoding="utf-8").write("\n".join(OUT))
    print("\nwrote results/mech_e3/Q_D3IVL_report.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "merge", "aggregate", "report"])
    a = ap.parse_args()
    if a.stage in ("all", "merge"):
        stage_merge()
    if a.stage in ("all", "aggregate"):
        stage_aggregate()
    if a.stage in ("all", "report"):
        stage_report()
