"""Reproducible analysis of the D2' gate re-check and the D4 hook-free run.
usage (from the repo root):  python analysis/analyze_D2prime_D4.py
Reads  match_D2_results/ (gate-only re-check)  and  match_D4_results/ (E3e, E3f, text-light pilot);
merges D4 results into results/mech_e3/<model>/ WITHOUT overwriting; writes results/mech_e3/D2prime_D4_report.txt.
Thresholds are the pre-registered ones (PREREG_MECHANISM.md Section 4); nothing here changes them."""
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2, D4 = os.path.join(ROOT, "match_D2_results"), os.path.join(ROOT, "match_D4_results")
RES = os.path.join(ROOT, "results", "mech_e3")
MODELS = ["qwen2.5-vl-3b", "internvl3-2b", "qwen2.5-vl-7b"]
OUT = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def hour_of(t):
    m = re.search(r"(\d{1,2}):(\d{2})", str(t))
    return None if not m else (int(m.group(1)) % 12 or 12)


# ------------------------------------------------------------------------------------------------ merge (no clobber)
def merge():
    n_new, n_skip = 0, 0
    for m in MODELS:
        src = os.path.join(D4, "results", "mech_e3", m)
        dst = os.path.join(RES, m)
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(src, "*"))):
            t = os.path.join(dst, os.path.basename(f))
            if os.path.exists(t):
                n_skip += 1
            else:
                shutil.copy2(f, t); n_new += 1
        for f, new in (("verify.json", "verify_D2prime.json"), ("verify_g2_trials.csv", "verify_g2_trials_D2prime.csv")):
            s, t = os.path.join(D2, "results", "mech_e3", m, f), os.path.join(dst, new)
            if os.path.exists(s) and not os.path.exists(t):
                shutil.copy2(s, t); n_new += 1
    p(f"[merge] copied {n_new} new files into results/mech_e3/<model>/, skipped {n_skip} that already existed (no overwrite)")


# ------------------------------------------------------------------------------------------------ PART 1: gate
def part1():
    p("=" * 100, "\nPART 1  D2' GATE RE-CHECK\n" + "=" * 100)
    p(open(os.path.join(D2, "STATUS.txt")).read())
    summ = {}
    for m in MODELS:
        f = os.path.join(D2, "results", "mech_e3", m, "verify.json")
        v = json.load(open(f))
        summ[m] = v["pass"]
        p(f"\n{m}: GATE {'PASSED' if v['pass'] else 'FAILED'}   attention impl: {v.get('attn_impl')}")
        for k, c in v["checks"].items():
            tag = ("PASS" if c["pass"] else "FAIL") if c.get("gating", True) else ("info:pass" if c["pass"] else "info:FAIL")
            info = {a: b for a, b in c.items() if a not in ("pass", "gating", "note")}
            p(f"  [{tag:<9}] {k}  {str(info)[:400]}")
    p("\n--- Bug A: where do cached `generate` and the no-cache greedy loop differ? (verify_g2_trials.csv, n=100 each)")
    rows = []
    for m in MODELS:
        d = pd.read_csv(os.path.join(D2, "results", "mech_e3", m, "verify_g2_trials.csv"))
        H = d.H.map(lambda h: h % 12 or 12)
        hc, hn = d.free_text.map(hour_of), d.nocache_text.map(hour_of)
        diff = d[~d.nocache_eq_free.astype(bool)]
        # first character of the answer: do the two decoders disagree already in the HOUR?
        hour_div = int((hc[diff.index] != hn[diff.index]).sum())
        rows.append({"model": m, "cached==nocache": round(float(d.nocache_eq_free.astype(bool).mean()), 2),
                     "n_differ": len(diff), "differ_already_in_hour": hour_div,
                     "acc_cached": round(float(d.free_ok.mean()), 2), "acc_nocache": round(float(d.nocache_ok.astype(bool).mean()), 2),
                     "hour_correct_cached": round(float((hc == H).mean()), 2), "hour_correct_nocache": round(float((hn == H).mean()), 2)})
    p(pd.DataFrame(rows).to_string(index=False))
    d = pd.read_csv(os.path.join(D2, "results", "mech_e3", "qwen2.5-vl-3b", "verify_g2_trials.csv"))
    dis = d[d.own_tf_ok.notna() & (d.free_ok.astype(bool) != d.own_tf_ok.astype(bool))]
    p("\nQwen-3B G2-own disagreements (cached free generation vs own-prefix teacher forcing):")
    p(dis[["lead", "N", "H", "free_text", "nocache_text", "own_prefix", "prefix_class", "free_on_grid", "own_tf_margin"]].to_string(index=False))
    p("\nGenerate-config check (Qwen checkpoints ship repetition_penalty=1.05 in generation_config.json; InternVL3-hf ships none).")
    p("`adapter.generate_answer` only overrides do_sample=False, so cached generate for Qwen is run WITH that penalty; the no-cache argmax loop and\n"
      "the teacher-forced scoring are NOT. See report for what this means.")
    return summ


# ------------------------------------------------------------------------------------------------ PART 2: D4
def part2():
    p("\n" + "=" * 100, "\nPART 2  D4 HOOK-FREE EXPERIMENTS\n" + "=" * 100)
    p(open(os.path.join(D4, "STATUS.txt")).read())
    flagged = [l for l in open(os.path.join(D4, "STATUS.txt")) if "FAILED" in l or "SKIPPED" in l]
    p("steps FAILED/SKIPPED:", flagged or "none")
    merge()

    p("\n--- E3e description transplant (CONFIRMATORY for Qwen-3B primary = verbatim; other models and variants DESCRIPTIVE)")
    for m in MODELS:
        s = pd.read_csv(os.path.join(RES, m, "e3e_summary.csv"))
        s["status"] = np.where(s.is_primary, "PRIMARY", "secondary/exploratory")
        s = s.rename(columns={"wrong_with_image": "wrong_img", "wrong_transplant": "wrong_noimg", "PREREG_verdict_if_primary": "verdict_by_thresholds"})
        p(f"\n{m}" + ("   <- confirmatory claim (verbatim row only)" if m == "qwen2.5-vl-3b" else "   (descriptive: thresholds were written for Qwen-3B)"))
        p(s[["variant", "status", "n_clocks", "wrong_img", "wrong_noimg", "paired_diff", "diff_lo", "diff_hi", "sign_flip_p", "verdict_by_thresholds"]]
          .round(3).to_string(index=False))
    q = pd.read_csv(os.path.join(RES, "qwen2.5-vl-3b", "e3e_summary.csv"))
    r = q[q.is_primary].iloc[0]
    ok_lang = (r.wrong_transplant >= 0.40) and (r.diff_lo >= -0.15) and (r.diff_hi <= 0.15)
    ok_img = (r.wrong_transplant <= 0.25) and (r.diff_lo > 0 or r.diff_hi < 0)
    p(f"\nQwen-3B verbatim, pre-registered rule: wrong without image {r.wrong_transplant:.1%} (>=40%: {r.wrong_transplant >= 0.40}); paired diff "
      f"{r.paired_diff:+.1%} CI [{r.diff_lo:+.1%}, {r.diff_hi:+.1%}] inside +/-15 pts: {(r.diff_lo >= -0.15) and (r.diff_hi <= 0.15)}"
      f"  ->  LANGUAGE-SIDE {'SUPPORTED' if ok_lang else 'not supported'};  image-interference {'supported' if ok_img else 'not supported'}")

    p("\n--- E3f H=12 (rule: 'numeral/tokenisation form' if the word form raises H=12 minute accuracy to within 10 pts of the H!=12 accuracy while the numeral keeps the gap)")
    for m in MODELS:
        d = pd.read_csv(os.path.join(RES, m, "e3f_summary.csv"))
        d["cond"] = d["cond(formN/formH)"]
        rows = []
        for c, g in d.groupby("cond", sort=False):
            g = g.set_index("H_group")
            nn = g.loc[["H<10", "H=10,11"], "n"]
            not12 = float((g.loc[["H<10", "H=10,11"], "minute_acc"] * nn).sum() / nn.sum())
            h12 = float(g.loc["H=12", "minute_acc"])
            rows.append({"cond(N/H)": c, "acc_H!=12": not12, "acc_H=12": h12, "gap_pts": 100 * (not12 - h12),
                         "H12_minute_00": float(g.loc["H=12", "minute_00"]), "H12_leak": float(g.loc["H=12", "leak"]),
                         "H12_hour_acc": float(g.loc["H=12", "hour_acc"]), "H12_hour_copied_from_N": float(g.loc["H=12", "hour_copied_from_N"]),
                         "H12_hour_01": float(g.loc["H=12", "hour_01_at_H12"])})
        t = pd.DataFrame(rows)
        p(f"\n{m}")
        p(t.round(3).to_string(index=False))
        base = t[t["cond(N/H)"] == "num/num"].iloc[0]
        word = t[t["cond(N/H)"] == "num/word"]
        if len(word):
            w = word.iloc[0]
            within10 = w["gap_pts"] <= 10
            numeral_gap = base["gap_pts"] > 10
            p(f"  rule: word-H gap {w['gap_pts']:.1f} pts (<=10: {within10}); numeral gap {base['gap_pts']:.1f} pts (keeps gap: {numeral_gap})  ->  "
              + ("numeral/tokenisation-form explanation SUPPORTED" if (within10 and numeral_gap) else
                 "'semantic 12=00' explanation (gap remains for all forms)" if not within10 else "no gap at baseline: nothing to explain"))

    p("\n--- Text-light pilot (E2 floor rule: intact minute accuracy >= 0.15; blank = no clock image, chance 1/12 = 0.083)")
    rows = []
    for m in MODELS + ["internvl3-8b"]:
        f = os.path.join(RES, m, "text_light_pilot_summary.csv")
        if os.path.exists(f):
            rows.append(pd.read_csv(f))
    t = pd.concat(rows)
    p(t.round(3).to_string(index=False))
    p("clears the 15% floor:", sorted(t[(t.cond == "matching_clock") & (t.iloc[:, -1].astype(bool))].model.tolist()))


if __name__ == "__main__":
    s = part1()
    part2()
    open(os.path.join(RES, "D2prime_D4_report.txt"), "w", encoding="utf-8").write("\n".join(OUT))
    print("\nwrote results/mech_e3/D2prime_D4_report.txt")
