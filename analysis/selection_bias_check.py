"""selection_bias_check.py -- all four models (Qwen2.5-VL-3B/7B, InternVL3-2B/8B).

Question 1 (selection bias): the describe_check metric "final answer WRONG given both hands described
correctly" is computed only on the clocks where both hands were described right. Is a model's rate
explained by WHICH minute positions made it into that selected set?
Question 2 (per position): per-position minute accuracy text_only vs matching_clock vs blank with the
Holm-corrected paired McNemar p-values from conversion_test.py.
Question 3 (truncation sensitivity): some models (InternVL3-8B) start a long explanation and are cut
off by max_new_tokens=32 before giving a number; those trials count as WRONG in minute_correct.
Section C repeats the overall McNemar tests after dropping every trial key (position, hour, lead-in)
where ANY of the four conditions failed to parse, and on lead-ins 0 and 2 only.

Reads (CPU only, no model); each file is looked up as <root>/<model>/<file>, first hit wins:
  roots = results, results_v2/results, conv_results/results, qwen_prior/results
  describe.csv, text_only_v2.csv, conversion_test.csv ; results/conversion_test/paired_tests.csv
  (run `python conversion_test.py --aggregate` first).
Definitions: position = minute-hand number 1..12 (12 = minute 0). selected = minute_desc_ok & hour_desc_ok.
text-only = task b, phrasing b2, lead_in_id < 3 (n=36 per position, same as compare_families.py).

USAGE (repo root):  python analysis/selection_bias_check.py [--models a,b,...]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

MODELS = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl3-2b", "internvl3-8b"]
ROOTS = ["results", "results_v2/results", "conv_results/results", "qwen_prior/results"]
POS = range(1, 13)


def find(model, fname):
    for r in ROOTS:
        p = os.path.join(r, model, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{model}/{fname} not found under {ROOTS}")


def load(m):
    d = pd.read_csv(find(m, "describe.csv"))
    for k in ("minute_desc_ok", "hour_desc_ok", "answer_ok"):
        d[k] = d[k].astype(bool)
    t = pd.read_csv(find(m, "text_only_v2.csv"))
    t = t[(t.task == "b") & (t.phrasing == "b2") & (t.lead_in_id < 3)].copy()
    t["ok"] = t.minute_correct.astype(bool)
    c = pd.read_csv(find(m, "conversion_test.csv"))
    c["ok"] = c.minute_correct.astype(bool)
    c["parse_ok"] = c.parse_ok.astype(bool)
    return d, t, c


def selection_table(m):
    d, t, c = load(m)
    d["pos"] = d.true_minute_number
    sel = d[d.minute_desc_ok & d.hour_desc_ok]
    out = pd.DataFrame({
        "all480": d.groupby("pos").size().reindex(POS, fill_value=0),
        "both_ok": sel.groupby("pos").size().reindex(POS, fill_value=0),
        "wrong|both": sel.groupby("pos").apply(lambda g: (~g.answer_ok).sum()).reindex(POS, fill_value=0),
    })
    out["%incl"] = (out.both_ok / out.all480 * 100).round(0)
    out["share_sel%"] = (out.both_ok / out.both_ok.sum() * 100).round(1)
    out["share_all%"] = (out.all480 / out.all480.sum() * 100).round(1)
    out["wrong_rate%"] = (out["wrong|both"] / out.both_ok.replace(0, np.nan) * 100).round(0)
    out["textonly_b2_acc%"] = (t.groupby("position").ok.mean() * 100).round(0)
    out["conv_match_acc%"] = (c[c.condition == "matching_clock"].groupby("position").ok.mean() * 100).round(0)
    out["conv_blank_acc%"] = (c[c.condition == "blank"].groupby("position").ok.mean() * 100).round(0)
    print(f"\n===== {m}: both-hands-described-correct = {len(sel)} of {len(d)}; wrong = {(~sel.answer_ok).sum()} "
          f"({(~sel.answer_ok).mean():.1%}) =====")
    print(out.to_string())
    w, wa = out.both_ok / out.both_ok.sum(), out.all480 / out.all480.sum()
    for nm, col in (("text-only b2 acc", "textonly_b2_acc%"), ("conv matching acc", "conv_match_acc%")):
        print(f"  {nm}: mean over ALL-480 position mix {np.nansum(wa * out[col]):.1f}%   "
              f"over the SELECTED mix {np.nansum(w * out[col]):.1f}%")
    return out


def reweight(rates, mix_counts):
    """model A's per-position wrong rate averaged with model B's selected-position mix (positions where A has cases)."""
    w = mix_counts / mix_counts.sum()
    m = rates.notna() & (w > 0)
    return float(np.sum(rates[m] * w[m]) / w[m].sum())


def per_position_mcnemar(m):
    p = pd.read_csv("results/conversion_test/paired_tests.csv")
    p = p[p.model == m]
    _, t, c = load(m)
    rows = []
    for pos in POS:
        r = {"pos": pos, "text_only%": round(t[t.position == pos].ok.mean() * 100),
             "matching%": round(c[(c.condition == "matching_clock") & (c.position == pos)].ok.mean() * 100),
             "blank%": round(c[(c.condition == "blank") & (c.position == pos)].ok.mean() * 100)}
        for cmp in ("matching_clock", "blank"):
            q = p[(p.comparison == f"text_only_vs_{cmp}") & (p.position.astype(str) == str(pos))].iloc[0]
            r[f"p_holm_{cmp}"] = round(q.p_holm, 4)
            r[f"onlyT/onlyI_{cmp}"] = f"{q.only_text_correct}/{q.only_image_correct}"
        rows.append(r)
    print(f"\n===== {m}: per-position minute accuracy (n=36 each) + paired McNemar (Holm over 12) =====")
    print(pd.DataFrame(rows).to_string(index=False))


def truncation_sensitivity(m):
    from conversion_test import paired_tests
    _, _, c = load(m)
    k = ["position", "hour", "lead_in_id"]
    fails = c.groupby("condition").parse_ok.apply(lambda s: int((~s).sum()))
    print(f"\n===== {m}: unparsed trials per condition (of 432 each): "
          + ", ".join(f"{a}={b}" for a, b in fails.items()) + " =====")
    bad = c[~c.parse_ok][k].drop_duplicates().assign(_b=1)
    kept = c.merge(bad, on=k, how="left")
    kept = kept[kept._b.isna()].drop(columns="_b")
    for name, df in ((f"all trials ({len(c) // 4} per condition)", c),
                     (f"drop keys with any unparsed condition ({len(kept) // 4} per condition)", kept),
                     (f"lead-ins 0 and 2 only ({len(c[c.lead_in_id.isin([0, 2])]) // 4} per condition)", c[c.lead_in_id.isin([0, 2])])):
        tt = paired_tests(df)
        tt = tt[tt.position == "all"]
        acc = {cond: round(df[df.condition == cond].ok.mean() * 100, 1) for cond in
               ("text_only", "matching_clock", "blank", "mismatched_clock")}
        print(f"  [{name}] minute acc %: {acc}")
        print(tt[["comparison", "n", "only_text_correct", "only_image_correct", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    models = [x.strip() for x in ap.parse_args().models.split(",")]

    print("#################### A. SELECTION BIAS (describe_check) ####################")
    tabs = {m: selection_table(m) for m in models}
    print("\n===== wrong-given-both rate of ROW model re-weighted to COLUMN model's selected-position mix (%) =====")
    mat = pd.DataFrame({b: {a: reweight(tabs[a]["wrong_rate%"], tabs[b].both_ok) for a in models} for b in models})
    mat.insert(0, "own_mix", {a: round(100 * tabs[a]["wrong|both"].sum() / tabs[a].both_ok.sum(), 1) for a in models})
    print(mat.round(1).to_string())
    print("\n===== same-position comparison: wrong/selected for each model, positions with >=3 selected cases in ALL models =====")
    common = [p for p in POS if all(tabs[m].both_ok[p] >= 3 for m in models)]
    print("common positions:", common)
    print(pd.DataFrame({m: tabs[m].loc[common, "wrong|both"].astype(str) + "/" + tabs[m].loc[common, "both_ok"].astype(str)
                        for m in models}).to_string())
    for m in models:
        k, n = tabs[m].loc[common, "wrong|both"].sum(), tabs[m].loc[common, "both_ok"].sum()
        print(f"  {m}: {k}/{n} = {k / n:.1%} on the common positions")

    print("\n#################### B. PER-POSITION text_only vs matching vs blank (McNemar, Holm) ####################")
    for m in models:
        per_position_mcnemar(m)

    print("\n#################### C. TRUNCATION / PARSE-FAILURE SENSITIVITY (conversion test) ####################")
    for m in models:
        truncation_sensitivity(m)

    print("\n#################### D. WHAT ARE THE 'WRONG GIVEN BOTH DESCRIBED RIGHT' CASES? ####################")
    print("minute_desc_ok accepts a described number within +/-1 of the truth. 'off_by_one_desc' = the model said a minute")
    print("number != the true one (accepted only by the tolerance); 'follows_own_desc' = its final minute equals (said number*5) mod 60.")
    rows = []
    for m in models:
        d, _, _ = load(m)
        s = d[d.minute_desc_ok & d.hour_desc_ok & ~d.answer_ok]
        off = s.said_minute_number != s.true_minute_number
        fol = (s.pred_minute == (s.said_minute_number * 5) % 60)
        rows.append({"model": m, "wrong_given_both": len(s), "off_by_one_desc": int(off.sum()),
                     "exact_desc_but_wrong": int((~off).sum()), "follows_own_desc": int(fol.sum()),
                     "exact_desc_wrong_AND_not_following": int(((~off) & (~fol)).sum()),
                     "hour_wrong": int((s.pred_hour != s.true_hour).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
