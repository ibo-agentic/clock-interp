"""error_taxonomy.py -- E0 (CPU only): pre-registered error labels for the conversion test + the H=12 anomaly.

Reads results/<model>/conversion_test.csv (text_only, matching_clock, blank, mismatched_clock; 432 trials each).
Every trial gets exactly ONE label, assigned in this fixed priority order (the order is part of the pre-registration):
  correct        parsed minute == truth minute
  unparsed       no HH:MM could be parsed
  leak           minute == (H mod 12)*5   (the HOUR number was converted into the minute slot)
  follows_image  minute == image_minute (mismatched_clock only; the clock in the picture showed that minute)
  adjacent       minute == truth +/- 5    (neighbouring number)
  default_0_30   minute in {0, 30}
  other
Ambiguity that must be kept in mind: for H=12 the leak minute is 0, i.e. identical to the "00" default, and for
N=H the leak minute equals the truth (then the trial is labelled correct, never leak).  Section B therefore reports
H=12 separately and also shows the raw minute histogram.

USAGE (repo root):  python analysis/error_taxonomy.py [--models a,b,...] [--out_dir results/mechanism_E0]
Also tries to tokenize the numbers ("12", " 12", ...) with each model's tokenizer (Section C; skipped if offline).
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
HF_IDS = {"qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct", "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
          "internvl3-2b": "OpenGVLab/InternVL3-2B-hf", "internvl3-8b": "OpenGVLab/InternVL3-8B-hf"}
CONDS = ["text_only", "matching_clock", "blank", "mismatched_clock"]
LABELS = ["correct", "leak", "follows_image", "adjacent", "default_0_30", "other", "unparsed"]


def parsed_minute(s):
    try:
        return int(str(s).split(":")[1][:2])
    except Exception:
        return np.nan


def label_trial(r):
    m = r.pred_min
    if not bool(r.parse_ok) or np.isnan(m):
        return "unparsed"
    truth, leak = int(r.truth_minute), (int(r.hour) % 12) * 5
    if int(m) == truth:
        return "correct"
    if int(m) == leak:
        return "leak"
    if r.condition == "mismatched_clock" and not pd.isna(r.image_minute) and int(m) == int(r.image_minute):
        return "follows_image"
    if int(m) in ((truth + 5) % 60, (truth - 5) % 60):
        return "adjacent"
    if int(m) in (0, 30):
        return "default_0_30"
    return "other"


def load(model):
    c = pd.read_csv(os.path.join("results", model, "conversion_test.csv"))
    c["pred_min"] = c.parsed_answer.map(parsed_minute)
    c["label"] = c.apply(label_trial, axis=1)
    c["ok"] = c.label == "correct"
    return c


def taxonomy(c, model):
    t = pd.crosstab(c.condition, c.label).reindex(index=CONDS, columns=LABELS, fill_value=0)
    t.insert(0, "n", t.sum(axis=1))
    print(f"\n===== {model}: error labels per condition (counts of 432) =====")
    print(t.to_string())
    return t.assign(model=model).reset_index()


def hour12(c, model):
    print(f"\n===== {model}: B. H=12 anomaly (minute accuracy, text_only and matching_clock) =====")
    rows = []
    for cond in ("text_only", "matching_clock", "blank"):
        d = c[c.condition == cond]
        for name, g in (("H=12", d[d.hour == 12]), ("H!=12", d[d.hour != 12])):
            vc = g.label.value_counts()
            rows.append({"cond": cond, "hour": name, "n": len(g), "acc%": round(g.ok.mean() * 100, 1),
                         "hour_acc%": round(g.hour_correct.mean() * 100, 1),
                         **{k: int(vc.get(k, 0)) for k in LABELS if k != "correct"}})
    print(pd.DataFrame(rows).to_string(index=False))
    d = c[(c.condition == "text_only") & (c.hour == 12)]
    print("  text_only, H=12: predicted minute histogram:", d.pred_min.value_counts().sort_index().to_dict())
    print("  text_only, H=12: accuracy by minute number N:", (d.groupby("position").ok.mean() * 100).round(0).astype(int).to_dict())
    d2 = c[(c.condition == "text_only") & (c.hour != 12)]
    print("  text_only, H!=12: accuracy by H:", (d2.groupby("hour").ok.mean() * 100).round(0).astype(int).to_dict())
    return rows


def leak_vs_default_at_h12(c, model):
    """At H=12 'leak' and '00 default' coincide; at H!=12 they differ. Report the 0-answer rate as a function of N."""
    d = c[c.condition == "text_only"]
    z = d.assign(said_00=(d.pred_min == 0))
    print(f"  {model}: share of text_only answers that are minute 00: H=12 {z[z.hour == 12].said_00.mean():.0%}, "
          f"H!=12 {z[(z.hour != 12) & (z.position != 12)].said_00.mean():.0%} (N=12 excluded since 00 is correct there)")


def tokenization():
    print("\n===== C. How each tokenizer splits the numbers used in the prompt =====")
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # noqa: BLE001
        print("transformers not importable:", e)
        return
    for m in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(HF_IDS[m])
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: tokenizer not available ({type(e).__name__}); run once with network or HF cache")
            continue
        sp = {n: len(tok(f"hand is pointing at the number {n}, and", add_special_tokens=False).input_ids)
              - len(tok("hand is pointing at the number , and", add_special_tokens=False).input_ids) for n in range(1, 13)}
        pieces = {n: tok.convert_ids_to_tokens(tok(f" {n}", add_special_tokens=False).input_ids) for n in (1, 9, 10, 11, 12)}
        print(f"  {m}: tokens used per number 1..12 in-prompt = {sp}")
        print(f"      pieces: {pieces}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out_dir", default=os.path.join("results", "mechanism_E0"))
    a = ap.parse_args()
    models = [x.strip() for x in a.models.split(",")]
    os.makedirs(a.out_dir, exist_ok=True)
    print("#################### A. ERROR TAXONOMY ####################")
    tabs, rows, trial = [], [], []
    data = {m: load(m) for m in models}
    for m in models:
        tabs.append(taxonomy(data[m], m))
        trial.append(data[m][["model", "condition", "position", "hour", "lead_in_id", "pred_min", "label"]])
    pd.concat(tabs).to_csv(os.path.join(a.out_dir, "taxonomy_counts.csv"), index=False)
    pd.concat(trial).to_csv(os.path.join(a.out_dir, "trial_labels.csv"), index=False)
    print("\n#################### B. H=12 ANOMALY ####################")
    for m in models:
        rows += [{"model": m, **r} for r in hour12(data[m], m)]
    print()
    for m in models:
        leak_vs_default_at_h12(data[m], m)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "h12_anomaly.csv"), index=False)
    tokenization()
    print(f"\nwrote {a.out_dir}/taxonomy_counts.csv, trial_labels.csv, h12_anomaly.csv")
