"""compare_families.py -- ONE comparison table across model families
(Qwen2.5-VL-3B/7B, InternVL3-2B/8B) from the per-trial CSVs the experiment
scripts write.  CPU only; reads files, runs no model.

Metrics (rows) x models (columns):
  behavior (eval_behavior.py, data/, n=500):   exact / hour / minute accuracy, parse rate
        -- same definitions as analyze.py: accuracies over PARSED replies
  describe (describe_check.py, data_positions/, n=480):
        PRIMARY = EXACT description (said minute number == true number, hour exact), parsed with the TOLERANT parse (all models):
        minute-exact, both-exact, wrong-given-both-exact.  SECONDARY = the +/-1-minute-number tolerance versions.
        SENSITIVITY = the same rows with the original strict regexes (rows marked "strict").
        Plus the parse rate of the description
  text-only gate (text_only_v2.py): b2 minute accuracy (headline phrasing)
  conversion test (conversion_test.py, n=432 per condition):
        minute accuracy vs the TEXT for text_only / matching_clock / blank /
        mismatched_clock, how often the mismatched answer followed the IMAGE
        instead, and the paired McNemar test (text_only vs each image condition)

Every number is a rate with a 95% Wilson interval and its n.  Each file is
looked up as <root>/<model_short_name>/<file> across --roots (first hit wins),
with two documented legacy fallbacks for the original Qwen-3B files.  The file
used for every cell is printed and saved to four_model_table_sources.json, so
nothing is silently mixed.  A missing file gives "NOT RUN", never a guess.

USAGE:
    python compare_families.py
    python compare_families.py --roots results,results_v2/results,conv_results/results,qwen_prior/results
Outputs (default results/comparison/): four_model_table.md, four_model_table.csv
(formatted, metrics x models), four_model_long.csv (numeric: model, metric,
value, ci_lo, ci_hi, k, n, extra, source), four_model_table_sources.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from text_only_v2 import wilson

MODELS = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl3-2b", "internvl3-8b"]
DEFAULT_ROOTS = ["results", "results_v2/results", "conv_results/results", "qwen_prior/results"]
# original Qwen-3B files that predate the per-model folder layout; used ONLY if the per-model file is absent
LEGACY = {
    ("qwen2.5-vl-3b", "results.csv"): ["results/results.csv"],
    ("qwen2.5-vl-3b", "describe.csv"): ["results/describe_positions.csv"],
}
CONDITIONS = ["text_only", "matching_clock", "blank", "mismatched_clock"]


def find_file(model, fname, roots):
    for r in roots:
        p = os.path.join(r, model, fname)
        if os.path.exists(p):
            return p, False
    for p in LEGACY.get((model, fname), []):
        if os.path.exists(p):
            return p, True
    return None, False


def rate(k, n, extra="", source=None):
    lo, hi = wilson(int(k), int(n)) if n else (float("nan"), float("nan"))
    return {"value": (k / n) if n else float("nan"), "ci_lo": lo, "ci_hi": hi, "k": int(k), "n": int(n),
            "extra": extra, "source": source}


def behavior_metrics(df, src):
    parsed = df[df["parse_success"].astype(bool)]
    n = len(parsed)
    ph, pm = parsed["pred_hour"].astype(int), parsed["pred_minute"].astype(int)
    hour = (ph == parsed["true_hour"]).sum()
    minute = (pm == parsed["true_minute"]).sum()
    exact = ((ph == parsed["true_hour"]) & (pm == parsed["true_minute"])).sum()
    return {
        "behavior_exact": rate(exact, n, source=src), "behavior_hour": rate(hour, n, source=src),
        "behavior_minute": rate(minute, n, source=src),
        "behavior_parse_rate": rate(n, len(df), source=src),
    }


def describe_metrics(df, src):
    """PRIMARY (from 2026-11-09, deviations log fifth/sixth entries) = the TOLERANT parse for ALL models: said_minute_number_tolerant /
    said_hour_number_tolerant (describe_check.MIN_RE_TOL / HOUR_RE_TOL, original regex tried first), recomputed from the saved raw answers.
    Exact description = said minute number == true number AND said hour number == true hour number; the +/-1-minute-number version is secondary.
    The ORIGINAL strict regex columns (minute_desc_ok / hour_desc_ok / said_*_number) are reported beside it as a SENSITIVITY table (keys *_strict).
    'answer_ok' is the full final HH:MM (identical under both parses)."""
    from describe_check import add_tolerant_columns
    if not {"hour_desc_ok_tolerant", "minute_desc_ok_tolerant", "said_minute_number_tolerant"} <= set(df.columns):   # older CSVs / G1 CSVs: from raw answers
        df = add_tolerant_columns(df[[c for c in df.columns if not c.endswith("_tolerant")]])
    n = len(df)
    ao = df["answer_ok"].astype(bool)

    def block(mn, hn, mo, ho, sfx, tag):
        mx = (mn == df["true_minute_number"]).fillna(False).astype(bool)          # exact (NaN -> False)
        mo, ho = mo.astype(bool), ho.astype(bool)
        both, both_x = mo & ho, mx & ho
        parsed = mn.notna() & hn.notna() & df["pred_hour"].notna()
        return {
            f"describe_minute_exact{sfx}": rate(mx.sum(), n, source=src),
            f"describe_hour_ok{sfx}": rate(ho.sum(), n, source=src),
            f"describe_both_exact{sfx}": rate(both_x.sum(), n, source=src),
            f"describe_wrong_given_both_exact{sfx}": rate((both_x & ~ao).sum(), both_x.sum(),
                                                          extra=f"{int(both_x.sum())} of {n} images had both hands described EXACTLY right ({tag})", source=src),
            f"describe_minute_ok{sfx}": rate(mo.sum(), n, source=src),
            f"describe_wrong_given_both_ok{sfx}": rate((both & ~ao).sum(), both.sum(),
                                                       extra=f"{int(both.sum())} of {n} images had both hands described right (+/-1 minute number, {tag})", source=src),
            f"describe_parse_rate{sfx}": rate(parsed.sum(), n, source=src),
        }
    out = block(df["said_minute_number_tolerant"], df["said_hour_number_tolerant"], df["minute_desc_ok_tolerant"], df["hour_desc_ok_tolerant"], "", "tolerant parse")
    out.update(block(df["said_minute_number"], df["said_hour_number"], df["minute_desc_ok"], df["hour_desc_ok"], "_strict", "strict original regexes"))
    return out


def text_only_metrics(df, src):
    d = df[(df["task"] == "b") & (df["phrasing"] == "b2") & (df["lead_in_id"] < 3)]
    return {"textonly_b2_minute": rate(d["minute_correct"].astype(bool).sum(), len(d), source=src)}


def conversion_metrics(df, src):
    from conversion_test import paired_tests
    out = {}
    df = df.copy()
    df["minute_correct"] = df["minute_correct"].astype(bool)
    for cond in CONDITIONS:
        d = df[df.condition == cond]
        out[f"conv_{cond}_minute"] = rate(d["minute_correct"].sum(), len(d), source=src)
    mm = df[df.condition == "mismatched_clock"]
    out["conv_mismatched_follows_image_minute"] = rate(mm["matches_image_minute"].astype(bool).sum(), len(mm), source=src)
    out["conv_mismatched_follows_image_hour"] = rate(mm["matches_image_hour"].astype(bool).sum(), len(mm), source=src)
    out["conv_parse_rate"] = rate(df["parse_ok"].astype(bool).sum(), len(df), source=src)
    t = paired_tests(df)
    t = t[t.position == "all"]
    for cond in ("matching_clock", "blank", "mismatched_clock"):
        r = t[t.comparison == f"text_only_vs_{cond}"].iloc[0]
        out[f"conv_mcnemar_{cond}"] = {
            "value": float(r.p_value), "ci_lo": float("nan"), "ci_hi": float("nan"), "k": int(r.only_image_correct),
            "n": int(r.n), "extra": f"only-text-correct={int(r.only_text_correct)}, only-image-correct={int(r.only_image_correct)}, "
                                    f"both={int(r.both_correct)}, neither={int(r.both_wrong)}", "source": src}
    return out


LOADERS = [  # (file name, loader, label for messages)
    ("results.csv", behavior_metrics, "behavior"),
    ("describe.csv", describe_metrics, "describe"),
    ("text_only_v2.csv", text_only_metrics, "text-only gate"),
    ("conversion_test.csv", conversion_metrics, "conversion"),
]

ROWS = [  # (key, display label, kind)
    ("behavior_exact", "Behavior: exact accuracy (data/, parsed replies)", "pct"),
    ("behavior_hour", "Behavior: hour accuracy", "pct"),
    ("behavior_minute", "Behavior: minute accuracy", "pct"),
    ("behavior_parse_rate", "Behavior: parse rate", "pct"),
    ("describe_minute_exact", "Describe [PRIMARY, tolerant parse]: minute hand described EXACTLY right", "pct"),
    ("describe_hour_ok", "Describe [PRIMARY, tolerant parse]: hour hand described right (exact)", "pct"),
    ("describe_both_exact", "Describe [PRIMARY, tolerant parse]: BOTH hands described exactly right", "pct"),
    ("describe_wrong_given_both_exact", "Describe [PRIMARY, tolerant parse]: final answer WRONG given both hands described EXACTLY right", "pct"),
    ("describe_minute_ok", "Describe [secondary, tolerant parse]: minute hand described right (+/-1 number)", "pct"),
    ("describe_wrong_given_both_ok", "Describe [secondary, tolerant parse]: final answer WRONG given both hands described right (+/-1 number)", "pct"),
    ("describe_parse_rate", "Describe [tolerant parse]: reply fully parsed (parse-fit check)", "pct"),
    ("describe_minute_exact_strict", "Describe [SENSITIVITY, strict original regexes]: minute hand described EXACTLY right", "pct"),
    ("describe_hour_ok_strict", "Describe [SENSITIVITY, strict]: hour hand described right (exact)", "pct"),
    ("describe_both_exact_strict", "Describe [SENSITIVITY, strict]: BOTH hands described exactly right", "pct"),
    ("describe_wrong_given_both_exact_strict", "Describe [SENSITIVITY, strict]: final answer WRONG given both hands described EXACTLY right", "pct"),
    ("describe_minute_ok_strict", "Describe [SENSITIVITY, strict]: minute hand described right (+/-1 number)", "pct"),
    ("describe_wrong_given_both_ok_strict", "Describe [SENSITIVITY, strict]: final answer WRONG given both hands described right (+/-1 number)", "pct"),
    ("describe_parse_rate_strict", "Describe [SENSITIVITY, strict]: reply fully parsed (original regexes)", "pct"),
    ("textonly_b2_minute", "Text-only gate: b2 minute accuracy (no image)", "pct"),
    ("conv_text_only_minute", "Conversion: minute acc., no image (re-run of text-only)", "pct"),
    ("conv_matching_clock_minute", "Conversion: minute acc., matching clock", "pct"),
    ("conv_blank_minute", "Conversion: minute acc., blank image", "pct"),
    ("conv_mismatched_clock_minute", "Conversion: minute acc., mismatched clock (follows TEXT)", "pct"),
    ("conv_mismatched_follows_image_minute", "Conversion: mismatched clock, answer minute follows IMAGE", "pct"),
    ("conv_mismatched_follows_image_hour", "Conversion: mismatched clock, answer hour follows IMAGE", "pct"),
    ("conv_parse_rate", "Conversion: parse rate (all conditions)", "pct"),
    ("conv_mcnemar_matching_clock", "McNemar p, no-image vs matching clock (minute)", "p"),
    ("conv_mcnemar_blank", "McNemar p, no-image vs blank (minute)", "p"),
    ("conv_mcnemar_mismatched_clock", "McNemar p, no-image vs mismatched clock (minute)", "p"),
]


def fmt(cell, kind):
    if cell is None:
        return "NOT RUN"
    if kind == "p":
        return f"p={cell['value']:.2g} ({cell['extra'].split(',')[0]} / {cell['extra'].split(',')[1].strip()})"
    s = f"{cell['value']:.1%} [{cell['ci_lo']:.1%}, {cell['ci_hi']:.1%}] n={cell['n']}"
    return s


def build(models, roots, out_dir):
    results, sources, notes = {}, {}, []
    for m in models:
        results[m] = {}
        for fname, loader, label in LOADERS:
            path, legacy = find_file(m, fname, roots)
            if path is None:
                notes.append(f"{m}: {label} ({fname}) NOT FOUND under any --roots folder")
                continue
            if legacy:
                notes.append(f"{m}: {label} uses LEGACY file '{path}' (predates the per-model folder layout)")
            sources[f"{m}/{fname}"] = path + (" (legacy)" if legacy else "")
            results[m].update(loader(pd.read_csv(path), path))

    os.makedirs(out_dir, exist_ok=True)
    long_rows = []
    for m in models:
        for key, label, kind in ROWS:
            c = results[m].get(key)
            long_rows.append({"model": m, "metric": key, **({} if c is None else c)})
    pd.DataFrame(long_rows).to_csv(os.path.join(out_dir, "four_model_long.csv"), index=False)

    wide = pd.DataFrame({m: [fmt(results[m].get(key), kind) for key, _, kind in ROWS] for m in models},
                        index=[label for _, label, _ in ROWS])
    wide.index.name = "metric"
    wide.to_csv(os.path.join(out_dir, "four_model_table.csv"))

    md = ["| metric | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for (key, label, kind) in ROWS:
        md.append(f"| {label} | " + " | ".join(fmt(results[m].get(key), kind) for m in models) + " |")
    foot = [
        "",
        "Rates are shown as value [95% Wilson CI] n=count. Wilson CIs treat trials as independent; in the "
        "conversion and text-only tests the prompts are crossed (hours x lead-ins), so those CIs are somewhat optimistic.",
        "",
        "Known protocol differences (not hidden): InternVL3's chat template injects NO default system prompt while "
        "Qwen2.5-VL's does; InternVL images are one 448x448 tile = 256 tokens, Qwen's 512x512 clocks use ~324 tokens; "
        "max_new_tokens is 16 (behavior) and 120 (describe) for every model; for text-only / conversion it is 32 for "
        "every model EXCEPT any run whose run_config.json says otherwise (InternVL3-8B was re-run at 64 after 32 cut "
        "off replies; no Qwen or InternVL3-2B reply reached 32 tokens, so those results are unaffected); dtype is "
        "float16 for all four (see each run_config.json).",
        "Describe rows: from 2026-11-09 the PRIMARY parse is the tolerant one for every model (it also reads e.g. 'pointing just past 12' and "
        "'Minute hand: The long hand is pointing at ...'); the original strict regexes are kept as a SENSITIVITY block (rows marked strict). "
        "The two differ only for InternVL3-8B among the original four models (hour parse: 66.9% -> 70.6% hour right). The +/-1-number tolerance is unchanged.",
    ]
    if notes:
        foot += ["", "Source notes:"] + [f"- {n}" for n in notes]
    with open(os.path.join(out_dir, "four_model_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md + foot) + "\n")
    with open(os.path.join(out_dir, "four_model_table_sources.json"), "w", encoding="utf-8") as f:
        json.dump({"sources": sources, "notes": notes, "roots": roots}, f, indent=2)
    print("\n".join(md))
    print("\n".join(foot))
    print(f"\nWrote four_model_table.md/.csv, four_model_long.csv, four_model_table_sources.json to '{out_dir}'.")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=str, default=",".join(MODELS))
    ap.add_argument("--roots", type=str, default=",".join(DEFAULT_ROOTS),
                    help="comma-separated folders searched for <root>/<model>/<file>; first hit wins")
    ap.add_argument("--out_dir", type=str, default="results/comparison")
    args = ap.parse_args()
    build([m.strip() for m in args.models.split(",")], [r.strip() for r in args.roots.split(",")], args.out_dir)
