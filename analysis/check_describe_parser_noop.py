"""Does switching describe_check's PRIMARY parse from the strict regexes (MIN_RE / HOUR_RE) to the tolerant ones
(parse_minute_tolerant / parse_hour_tolerant) change any describe metric for the existing models?

For every saved describe CSV (results/**/describe*.csv, original zips excluded) it recomputes, from raw_answer alone:
said_minute_number, said_hour_number, pred_hour, pred_minute, minute_desc_ok, hour_desc_ok, answer_ok (and the stored tolerant hour columns)
under (a) the STRICT parse (old code, from git HEAD's describe_check logic = MIN_RE/HOUR_RE/TIME_RE) and (b) the TOLERANT parse, and compares both to the stored values.
usage: python analysis/check_describe_parser_noop.py        writes results/describe_parser_noop_check.txt"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd  # noqa: E402

import describe_check as D  # noqa: E402
from textnorm import normalize_reply  # noqa: E402


def strict(text):
    t = normalize_reply(text if isinstance(text, str) else "")
    m, h, tm = D.MIN_RE.search(t), D.HOUR_RE.search(t), D.TIME_RE.search(t)
    return (int(m.group(1)) if m else None, int(h.group(1)) if h else None, (int(tm.group(1)), int(tm.group(2))) if tm else None)


def tolerant(text):
    tm = D.TIME_RE.search(normalize_reply(text if isinstance(text, str) else ""))
    return (D.parse_minute_tolerant(text), D.parse_hour_tolerant(text), (int(tm.group(1)), int(tm.group(2))) if tm else None)


def metrics(df, parser):
    out = {k: [] for k in ["said_minute_number", "said_hour_number", "pred_hour", "pred_minute", "minute_desc_ok", "hour_desc_ok", "answer_ok"]}
    for raw, tmn, thn, th, tmi in zip(df["raw_answer"], df["true_minute_number"], df["true_hour_number"], df["true_hour"], df["true_minute"]):
        mn, hn, tm = parser(raw)
        out["said_minute_number"].append(mn); out["said_hour_number"].append(hn)
        out["pred_hour"].append(tm[0] if tm else None); out["pred_minute"].append(tm[1] if tm else None)
        out["minute_desc_ok"].append(mn is not None and min((mn - tmn) % 12, (tmn - mn) % 12) <= 1)
        out["hour_desc_ok"].append(hn is not None and hn == thn)
        out["answer_ok"].append(tm is not None and tm[0] % 12 == th % 12 and tm[1] == tmi)
    return pd.DataFrame(out)


def same(a, b):
    a, b = pd.Series(a).astype("float64"), pd.Series(b).astype("float64")
    return int((~((a == b) | (a.isna() & b.isna()))).sum())


def main():
    files = sorted(f for f in glob.glob(os.path.join(ROOT, "results", "**", "describe*.csv"), recursive=True))
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    P("DESCRIBE PARSER SWITCH CHECK: strict (MIN_RE/HOUR_RE) vs tolerant (MIN_RE_TOL/HOUR_RE_TOL, original tried first)")
    tot_strict_bad = tot_tol_diff = 0
    cols = ["said_minute_number", "said_hour_number", "pred_hour", "pred_minute", "minute_desc_ok", "hour_desc_ok", "answer_ok"]
    for f in files:
        df = pd.read_csv(f)
        rel = os.path.relpath(f, ROOT)
        if "raw_answer" not in df.columns or "true_minute_number" not in df.columns:
            P(f"{rel}: skipped (no raw_answer / true_minute_number columns)")
            continue
        s, t = metrics(df, strict), metrics(df, tolerant)
        bad_strict = sum(same(s[c], df[c]) for c in cols)   # does the recomputation reproduce the stored file?
        diff = {c: same(s[c], t[c]) for c in cols}
        tot_strict_bad += bad_strict; tot_tol_diff += sum(diff.values())
        P(f"\n{rel}: n={len(df)}  strict recompute vs stored: {bad_strict} cell mismatches")
        P(f"   strict vs tolerant differences per column: {diff}")
        for c in ["minute_desc_ok", "hour_desc_ok", "answer_ok"]:
            P(f"   {c}: strict {s[c].mean():.4f}  tolerant {t[c].mean():.4f}")
        P(f"   minute parse failures: strict {int(s['said_minute_number'].isna().sum())}  tolerant {int(t['said_minute_number'].isna().sum())}; "
          f"hour parse failures: strict {int(s['said_hour_number'].isna().sum())}  tolerant {int(t['said_hour_number'].isna().sum())}")
        ex = df.index[(s["said_minute_number"].fillna(-1) != t["said_minute_number"].fillna(-1)) | (s["said_hour_number"].fillna(-1) != t["said_hour_number"].fillna(-1))]
        for i in list(ex)[:3]:
            P(f"   e.g. row {i}: strict ({s.loc[i,'said_minute_number']},{s.loc[i,'said_hour_number']}) tolerant ({t.loc[i,'said_minute_number']},{t.loc[i,'said_hour_number']}) :: {str(df.loc[i,'raw_answer'])[:140]!r}")
    P(f"\nTOTAL: strict-recompute mismatches vs stored {tot_strict_bad}; strict-vs-tolerant cell differences {tot_tol_diff}")
    P("RESULT: " + ("NO-OP (tolerant parse changes no describe metric for any existing file)" if tot_tol_diff == 0 and tot_strict_bad == 0 else "NOT A NO-OP -- see differences above"))
    open(os.path.join(ROOT, "results", "describe_parser_noop_check.txt"), "w", encoding="utf8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
