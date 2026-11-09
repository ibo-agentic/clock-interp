"""Proof that the markdown-stripping step (textnorm.normalize_reply) changes nothing for the existing models.
For every saved reply (columns raw_answer / raw) in every results CSV under results/ (all models, incl. sensitivity folders), it
 (1) counts replies that contain any markdown character handled by the normaliser,
 (2) recomputes the parses with the CURRENT code and with the code at git HEAD (pre-change) and requires identical output,
 (3) for describe/results/text-only/conversion CSVs recomputes the stored parsed columns with the current code and requires them equal to the stored values.
usage (repo root): python analysis/check_normalization_noop.py     -> results/normalization_noop_check.txt ; exit code 1 on any difference."""
import glob, os, re, subprocess, sys, types
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import eval_behavior as EB, describe_check as DC, textnorm
OUT = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

def head_module(path):
    src = subprocess.check_output(["git", "show", f"HEAD:{path}"]).decode("utf-8")
    m = types.ModuleType("old_" + os.path.basename(path)[:-3]); m.__file__ = path
    exec(compile(src, path, "exec"), m.__dict__)
    return m
old_eb, old_dc = head_module("eval_behavior.py"), head_module("describe_check.py")
INT_RE = re.compile(r"-?\d+")
files = sorted(f for f in glob.glob("results/**/*.csv", recursive=True) if "mech_" not in f and "match_" not in f)
n_replies = n_md = n_diff = n_stored_diff = 0
checked_files = []
for f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception:
        continue
    col = "raw_answer" if "raw_answer" in df.columns else ("raw_output" if "raw_output" in df.columns else None)
    if col is None:
        continue
    raws = df[col].fillna("").astype(str).tolist()
    checked_files.append((f, len(raws)))
    for r in raws:
        n_replies += 1
        n_md += bool(re.search(r"\*|`|__|~~", r))
        if (EB.parse_time_answer(r) != old_eb.parse_time_answer(r) or DC.parse_reply(r) != old_dc.parse_reply(r)
                or DC.parse_hour_tolerant(r) != old_dc.parse_hour_tolerant(r)
                or (INT_RE.search(r) and INT_RE.search(r).group(0)) != (INT_RE.search(textnorm.normalize_reply(r)) and INT_RE.search(textnorm.normalize_reply(r)).group(0))):
            n_diff += 1
    # stored columns
    if {"pred_hour", "pred_minute"} <= set(df.columns) and "true_hour" in df.columns:      # results.csv / describe.csv style
        for r, ph, pm in zip(raws, df.pred_hour, df.pred_minute):
            if "said_minute_number" in df.columns:
                mn, hn, tm = DC.parse_reply(r); h_, m_ = (tm if tm else (None, None))
                ok = (pd.isna(ph) and h_ is None) or (not pd.isna(ph) and h_ == ph and m_ == pm)
            else:
                h_, m_, okp = EB.parse_time_answer(r)
                ok = (pd.isna(ph) and h_ is None) or (not pd.isna(ph) and h_ == ph and m_ == pm)
            n_stored_diff += (not ok)
    elif "parse_ok" in df.columns and "parsed_answer" in df.columns:                        # text-only v2 / conversion test: full score() recomputed
        import text_only_v2 as T2
        for (_, row), r in zip(df.iterrows(), raws):
            task = row["task"] if "task" in df.columns else "b"
            sc = T2.score({"task": task, "hour": row["hour"], "truth_minute": row["truth_minute"]}, r)
            same = (bool(sc["parse_ok"]) == bool(row["parse_ok"]) and bool(sc["format_ok"]) == bool(row["format_ok"]) and bool(sc["minute_correct"]) == bool(row["minute_correct"])
                    and (pd.isna(row["parsed_answer"]) if sc["parsed_answer"] is None else str(sc["parsed_answer"]) == str(row["parsed_answer"])))
            n_stored_diff += (not same)
p(f"CSV files with a reply column: {len(checked_files)}; replies checked: {n_replies}")
p(f"replies containing markdown characters (* ` __ ~~): {n_md}")
p(f"replies where any old-vs-new PARSE output differs (time parse, describe parse, tolerant hour parse, integer parse): {n_diff}")
p(f"replies where the current parser disagrees with the STORED parsed columns: {n_stored_diff}")
ok = (n_diff == 0 and n_stored_diff == 0)
p("RESULT:", ("IDENTICAL: no parse output and no stored result changes" + (f" ({n_md} existing replies DO contain markdown characters, but none of them changes any parsed value)" if n_md else " (no existing reply contains markdown)")) if ok else "DIFFERENCES FOUND")
open("results/normalization_noop_check.txt", "w", encoding="utf-8").write("\n".join(OUT + ["", "files: " + ", ".join(f"{a}({b})" for a, b in checked_files)]))
sys.exit(0 if ok else 1)
