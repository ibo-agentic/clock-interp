"""Old (git HEAD) vs new mech_lib.parse_hhmm on every saved reply text in results/mech_e1, mech_e2, mech_e3 (all models). Reply columns = any text column whose name contains
'raw' or 'answer' or 'reply' or 'gen' with string values. usage: python analysis/check_parse_hhmm_noop.py"""
import glob, os, re, subprocess, sys, types
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import mech_lib as NEW
src = subprocess.check_output(["git", "show", "HEAD:mech_lib.py"]).decode("utf-8")
OLD = types.ModuleType("old_mech_lib"); OLD.__file__ = "mech_lib.py"; exec(compile(src, "mech_lib.py", "exec"), OLD.__dict__)
files = sorted(glob.glob("results/mech_e[123]/**/*.csv", recursive=True))
n = nd = nmd = 0; per = {}; cols_seen = {}
for f in files:
    try: df = pd.read_csv(f, low_memory=False)
    except Exception: continue
    cols = [c for c in df.columns if re.search(r"raw|answer|reply|gen", c, re.I) and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]
    for c in cols:
        for t in df[c].dropna().astype(str):
            n += 1; per[f] = per.get(f, 0) + 1; cols_seen.setdefault(f, set()).add(c)
            nmd += bool(re.search(r"\*|`|__|~~", t))
            if OLD.parse_hhmm(t) != NEW.parse_hhmm(t):
                nd += 1; print("DIFF", f, c, repr(t[:80]), OLD.parse_hhmm(t), NEW.parse_hhmm(t))
print(f"CSV files scanned: {len(files)}; files with reply columns: {len(per)}; replies compared: {n}")
print(f"replies containing markdown characters: {nmd}; replies where old != new: {nd}")
print("RESULT:", "IDENTICAL" if nd == 0 else "DIFFERENCES")
for f, k in sorted(per.items()): print(f"  {k:>7} {f}  cols={sorted(cols_seen[f])}")
