"""Notebook D health check + checks-before-results + the analysable (InternVL3-8B) parts.

Reads the unzipped notebook-D folder and the merged results/. CPU only; prints everything and writes
results/mech_e3/notebook_D_health.txt.   Usage:  python analysis/analyze_notebook_D.py [match_D_results]
Nothing here touches the model; models whose verify gate did not pass are NOT analysed (prereg rule).
"""
import glob, io, json, os, re, sys
import numpy as np
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "match_D_results"
buf = io.StringIO()


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); buf.write(s + "\n")


P("=== 1. RUN HEALTH ===")
P(open(f"{D}/STATUS.txt").read().split("MISSING")[0].rstrip())
P("\n--- verify gate per model (from step logs; verify.json is only written when the gate code finishes) ---")
for tag, name in (("01_", "qwen2.5-vl-3b"), ("04_", "internvl3-2b"), ("08_", "qwen2.5-vl-7b")):
    log = open(glob.glob(f"{D}/logs/{tag}*.log")[0], errors="replace").read()
    gates = re.findall(r"\[(PASS|FAIL)\] (G\w+)", log)
    err = re.findall(r"^(\w+Error: .*)$", log, flags=re.M)
    P(f"{name}: gates reached = {gates or 'none'};  crash = {err[-1] if err else 'none'};  verify.json exists = "
      f"{os.path.exists(f'{D}/results/mech_e3/{name}/verify.json')}")
P("-> G3..G9 were never reached for any model (crash inside the G4 logit-lens check). No model has a passing gate.")

P("\n=== 3. CHECKS BEFORE RESULTS ===")
P("--- E3e coverage: clocks with BOTH hands exactly right in describe.csv, and how many have an extractable verbatim description ---")
CUT = re.compile(r"(?i)\b(therefore|finally|thus|so,? the time|the time (?:shown|is))")
for m, path in (("qwen2.5-vl-3b", "results/qwen2.5-vl-3b/describe.csv"), ("internvl3-2b", "results/internvl3-2b/describe.csv"),
                ("qwen2.5-vl-7b", "qwen_prior/results/qwen2.5-vl-7b/describe.csv"), ("internvl3-8b", "results/internvl3-8b/describe.csv")):
    d = pd.read_csv(path)
    sel = d[(d.said_minute_number == d.true_minute_number) & d.hour_desc_ok.astype(bool)]
    ok = 0
    for raw in sel.raw_answer.astype(str):
        mm = CUT.search(raw)
        ok += bool(mm and len(raw[:mm.start()].strip()) > 20)
    P(f"{m}: both-exact={len(sel)}  verbatim-extractable={ok}  dropped={len(sel) - ok}  "
      f"with-image wrong rate on these={1 - sel.answer_ok.astype(bool).mean():.3f}   [offline recomputation with the same regex as mech_e3.py]")
e = f"results/mech_e3/internvl3-8b/e3e_summary.csv"
P("n_clocks actually run (only InternVL3-8B ran):"); P(pd.read_csv(e)[["variant", "n_clocks"]].to_string(index=False))

P("\n--- text-light pilot (hour in text, minute only in image) ---")
ps = pd.read_csv("results/mech_e3/internvl3-8b/text_light_pilot_summary.csv")
P(ps.to_string(index=False)); P("Only InternVL3-8B has a pilot; the other three crashed before the pilot.")

P("\n--- InternVL3-8B at max_new_tokens=128 ---")
P(pd.read_csv("results/internvl3-8b/cap_check_128.csv").to_string(index=False))
for n in ("text_only_v2", "conversion_test"):
    for suf, lab in (("_mnt64", "64 tok"), ("", "128 tok")):
        x = pd.read_csv(f"results/internvl3-8b/{n}{suf}.csv")
        P(f"{n:<16} {lab:<8} n={len(x)} parse_ok={x.parse_ok.mean():.4f} minute_correct={x.minute_correct.mean():.4f}")
x = pd.read_csv("results/internvl3-8b/conversion_test.csv")
P(x.groupby("condition").agg(n=("minute_correct", "size"), minute_correct=("minute_correct", "mean"), parse_ok=("parse_ok", "mean")).round(4).to_string())

P("\n=== 4. RESULTS (only what was validly run) ===")
P("Confirmatory tests (E3d for Qwen-3B / InternVL3-2B, E3e verbatim for Qwen-3B, E3a/b/c/f): NOT RUN -- gate failed/crashed.")
P("InternVL3-8B E3e (prereg lists the 8B among the 'analogous sets'; no confirmatory claim is attached to the 8B):")
P(pd.read_csv(e).T.to_string())
d = pd.read_csv("results/mech_e3/internvl3-8b/e3e_transplant.csv")
g = d[d.variant == "describe_style"]
P(f"describe_style exploratory variant: {int((g.raw.str.len() > 200).sum())}/{len(g)} replies are long chain-of-thought text (cut off at 128 tokens) -> 'wrong' mostly = unparsed, NOT a language-side result.")

os.makedirs("results/mech_e3", exist_ok=True)
open("results/mech_e3/notebook_D_health.txt", "w", encoding="utf-8").write(buf.getvalue())
