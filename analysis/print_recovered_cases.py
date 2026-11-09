"""Print every describe reply where the tolerant parse recovers a number the strict regexes missed (strict None, tolerant a number),
with truth, for hand-checking.  usage: python analysis/print_recovered_cases.py results/internvl3-8b/describe.csv [--out FILE]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

import describe_check as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default=None)
a = ap.parse_args()
df = pd.read_csv(a.csv)
lines = []
n = 0
for i, r in df.iterrows():
    sm, sh, _ = D.parse_reply(r["raw_answer"])
    tm, th = D.parse_minute_tolerant(r["raw_answer"]), D.parse_hour_tolerant(r["raw_answer"])
    if (sm, sh) == (tm, th):
        continue
    n += 1
    lines += ["=" * 100,
              f"[row {i}] {r['filename']}  true time {int(r['true_hour']):02d}:{int(r['true_minute']):02d}  -> true minute-hand number {r['true_minute_number']}, true hour-hand number {r['true_hour_number']}",
              "REPLY: " + str(r["raw_answer"]).replace("\n", "\n       "),
              f"PARSED  strict: minute={sm} hour={sh}   |   tolerant: minute={tm} hour={th}   |   stored final answer {r['pred_hour']}:{r['pred_minute']}  answer_ok={r['answer_ok']}",
              f"        tolerant hour matches truth: {th == r['true_hour_number']}"]
head = f"{a.csv}: {n} of {len(df)} replies where the tolerant parse differs from the strict parse\n"
out = head + "\n".join(lines) + "\n"
print(out)
if a.out:
    open(a.out, "w", encoding="utf8").write(out)
