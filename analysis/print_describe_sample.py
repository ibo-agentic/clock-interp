"""Print N random describe replies (default 30, seed 0) with truth and parsed values (strict and tolerant) for hand-checking.
usage: python analysis/print_describe_sample.py results/gemma-3-4b-it/describe.csv [--n 30] [--seed 0]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

import describe_check as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
df = pd.read_csv(a.csv)
sub = df.sample(n=min(a.n, len(df)), random_state=a.seed).sort_index()
print(f"{a.csv}: {len(df)} replies; showing {len(sub)} random (seed {a.seed}).  STRICT = original regexes, TOL = tolerant parse; truth numbers are the clock numbers 1-12 nearest each hand\n")
for i, r in sub.iterrows():
    sm, sh, st = D.parse_reply(r["raw_answer"])
    print("=" * 100)
    print(f"[row {i}] {r['filename']}  true time {int(r['true_hour']):02d}:{int(r['true_minute']):02d}  -> true minute-hand number {r['true_minute_number']}, true hour-hand number {r['true_hour_number']}")
    print("REPLY:", str(r["raw_answer"]).replace("\n", "\n       "))
    print(f"PARSED  strict: minute={sm} hour={sh} time={st}   |   TOL: minute={D.parse_minute_tolerant(r['raw_answer'])} hour={D.parse_hour_tolerant(r['raw_answer'])}")
