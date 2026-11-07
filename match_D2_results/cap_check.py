"""cap_check.py -- how many saved replies reached the max_new_tokens cap (i.e. were cut off)?

A greedy reply that ends with EOS is shorter than the cap; one that hits the cap was truncated.  Replies
are re-tokenised with the model's own tokenizer (special tokens excluded), so a count >= cap means
"hit (or exceeded, after round-tripping) the cap".  CPU only; downloads only the tokenizer.

USAGE:
  python cap_check.py --model_id OpenGVLab/InternVL3-8B-hf --cap 64 \
        --csv results/internvl3-8b/text_only_v2.csv:raw_output results/internvl3-8b/conversion_test.csv:raw_output
  (each --csv is  path:column ; add --cap_describe-style runs by passing --cap 120 with describe.csv:raw_answer)
Writes <out_csv> (default cap_check.csv next to the first CSV) with one row per file.
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--csv", nargs="+", required=True, help="path:column pairs")
    ap.add_argument("--out_csv", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_id)
    rows = []
    for spec in a.csv:
        path, col = spec.rsplit(":", 1)
        d = pd.read_csv(path)
        n = d[col].fillna("").map(lambda s: len(tok(str(s), add_special_tokens=False)["input_ids"]))
        parse = d["parse_ok"].astype(bool) if "parse_ok" in d.columns else None
        row = {"file": path, "cap": a.cap, "n": len(d), "max_tokens": int(n.max()), "p99": int(np.percentile(n, 99)),
               "hit_cap": int((n >= a.cap).sum()), "within_1_of_cap": int((n >= a.cap - 1).sum()),
               "unparsed": (int((~parse).sum()) if parse is not None else None)}
        rows.append(row)
        print(row, flush=True)
    out = a.out_csv or os.path.join(os.path.dirname(a.csv[0].rsplit(":", 1)[0]) or ".", "cap_check.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
    if any(r["hit_cap"] for r in rows):
        print("WARNING: some replies still hit the cap -- consider a larger --max_new_tokens.")


if __name__ == "__main__":
    main()
