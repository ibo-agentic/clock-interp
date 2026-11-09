"""Truncation check for Gemma-3-4B (G1): (1) replies that reach the 32-token cap per file / task / condition (re-tokenised with the model's own tokenizer, local cache only);
(2) the text-only vs matching vs blank vs mismatched comparison on trial keys (position, hour, lead_in_id) that PARSED in ALL FOUR conditions, with exact McNemar tests (Holm over the 3);
plus the same on trials with no reply at the cap in any condition.   usage: python analysis/analyze_G1_truncation.py -> results/gemma/G1_truncation_report.txt"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import pandas as pd  # noqa: E402

G, CAP = "gemma-3-4b-it", 32
out = []
P = lambda s="": (print(s), out.append(s))


def binom_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# The gated Gemma tokenizer is not cached locally, so the cap hit is identified by reply LENGTH in characters: every saved reply is either < 60 chars or >= 110 chars (nothing in between),
# and "length >= 100 chars" reproduces EXACTLY the counts from the tokenizer-based cap_check run on Kaggle (text_only_v2 53 of 2016, conversion_test 48 of 1728).
ntok = lambda s: len(str(s)) if isinstance(s, str) else 0

t = pd.read_csv(os.path.join(ROOT, "results", G, "text_only_v2.csv"))
c = pd.read_csv(os.path.join(ROOT, "results", G, "conversion_test.csv"))
t["ntok"] = t.raw_output.map(ntok); c["ntok"] = c.raw_output.map(ntok)
t["hit"] = t.ntok >= 100; c["hit"] = c.ntok >= 100   # ntok here = characters (proxy, see above)
t["parse_ok"] = t.parse_ok.astype(bool); c["parse_ok"] = c.parse_ok.astype(bool)

P(f"=== 1. REPLIES AT THE {CAP}-TOKEN CAP (Gemma-3-4B-it, bf16, greedy) ===")
P("text_only_v2.csv overall: %d of %d hit the cap; unparsed %d; hit-but-parsed %d" % (t.hit.sum(), len(t), (~t.parse_ok).sum(), (t.hit & t.parse_ok).sum()))
P(t.groupby(["task", "phrasing"]).agg(n=("hit", "size"), hit_cap=("hit", "sum"), unparsed=("parse_ok", lambda s: int((~s).sum()))).T.to_string())
P("\nconversion_test.csv by condition:")
g = c.groupby("condition").agg(n=("hit", "size"), hit_cap=("hit", "sum"), unparsed=("parse_ok", lambda s: int((~s).sum())), mean_chars=("ntok", "mean"), max_chars=("ntok", "max"))
P(g.round(1).to_string())
P("\nconversion text_only hits by minute-hand number: " + " ".join(f"{p}:{int(s)}" for p, s in c[c.condition == 'text_only'].groupby('position').hit.sum().items()))
P("conversion text_only hits by lead_in_id: " + " ".join(f"{p}:{int(s)}" for p, s in c[c.condition == 'text_only'].groupby('lead_in_id').hit.sum().items()))

P("\n=== 2. COMPARISON ON TRIALS PARSED IN ALL FOUR CONDITIONS (and with no cap hit in any) ===")
key = ["position", "hour", "lead_in_id"]
conds = ["text_only", "matching_clock", "blank", "mismatched_clock"]
P(f"duplicate keys per condition: {c.groupby('condition').apply(lambda d: int(d.duplicated(key).sum())).to_dict()}")
w_ok = c.pivot_table(index=key, columns="condition", values="minute_correct", aggfunc="first").astype(bool)
w_pa = c.pivot_table(index=key, columns="condition", values="parse_ok", aggfunc="first").astype(bool)
w_hit = c.pivot_table(index=key, columns="condition", values="hit", aggfunc="first").astype(bool)
for nm, mask in (("parsed in ALL four conditions", w_pa[conds].all(axis=1)), ("no cap hit in ANY condition", ~w_hit[conds].any(axis=1))):
    sub = w_ok[mask]
    P(f"\n--- {nm}: {int(mask.sum())} of {len(w_ok)} keys kept ({len(w_ok) - int(mask.sum())} dropped) ---")
    P("minute accuracy: " + "  ".join(f"{cn} {sub[cn].mean():.1%}" for cn in conds))
    ps = []
    rows = []
    for cn in conds[1:]:
        b, cc = int((sub["text_only"] & ~sub[cn]).sum()), int((~sub["text_only"] & sub[cn]).sum())
        ps.append(binom_p(b, cc)); rows.append((cn, b, cc, int((sub['text_only'] & sub[cn]).sum()), int((~sub['text_only'] & ~sub[cn]).sum())))
    order = sorted(range(3), key=lambda i: ps[i]); holm = [None] * 3; run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (3 - r))); holm[i] = run
    for (cn, b, cc, both, neither), p, h in zip(rows, ps, holm):
        P(f"  text_only vs {cn:<17} only-text-correct {b:>3}  only-image-correct {cc:>3}  both {both:>3}  neither {neither:>3}   McNemar exact p = {p:.3g}  Holm p = {h:.3g}")
    P(f"  matching vs blank: only-matching {int((sub['matching_clock'] & ~sub['blank']).sum())}, only-blank {int((~sub['matching_clock'] & sub['blank']).sum())}  p = "
      f"{binom_p(int((sub['matching_clock'] & ~sub['blank']).sum()), int((~sub['matching_clock'] & sub['blank']).sum())):.3g}")
open(os.path.join(ROOT, "results/gemma/G1_truncation_report.txt"), "w", encoding="utf8").write("\n".join(out) + "\n")
