"""G1 (Gemma-3-4B-it, bf16) analysis: step timings vs guesses, text-only conversion weakness and hour->minute leak (same label_answer as the other models),
what the image does (conversion conditions), bf16 vs fp32 behaviour cross-check, cap_check, probe vs the Qwen-3B probe.  CPU only.
usage: python analysis/analyze_G1.py  -> results/gemma/G1_report.txt"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mech_lib as ML  # noqa: E402

G = "gemma-3-4b-it"
MODELS = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl3-2b", "internvl3-8b", G]
GUESS = {"00_gate_4b_bf16": 8, "01_text_only_v2": 35, "02_conversion_test": 115, "03_aggregate_text_only": 2, "04_aggregate_conversion": 2, "05_cap_check": 3,
         "06_describe_check": 90, "07_eval_behavior_bf16": 42, "08_eval_behavior_fp32_crosscheck": 35, "09_probe_gemma4b": 60, "10_purge_4b_cache": 1}
out = []
P = lambda s="": (print(s), out.append(s))


def binom_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n):
    if n == 0:
        return (float("nan"),) * 2
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---- 1 steps
P("=== 1. G1 STEPS (zip STATUS) vs the guesses written in the notebook ===")
st = json.load(open(os.path.join(ROOT, "results/gemma/G1_status.json")))
for s in st:
    P(f"  {s['step']:<34} {s['status']:<4} {s['minutes']:>6.1f} min   guess {GUESS.get(s['step'], '?'):>4}")
P(f"  total {sum(s['minutes'] for s in st) / 60:.2f} h (guess 6.5 h); gate: {st[0]['status']}")
sm = json.load(open(os.path.join(ROOT, "results/gemma/gemma-3-4b-it_bf16/smoke.json")))
P(f"  gate smoke: finite logits + 256 image tokens + one BOS passed (smoke_pass={sm['smoke_pass']}); peak memory {sm['peak_memory_gib']}; device map {sm['device_map']}")

# ---- 2 text-only weakness + leak, all five models, same labeller
P("\n=== 2. TEXT-ONLY two-hand conversion (conversion_test, condition text_only, n=432): label of every answer (same label_answer as E3d/E1) ===")
rows = []
for m in MODELS:
    c = pd.read_csv(os.path.join(ROOT, "results", m, "conversion_test.csv"))
    to = c[c.condition == "text_only"]
    labs = [ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute))[0] for r in to.itertuples()]
    vc = pd.Series(labs).value_counts()
    k = int(vc.get("leak", 0))
    lo, hi = wilson(k, len(to))
    rows.append({"model": m, "n": len(to), "minute_correct": int(vc.get("correct", 0)), "leak(hour*5)": k, "leak%": f"{k / len(to):.1%} [{lo:.1%},{hi:.1%}]", "adjacent": int(vc.get("adjacent", 0)),
                 "default_0_30": int(vc.get("default_0_30", 0)), "other": int(vc.get("other", 0)), "unparsed": int(vc.get("unparsed", 0))})
P(pd.DataFrame(rows).to_string(index=False))
c = pd.read_csv(os.path.join(ROOT, "results", G, "conversion_test.csv"))
to = c[c.condition == "text_only"].copy()
to["lab"] = [ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute))[0] for r in to.itertuples()]
to["H12"] = to.hour.astype(int) == 12
g = to.groupby("H12").lab.apply(lambda s: pd.Series({"n": len(s), "leak": int((s == "leak").sum()), "correct": int((s == "correct").sum())})).unstack()
P("\nGemma-4B text-only by hour (H=12 -> leak minute 0): " + g.to_string().replace("\n", " | "))
pos = to.groupby("position").agg(n=("lab", "size"), correct=("lab", lambda s: int((s == "correct").sum())), leak=("lab", lambda s: int((s == "leak").sum())))
P("Gemma-4B text-only by minute-hand number (position): " + " ".join(f"{p}:{r.correct}/{r.n}(leak {r.leak})" for p, r in pos.iterrows()))

# ---- 3 image conditions
P("\n=== 3. WHAT THE IMAGE DOES (Gemma-4B, conversion conditions, n=432 each) ===")
rows = []
for cond, d in c.groupby("condition"):
    labs = [ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute), None if pd.isna(r.image_minute) else int(r.image_minute))[0] for r in d.itertuples()]
    vc = pd.Series(labs).value_counts()
    rows.append({"condition": cond, "n": len(d), "minute_correct": int(d.minute_correct.astype(bool).sum()), "minute_acc": f"{d.minute_correct.astype(bool).mean():.1%}",
                 "leak": int(vc.get("leak", 0)), "follows_image": int(vc.get("follows_image", 0)), "unparsed": int(vc.get("unparsed", 0))})
P(pd.DataFrame(rows).to_string(index=False))
pt = pd.read_csv(os.path.join(ROOT, "results/conversion_test/paired_tests.csv"))
P(pt[pt.model == G].to_string(index=False) if "model" in pt.columns else pt.to_string(index=False))

# ---- 4 bf16 vs fp32
P("\n=== 4. BEHAVIOUR (data/, 500 clocks): bf16 vs fp32 cross-check ===")
a = pd.read_csv(os.path.join(ROOT, "results", G, "results.csv"))
b = pd.read_csv(os.path.join(ROOT, "results", G + "_fp32_crosscheck", "results.csv"))
assert (a.filename == b.filename).all()


def ex(d):
    return ((d.pred_hour % 12 == d.true_hour % 12) & (d.pred_minute == d.true_minute)).fillna(False).astype(bool)


for nm, d in (("bf16", a), ("fp32", b)):
    e = ex(d)
    P(f"  {nm}: exact {e.mean():.1%} ({int(e.sum())}/500)  hour {(d.pred_hour % 12 == d.true_hour % 12).mean():.1%}  minute {(d.pred_minute == d.true_minute).mean():.1%}  parse {d.parse_success.mean():.1%}")
ea, eb = ex(a), ex(b)
same_raw = (a.raw_answer.astype(str) == b.raw_answer.astype(str))
same_time = ((a.pred_hour == b.pred_hour) & (a.pred_minute == b.pred_minute)) | (a.pred_hour.isna() & b.pred_hour.isna())
P(f"  per-image agreement: identical raw reply {int(same_raw.sum())}/500 ({same_raw.mean():.1%}); identical parsed time {int(same_time.sum())}/500 ({same_time.mean():.1%})")
bb, cc = int((ea & ~eb).sum()), int((~ea & eb).sum())
P(f"  exact-match paired: bf16-only {bb}, fp32-only {cc}, both {int((ea & eb).sum())}; exact binomial p = {binom_p(bb, cc):.3g}")
hm = (a.pred_minute == a.true_minute); hn = (b.pred_minute == b.true_minute)
P(f"  minute-correct paired: bf16-only {int((hm & ~hn).sum())}, fp32-only {int((~hm & hn).sum())}; p = {binom_p(int((hm & ~hn).sum()), int((~hm & hn).sum())):.3g}")

# ---- 5 cap check
P("\n=== 5. 32-token cap (cap_check) ===")
P(pd.read_csv(os.path.join(ROOT, "results", G, "cap_check_32.csv")).to_string(index=False))

# ---- 6 probe
P("\n=== 6. PROBE (data_balanced/, 480 clocks) Gemma-4B vs the existing Qwen-3B probe (probe_output/probe_results/summary.txt) ===")
def probe_rows(path):
    return [l for l in open(path, encoding="utf8").read().split(chr(10)) if l.startswith(("representation", "hidden_", "vision_"))]


P("Gemma-4B (columns: representation hand layer [data] n_all n_corr n_wrong R2_all R2_corr R2_wrong R2_shuf acc_all acc_corr acc_wrong acc_shuf acc_majrty; acc = 12-way classifier accuracy):")
for l in probe_rows(os.path.join(ROOT, "probe_output/probe_results/gemma-3-4b-it/summary.txt")):
    P("  " + l)
P("Qwen-3B (existing probe, same script; its R2_corr column overflowed for 12 correct images):")
for l in probe_rows(os.path.join(ROOT, "probe_output/probe_results/summary.txt")):
    P("  " + l)
open(os.path.join(ROOT, "results/gemma/G1_report.txt"), "w", encoding="utf8").write("\n".join(out) + "\n")
