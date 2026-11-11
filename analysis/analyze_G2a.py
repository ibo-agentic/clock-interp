"""G2a (Gemma-3-12B-it bf16, text_only_v2 + conversion_test at 256 tokens) analysis. CPU only.
usage: python analysis/analyze_G2a.py -> results/gemma/G2a_report.txt"""
import json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd  # noqa: E402
import mech_lib as ML  # noqa: E402

G = "gemma-3-12b-it"
MODELS = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl3-2b", "internvl3-8b", "gemma-3-4b-it", G]
GUESS = {"00_gate_12b_bf16": 14, "01_text_only_v2": 100, "02_conversion_test": 240, "03_aggregate_text_only": 2, "04_aggregate_conversion": 2, "05_cap_check": 3, "06_purge_12b_cache": 1}
out = []
P = lambda s="": (print(s), out.append(s))


def wilson(k, n):
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.1%},{hi:.1%}]"


P("=== 1. STEPS ===")
st = json.load(open(os.path.join(ROOT, "results/gemma/G2a_status.json")))
for s in st:
    P(f"  {s['step']:<26} {s['status']:<4} {s['minutes']:>6.1f} min   guess {GUESS.get(s['step'], '?'):>4}")
P(f"  total {sum(s['minutes'] for s in st) / 60:.2f} h (guess ~6 h)")
rc = json.load(open(os.path.join(ROOT, "results/gemma/gemma-3-12b-it_bf16/run_config.json")))
P(f"  device map counts {rc['device_map_counts']}; peak GiB {rc['peak_memory_gib']}; load {rc['load_seconds']:.0f} s; dtype {rc['dtype']}")
tm = json.load(open(os.path.join(ROOT, "results/gemma/gemma-3-12b-it_bf16/smoke.json")))
P(f"  timing: {tm.get('timing', tm.get('timings', 'n/a'))}")

P("\n=== 2. CAP CHECK at 256 (rule: 0 hits in conversion_test, <=5 in text_only_v2) ===")
cc = pd.read_csv(os.path.join(ROOT, "results", G, "cap_check_256.csv"))
P(cc.to_string(index=False))
t2 = pd.read_csv(os.path.join(ROOT, "results", G, "text_only_v2.csv"))
cv = pd.read_csv(os.path.join(ROOT, "results", G, "conversion_test.csv"))
P(f"  reply length (chars) max: text_only_v2 {t2.raw_output.astype(str).str.len().max()}, conversion_test {cv.raw_output.astype(str).str.len().max()}; unparsed text_only_v2 {int((~t2.parse_ok.astype(bool)).sum())}, conversion {int(cv.minute_correct.isna().sum())}")
P("  VERDICT: " + ("rule MET (0 and 0 hits; residual hits: none)" if int(cc.hit_cap.sum()) == 0 else f"rule NOT met, residual hits {cc.set_index('file').hit_cap.to_dict()}"))

P("\n=== 3. TEXT-ONLY minute accuracy, two hands (b2 headline in text_only_v2; text_only cell of conversion_test) ===")
rows = []
for m in MODELS:
    t = pd.read_csv(os.path.join(ROOT, "results", m, "text_only_v2.csv"))
    b2 = t[(t.task == "b") & (t.phrasing == "b2")]
    c = pd.read_csv(os.path.join(ROOT, "results", m, "conversion_test.csv"))
    to = c[c.condition == "text_only"]
    labs = pd.Series([ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute))[0] for r in to.itertuples()])
    vc = labs.value_counts()
    rows.append({"model": m, "b2_minute": fmt(int(b2.minute_correct.astype(bool).sum()), len(b2)), "conv_text_only_minute": fmt(int(to.minute_correct.astype(bool).sum()), len(to)),
                 "leak": fmt(int(vc.get("leak", 0)), len(to)), "adjacent": int(vc.get("adjacent", 0)), "default_0_30": int(vc.get("default_0_30", 0)), "other": int(vc.get("other", 0)), "unparsed": int(vc.get("unparsed", 0))})
P(pd.DataFrame(rows).to_string(index=False))
P("(Gemma-4B rows are the 32-token G1 files; at 128 tokens G1b: 39.4% minute, leak 133.)")
to = cv[cv.condition == "text_only"].copy()
to["lab"] = [ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute))[0] for r in to.itertuples()]
pos = to.groupby("position").agg(n=("lab", "size"), correct=("lab", lambda s: int((s == "correct").sum())), leak=("lab", lambda s: int((s == "leak").sum())))
P("12B text-only by minute-hand number: " + " ".join(f"{p}:{r.correct}/{r.n}(leak {r.leak})" for p, r in pos.iterrows()))
to["H12"] = to.hour.astype(int) == 12
P("12B text-only by hour==12: " + to.groupby("H12").lab.apply(lambda s: f"n={len(s)} correct={int((s == 'correct').sum())} leak={int((s == 'leak').sum())}").to_string().replace("\n", " | "))

P("\n=== 4. CONVERSION conditions (n=432 each, minute accuracy vs the text; paired exact McNemar vs text_only) ===")
for m in MODELS:
    c = pd.read_csv(os.path.join(ROOT, "results", m, "conversion_test.csv"))
    key = ["hour", "position"] + [k for k in ("lead_in_id", "variant", "trial") if k in c.columns]
    base = c[c.condition == "text_only"].set_index(key).minute_correct.astype(bool)
    line = [f"{m:<15}"]
    for cond in ("text_only", "matching_clock", "blank", "mismatched_clock"):
        d = c[c.condition == cond]
        line.append(f"{cond}: {d.minute_correct.astype(bool).mean():.1%}")
        if cond != "text_only":
            o = d.set_index(key).minute_correct.astype(bool).reindex(base.index)
            b, cc_ = int((base & ~o).sum()), int((~base & o).sum())
            line[-1] += f" (text-only-better {b}/image-better {cc_}, p={mcnemar(b, cc_):.2g})"
    P("  " + " | ".join(line))
c = cv
P("\n=== 5. MISMATCHED clock: what does the answer follow? (12B and the other models) ===")
for m in MODELS:
    c = pd.read_csv(os.path.join(ROOT, "results", m, "conversion_test.csv"))
    d = c[c.condition == "mismatched_clock"]
    labs = pd.Series([ML.label_answer(str(r.raw_output), int(r.hour), int(r.truth_minute), None if pd.isna(r.image_minute) else int(r.image_minute))[0] for r in d.itertuples()])
    vc = labs.value_counts()
    P(f"  {m:<15} follows_image_minute {fmt(int(vc.get('follows_image', 0)), len(d))}  correct(text) {int(vc.get('correct', 0))}  leak {int(vc.get('leak', 0))}  other {int(vc.get('other', 0))}")
open(os.path.join(ROOT, "results/gemma/G2a_report.txt"), "w", encoding="utf8").write("\n".join(out) + "\n")
