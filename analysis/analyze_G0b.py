"""Merge (no-clobber) the F1a and G0b result folders into results/ and print the G0b summary (fp16 diagnostic, bf16/fp32 smoke, timing).
usage (repo root): python analysis/analyze_G0b.py [--merge]   -> results/gemma/G0b_report.txt"""
import argparse, json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
OUT = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)
def merge():
    log = []
    for src_root in ("match_F1a_results", "match_G0b_results"):
        base = os.path.join(src_root, "results")
        for dp, _, fs in os.walk(base):
            for f in fs:
                s = os.path.join(dp, f); d = os.path.join("results", os.path.relpath(s, base))
                if os.path.exists(d):
                    log.append(f"SKIP (exists) {d}"); continue
                os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d); log.append(f"{s} -> {d}")
    json.dump(log, open(os.path.join("results", "_F1a_G0b_merge_log.json"), "w"), indent=1)
    print(f"merged {sum(not l.startswith('SKIP') for l in log)} files, skipped {sum(l.startswith('SKIP') for l in log)}")
def report():
    g = os.path.join("results", "gemma")
    d = json.load(open(os.path.join(g, "gemma-3-4b-it_fp16_diag", "diag.json")))
    p("=== fp16 diagnostic, gemma-3-4b-it (forward hooks on every language-model / vision / projector module)")
    for k, v in d.items():
        if not isinstance(v, dict): continue
        fb = v["first_bad"]
        p(f"{k:<16} logits_finite={v['logits_finite']}  first non-finite output: {fb['name']} (nan={fb['nan']}, inf={fb['inf']}, finite max|x|={fb['maxabs']:.0f}); bad modules {v['n_bad_modules']}")
    p("largest finite activations (fp16 max = 65504):")
    for r in d["sdpa/with_image"]["top_finite"][:5]: p(f"   {r['name']:<58} {r['maxabs']:.0f}")
    p("\n=== smoke + timing")
    for name in ("gemma-3-4b-it_bf16", "gemma-3-4b-it_fp32"):
        j = json.load(open(os.path.join(g, name, "smoke.json"))); t = j["timing"]
        p(f"{name}: pass={j['smoke_pass']} load {j['load_seconds']:.0f}s peak GiB {({k: round(v, 1) for k, v in j['peak_memory_gib'].items()})} device_map {j['device_map']}  "
          f"text-only {t['text_only']['seconds_per_generation_mean']:.2f}s/gen, with-image {t['with_image']['seconds_per_generation_mean']:.2f}s/gen (prefill {t['with_image']['prefill_forward_seconds']:.2f}s, {t['with_image']['prompt_tokens']} tokens)")
    p("gemma-3-12b-it_bf16: NO smoke.json (CUDA OOM in the SigLIP vision tower on GPU 0 at the first image forward; see logs/05_smoke_12b_bf16.log)")
    open(os.path.join(g, "G0b_report.txt"), "w", encoding="utf-8").write("\n".join(OUT))
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    if a.merge: merge()
    report()
