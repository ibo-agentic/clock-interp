"""No-clobber merge of the G1 (Gemma-3-4B) zip contents (unzipped to match_G1_results/) into the repo. Never overwrites: an existing target is skipped and logged.
usage: python analysis/merge_G1.py   -> results/_G1_merge_log.json"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "match_G1_results")
COPY = [("results/gemma-3-4b-it", "results/gemma-3-4b-it"), ("results/gemma-3-4b-it_fp32_crosscheck", "results/gemma-3-4b-it_fp32_crosscheck"),
        ("results/gemma/gemma-3-4b-it_bf16", "results/gemma/gemma-3-4b-it_bf16"), ("probe_output/probe_results/gemma-3-4b-it", "probe_output/probe_results/gemma-3-4b-it"),
        ("probe_output/activations/gemma-3-4b-it", "probe_output/activations/gemma-3-4b-it"), ("logs", "results/gemma/G1_logs")]
FILES = [("STATUS.txt", "results/gemma/G1_STATUS.txt"), ("status.json", "results/gemma/G1_status.json"), ("env.txt", "results/gemma/G1_env.txt")]
log = {"copied": [], "skipped_existing": []}
pairs = list(FILES)
for s, d in COPY:
    for dp, _, fs in os.walk(os.path.join(SRC, s)):
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), os.path.join(SRC, s))
            pairs.append((os.path.join(s, rel), os.path.join(d, rel)))
for s, d in pairs:
    sp, dp = os.path.join(SRC, s), os.path.join(ROOT, d)
    if os.path.exists(dp):
        log["skipped_existing"].append(d); continue
    os.makedirs(os.path.dirname(dp), exist_ok=True)
    shutil.copy2(sp, dp); log["copied"].append(d)
json.dump(log, open(os.path.join(ROOT, "results", "_G1_merge_log.json"), "w"), indent=1)
print(f"copied {len(log['copied'])}, skipped (already existed) {len(log['skipped_existing'])}")
