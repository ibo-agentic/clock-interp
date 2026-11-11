"""No-clobber merge of the G2a (Gemma-3-12B) zip contents (unzipped to match_G2a_results/) into the repo. Never overwrites: an existing target is skipped and logged.
usage: python analysis/merge_G1.py   -> results/_G2a_merge_log.json"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "match_G2a_results")
COPY = [("results/gemma-3-12b-it", "results/gemma-3-12b-it"), ("results/gemma/gemma-3-12b-it_bf16", "results/gemma/gemma-3-12b-it_bf16"), ("logs", "results/gemma/G2a_logs"), ("conversion_images", "results/gemma-3-12b-it/conversion_images")]
FILES = [("STATUS.txt", "results/gemma/G2a_STATUS.txt"), ("status.json", "results/gemma/G2a_status.json"), ("env.txt", "results/gemma/G2a_env.txt")]
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
json.dump(log, open(os.path.join(ROOT, "results", "_G2a_merge_log.json"), "w"), indent=1)
print(f"copied {len(log['copied'])}, skipped (already existed) {len(log['skipped_existing'])}")
