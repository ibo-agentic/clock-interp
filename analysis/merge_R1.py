"""No-clobber merge of the R1 zip (styled clocks, match_R1_results/) into results/. Never overwrites; existing targets are skipped and logged.
usage: python analysis/merge_R1.py -> results/_R1_merge_log.json"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "match_R1_results")
COPY = [("results", "results"), ("logs", "results/styles/R1_logs"), ("conversion_images_s1_thin", "results/styles/R1_conversion_images_s1_thin"),
        ("conversion_images_s2_ticks", "results/styles/R1_conversion_images_s2_ticks")]
FILES = [("STATUS.txt", "results/styles/R1_STATUS.txt"), ("status.json", "results/styles/R1_status.json"), ("env.txt", "results/styles/R1_env.txt")]
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
json.dump(log, open(os.path.join(ROOT, "results", "_R1_merge_log.json"), "w"), indent=1)
print(f"copied {len(log['copied'])}, skipped (already existed) {len(log['skipped_existing'])}")
