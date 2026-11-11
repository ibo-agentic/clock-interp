"""No-clobber merge of the GE3d RUN 2 zip (Gemma-4B E3d minimal, gates passed) from match_GE3d_v2_results/. Run-1 files were renamed *_run1 beforehand. Never overwrites.
usage: python analysis/merge_GE3d_v2.py -> results/_GE3d_run2_merge_log.json"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = {"match_GE3d_v2_results": ([("results/mech_e3/gemma-3-4b-it", "results/mech_e3/gemma-3-4b-it"), ("results/gemma/gemma-3-4b-it_bf16", "results/gemma/gemma-3-4b-it_bf16"), ("logs", "results/gemma/GE3d_logs")],
                               [("STATUS.txt", "results/gemma/GE3d_STATUS.txt"), ("status.json", "results/gemma/GE3d_status.json"), ("env.txt", "results/gemma/GE3d_env.txt")])}
log = {"copied": [], "skipped_existing": []}
for src, (copy, files) in JOBS.items():
    pairs = list(files)
    for s, d in copy:
        for dp, _, fs in os.walk(os.path.join(ROOT, src, s)):
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), os.path.join(ROOT, src, s))
                pairs.append((os.path.join(s, rel), os.path.join(d, rel)))
    for s, d in pairs:
        sp, dp = os.path.join(ROOT, src, s), os.path.join(ROOT, d)
        if os.path.exists(dp):
            log["skipped_existing"].append(d); continue
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        shutil.copy2(sp, dp); log["copied"].append(d)
json.dump(log, open(os.path.join(ROOT, "results", "_GE3d_run2_merge_log.json"), "w"), indent=1)
print(f"copied {len(log['copied'])}, skipped (already existed) {len(log['skipped_existing'])}")
