"""No-clobber merge of the G2b (Gemma-3-12B describe + behaviour) and GE3d (Gemma-4B E3d minimal; gate FAILED, no E3d output) zips (match_G2b_results/, match_GE3d_results/).
Never overwrites: an existing target is skipped and logged.   usage: python analysis/merge_G2b_GE3d.py -> results/_G2b_GE3d_merge_log.json"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = {"match_G2b_results": ([("results/gemma-3-12b-it", "results/gemma-3-12b-it"), ("results/gemma/gemma-3-12b-it_bf16", "results/gemma/gemma-3-12b-it_bf16"), ("logs", "results/gemma/G2b_logs")],
                              [("STATUS.txt", "results/gemma/G2b_STATUS.txt"), ("status.json", "results/gemma/G2b_status.json"), ("env.txt", "results/gemma/G2b_env.txt")]),
        "match_GE3d_results": ([("results/mech_e3/gemma-3-4b-it", "results/mech_e3/gemma-3-4b-it"), ("results/gemma/gemma-3-4b-it_bf16", "results/gemma/gemma-3-4b-it_bf16"), ("logs", "results/gemma/GE3d_logs")],
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
json.dump(log, open(os.path.join(ROOT, "results", "_G2b_GE3d_merge_log.json"), "w"), indent=1)
print(f"copied {len(log['copied'])}, skipped (already existed) {len(log['skipped_existing'])}")
