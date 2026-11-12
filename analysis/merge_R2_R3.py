"""No-clobber merge of the R2 (styled clocks, Gemma-4B) and R3 (real photos, five models) zips into results/. Never overwrites; existing targets are skipped and logged.
R3 contains NO photo images (only data.csv labels + manifest_provenance.csv); this script refuses to copy any image file.
usage: python analysis/merge_R2_R3.py -> results/_R2_R3_merge_log.json"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = {"match_R2_results": ([("results", "results"), ("logs", "results/styles/R2_logs"), ("conversion_images_s1_thin", "results/styles/R2_conversion_images_s1_thin"),
                              ("conversion_images_s2_ticks", "results/styles/R2_conversion_images_s2_ticks")],
                             [("STATUS.txt", "results/styles/R2_STATUS.txt"), ("status.json", "results/styles/R2_status.json"), ("env.txt", "results/styles/R2_env.txt")]),
        "match_R3_results": ([("results", "results"), ("logs", "results/photos/R3_logs"), ("data_photos_tt500", "results/photos/data_photos_tt500")],
                             [("STATUS.txt", "results/photos/R3_STATUS.txt"), ("status.json", "results/photos/R3_status.json"), ("env.txt", "results/photos/R3_env.txt")])}
log = {"copied": [], "skipped_existing": []}
for src, (copy, files) in JOBS.items():
    pairs = list(files)
    for s, d in copy:
        for dp, _, fs in os.walk(os.path.join(ROOT, src, s)):
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), os.path.join(ROOT, src, s))
                pairs.append((os.path.join(s, rel), os.path.join(d, rel)))
    for s, d in pairs:
        if src == "match_R3_results" and s.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            raise SystemExit(f"refusing to copy an image from the photo run: {s}")
        sp, dp = os.path.join(ROOT, src, s), os.path.join(ROOT, d)
        if os.path.exists(dp):
            log["skipped_existing"].append(d); continue
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        shutil.copy2(sp, dp); log["copied"].append(d)
json.dump(log, open(os.path.join(ROOT, "results", "_R2_R3_merge_log.json"), "w"), indent=1)
print(f"copied {len(log['copied'])}, skipped (already existed) {len(log['skipped_existing'])}")
