"""prep_ticktock_photos.py -- real-photo evaluation set for experiment R (PREREG_MECHANISM.md ninth entry, item 4).

Source: TickTockVQA (paper "It's Time to Get It Right", arXiv 2603.08011), Hugging Face mirror monica-sekoyan/TickTockVQA-segmented.
EVALUATION ONLY, NO REDISTRIBUTION: the images are downloaded inside the session and written to a local folder that is never added to the repository or to a shared
dataset; only the provenance manifest (no pixels) is meant to be kept. Cite the paper and the dataset wherever results are reported.

Selection (fixed in advance): split `test`, per-row `license == "CC BY 4.0"`, rows sorted by `sample_id`, 500 rows drawn without replacement with
numpy.random.RandomState(0); image = the `image` column (the segmented clock crop); `test_small` is not used. Labels: `hour` stored as 1-12 (hour % 12, 0 -> 12;
the original value is kept in the manifest) and `minute` exactly.
Writes <out>/tt_0000.png ... , <out>/data.csv (filename, hour, minute, minute_position) usable by eval_behavior.py / describe_check.py, and
<out>/manifest_provenance.csv (filename, sample_id, hour_original, minute, source, license, license_url, source_url, image_name).

  python prep_ticktock_photos.py --out data_photos_tt500                (downloads the test parquet shards, needs internet; HF token not required)
"""
import argparse
import io
import os

import numpy as np
import pandas as pd

REPO = "monica-sekoyan/TickTockVQA-segmented"
LICENSE_KEEP = "CC BY 4.0"
N_SAMPLE, SEED = 500, 0
META_COLS = ["sample_id", "hour", "minute", "source", "license", "license_url", "source_url", "image_name"]


def select_rows(df, n=N_SAMPLE, seed=SEED, license_keep=LICENSE_KEEP):
    """df needs META_COLS (+ image). Returns the sampled rows, in drawn order, as a new DataFrame (index reset)."""
    d = df[df["license"] == license_keep].sort_values("sample_id").reset_index(drop=True)
    if len(d) < n:
        raise ValueError(f"only {len(d)} rows have license == {license_keep!r}; need {n} (the licence rule is not loosened automatically)")
    idx = np.random.RandomState(seed).choice(len(d), size=n, replace=False)
    return d.iloc[idx].reset_index(drop=True)


def write_set(sel, out):
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    rows, prov = [], []
    for i, r in sel.iterrows():
        img = r["image"]
        raw = img["bytes"] if isinstance(img, dict) else img
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        fn = f"tt_{i:04d}.png"
        im.save(os.path.join(out, fn))
        h12 = int(r["hour"]) % 12 or 12
        m = int(r["minute"])
        rows.append({"filename": fn, "hour": h12, "minute": m, "minute_position": (round(m / 5) % 12) or 12})
        prov.append({"filename": fn, "sample_id": r["sample_id"], "hour_original": int(r["hour"]), "minute": m, "source": r["source"], "license": r["license"],
                     "license_url": r["license_url"], "source_url": r["source_url"], "image_name": r["image_name"], "width": im.width, "height": im.height})
    pd.DataFrame(rows).to_csv(os.path.join(out, "data.csv"), index=False)
    pd.DataFrame(prov).to_csv(os.path.join(out, "manifest_provenance.csv"), index=False)
    return len(rows)


def load_test_split(cache_dir=None):
    """Download only the `test` parquet shards (not test_small, not image_full) and read the needed columns."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    shards = sorted(f for f in files if os.path.basename(f).startswith("test-"))
    if not shards:
        raise RuntimeError(f"no 'test-*.parquet' shard found in {REPO}; files: {files[:20]}")
    frames = []
    for f in shards:
        p = hf_hub_download(REPO, f, repo_type="dataset", cache_dir=cache_dir)
        frames.append(pq.read_table(p, columns=META_COLS + ["image"]).to_pandas())
        print(f"read {f}: {len(frames[-1])} rows")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_photos_tt500")
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--cache_dir", default=None)
    a = ap.parse_args()
    df = load_test_split(a.cache_dir)
    print(f"test rows {len(df)}; licence values: {df['license'].value_counts().to_dict()}; sources of CC BY 4.0 rows: {df[df['license'] == LICENSE_KEEP]['source'].value_counts().to_dict()}")
    sel = select_rows(df, a.n)
    n = write_set(sel, a.out)
    print(f"wrote {n} photos + data.csv + manifest_provenance.csv to {a.out}; sources in the sample: {sel['source'].value_counts().to_dict()}")
