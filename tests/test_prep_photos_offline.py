"""Offline test of prep_ticktock_photos.py selection and writing (synthetic rows, no download)."""
import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import prep_ticktock_photos as P  # noqa: E402


def png_bytes(c):
    b = io.BytesIO()
    Image.new("RGB", (40 + c % 7, 30), (c % 255, 10, 10)).save(b, "PNG")
    return b.getvalue()


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")
    rng = np.random.RandomState(1)
    n = 700
    df = pd.DataFrame({"sample_id": [f"s{i:05d}" for i in rng.permutation(n)], "hour": rng.randint(0, 24, n), "minute": rng.randint(0, 60, n),
                       "source": rng.choice(["coco2017", "OpenImages", "ClockMovies"], n),
                       "license": rng.choice(["CC BY 4.0", "Unknown/Custom"], n, p=[0.8, 0.2]), "license_url": "u", "source_url": "s", "image_name": [f"i{i}.jpg" for i in range(n)]})
    df["image"] = [{"bytes": png_bytes(i), "path": None} for i in range(n)]
    a, b = P.select_rows(df), P.select_rows(df.sample(frac=1, random_state=3))
    check("500 rows, all CC BY 4.0", len(a) == 500 and (a.license == "CC BY 4.0").all())
    check("selection is deterministic and independent of input row order", list(a.sample_id) == list(b.sample_id))
    check("no duplicates", a.sample_id.is_unique)
    try:
        P.select_rows(df.head(100))
        check("too few CC BY rows -> error (rule not loosened)", False)
    except ValueError:
        check("too few CC BY rows -> error (rule not loosened)", True)
    with tempfile.TemporaryDirectory() as tmp:
        k = P.write_set(a.head(12), tmp)
        d = pd.read_csv(os.path.join(tmp, "data.csv")); m = pd.read_csv(os.path.join(tmp, "manifest_provenance.csv"))
        check("12 PNGs + data.csv + manifest", k == 12 and len(d) == 12 and len(m) == 12 and all(os.path.exists(os.path.join(tmp, f)) for f in d.filename))
        check("hour stored 1-12, original kept in the manifest", d.hour.between(1, 12).all() and
              all((int(o) % 12 or 12) == int(h) for o, h in zip(m.hour_original, d.hour)))
        check("data.csv has the columns eval_behavior/describe_check read", {"filename", "hour", "minute"} <= set(d.columns))
    return ok


if __name__ == "__main__":
    r = main()
    print("\nALL PASS" if r else "\nSOME FAILED")
    sys.exit(0 if r else 1)
