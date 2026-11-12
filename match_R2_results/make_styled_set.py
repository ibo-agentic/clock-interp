"""make_styled_set.py -- render a styled copy of an existing clock dataset (experiment R, PREREG_MECHANISM.md ninth entry).

The styled set is rendered from the SOURCE set's data.csv (hour, minute lists), keeps every filename, and copies data.csv unchanged, so each styled image is
paired with its standard-style original. Only the style parameters of clocks.generate_clock_image change (clocks.STYLES).

  python make_styled_set.py --src data_positions --style s2_ticks --out data_positions_s2_ticks
  python make_styled_set.py --src data --style s1_thin --out data_s1_thin
  python make_styled_set.py --src data,data_positions --all s1_thin,s2_ticks      # -> data_s1_thin, data_positions_s1_thin, data_s2_ticks, ...
  python make_styled_set.py --src data_positions --check            # gate: the DEFAULT style re-rendered from data.csv must equal the existing images outside the numeral ring (glyph rasterisation inside the ring is reported)
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

from clocks import STYLES, generate_clock_image, style_kwargs


def render(src, style, out):
    df = pd.read_csv(os.path.join(src, "data.csv"))
    os.makedirs(out, exist_ok=True)
    kw = style_kwargs(style)
    for r in df.itertuples():
        generate_clock_image(int(r.hour), int(r.minute), os.path.join(out, r.filename), **kw)
    shutil.copyfile(os.path.join(src, "data.csv"), os.path.join(out, "data.csv"))
    return df


NUMERAL_RING = (85, 140)      # pixel radius (512 px images, centre ~255.5) of the numeral glyphs; text rasterisation differs between matplotlib versions


def check_default(src, max_images=None, tmp=None):
    """Gate: re-render the DEFAULT style from data.csv and compare with the existing images.
    Strict part: every pixel OUTSIDE the numeral ring must be identical (rim, ticks, outer face, outside).
    Reported part: pixels inside the ring that differ (numeral glyph rasterisation depends on the matplotlib version; measured locally with matplotlib 3.11.2
    against the existing data_positions images: about 1.9k of 262k pixels per image, all inside the ring).
    Returns dict(n, n_exact, n_outside_diff_images, mean_ring_diff_px, max_ring_diff_px, examples)."""
    import tempfile
    df = pd.read_csv(os.path.join(src, "data.csv"))
    if max_images:
        df = df.head(max_images)
    tmp = tmp or tempfile.mkdtemp(prefix="style_check_")
    yy, xx = np.mgrid[0:512, 0:512]
    rad = np.hypot(yy - 255.5, xx - 255.5)
    ring = (rad >= NUMERAL_RING[0]) & (rad <= NUMERAL_RING[1])
    n_exact, outside_bad, ring_px, ex = 0, 0, [], []
    for r in df.itertuples():
        p = os.path.join(tmp, r.filename)
        generate_clock_image(int(r.hour), int(r.minute), p)               # no style arguments = the original call
        a = np.asarray(Image.open(os.path.join(src, r.filename)).convert("RGB")).astype(int)
        b = np.asarray(Image.open(p).convert("RGB")).astype(int)
        if a.shape != b.shape:
            outside_bad += 1; ex.append(r.filename); continue
        d = np.abs(a - b).sum(2) > 0
        n_exact += int(not d.any())
        ring_px.append(int((d & ring).sum()))
        if (d & ~ring).any():
            outside_bad += 1; ex.append(r.filename)
    return {"n": len(df), "n_exact": n_exact, "n_outside_diff_images": outside_bad, "mean_ring_diff_px": float(np.mean(ring_px)) if ring_px else None,
            "max_ring_diff_px": max(ring_px) if ring_px else None, "examples": ex[:5]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source folder with data.csv and the standard-style images")
    ap.add_argument("--style", default=None, choices=sorted(STYLES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="gate: default-style re-render equals the existing images pixel-for-pixel")
    ap.add_argument("--all", default=None, help="comma list of styles: render every style for every --src (comma list) into <src>_<style>")
    ap.add_argument("--max_images", type=int, default=None)
    a = ap.parse_args()
    if a.check:
        bad = 0
        for src in a.src.split(","):
            res = check_default(src, a.max_images)
            print(f"default-style re-render vs {src}: {res}", flush=True)
            bad += res["n_outside_diff_images"]
        print("GATE:", "PASS (identical outside the numeral ring)" if bad == 0 else "FAIL (differences outside the numeral ring)")
        sys.exit(0 if bad == 0 else 1)
    if a.all:                                       # render every (style, src) pair into <src>_<style>
        for style in a.all.split(","):
            for src in a.src.split(","):
                df = render(src, style, f"{src}_{style}")
                print(f"rendered {len(df)} images in style '{style}' ({style_kwargs(style)}) -> {src}_{style}", flush=True)
        sys.exit(0)
    if not (a.style and a.out):
        ap.error("--style and --out are required unless --check or --all")
    df = render(a.src, a.style, a.out)
    print(f"rendered {len(df)} images in style '{a.style}' ({style_kwargs(a.style)}) -> {a.out}")
