"""
clocks.py -- Draw synthetic analog clock images with matplotlib.

This is STEP 1 of the clock-reading behavior check: we need a controllable
source of clock images with known ground truth (the true hour/minute) so we
can later check whether a vision-language model reads them correctly.

Usage as a script:
    python clocks.py --n 500 --out_dir data --seed 42

Usage as a library (e.g. from the notebook):
    from clocks import generate_dataset
    df = generate_dataset(n=500, out_dir="data", seed=42)
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNG files
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polar_to_xy(angle_deg_from_12, length):
    """Convert an angle measured clockwise from the 12 o'clock position
    (i.e. how a clock hand angle is normally described) into (x, y)
    coordinates on a unit circle, with (0, 0) at the clock's center and
    (0, 1) being the 12 o'clock position.
    """
    # Standard math angles go counter-clockwise from the +x axis, so we
    # convert: clockwise-from-top angle theta -> math angle (90 - theta).
    theta = math.radians(90 - angle_deg_from_12)
    x = length * math.cos(theta)
    y = length * math.sin(theta)
    return x, y


def hand_angles(hour, minute):
    """Return (hour_hand_angle_deg, minute_hand_angle_deg), both measured
    clockwise from 12 o'clock.

    Public (not prefixed with `_`) so analyze.py can reuse the exact same
    geometry when checking whether the two hands nearly overlap.
    """
    # Minute hand: 360 degrees / 60 minutes = 6 degrees per minute.
    minute_angle = minute * 6.0

    # Hour hand: 360 degrees / 12 hours = 30 degrees per hour, PLUS it
    # creeps forward smoothly as minutes pass (0.5 degrees per minute).
    # Using hour % 12 so that hour=12 behaves like hour=0 (points to top).
    hour_angle = (hour % 12) * 30.0 + minute * 0.5

    return hour_angle, minute_angle


# ---------------------------------------------------------------------------
# Drawing a single clock
# ---------------------------------------------------------------------------

def generate_clock_image(
    hour,
    minute,
    save_path,
    hour_hand_length=0.50,
    minute_hand_length=0.85,
    hour_hand_thickness=9,
    minute_hand_thickness=4,
    show_numbers=True,
    show_ticks=True,
    face_color="white",
    hand_color="black",
    edge_color="black",
    image_size=512,
    dpi=100,
):
    """Draw one analog clock face showing `hour`:`minute` and save it as a PNG.

    Default style (per spec): clear white face, black hands, hour hand
    shorter AND thicker than the minute hand.

    Parameters
    ----------
    hour : int (1-12)
    minute : int (0-59)
    save_path : str, where to write the PNG
    hour_hand_length, minute_hand_length : float, hand length as a fraction
        of the clock face radius (radius = 1.0)
    hour_hand_thickness, minute_hand_thickness : float, matplotlib linewidth
    show_numbers : bool, draw the 1-12 numerals on the face
    show_ticks : bool, draw the 60 minute tick marks (with longer/thicker
        ticks at each hour)
    face_color, hand_color, edge_color : matplotlib color strings
    image_size : int, output image size in pixels (square)
    dpi : int, resolution used to convert the figure to pixels
    """
    figsize = image_size / dpi
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)

    # --- clock face ---
    face = plt.Circle((0, 0), 1.0, facecolor=face_color, edgecolor=edge_color,
                       linewidth=2.5, zorder=1)
    ax.add_patch(face)

    # --- tick marks (60 of them; every 5th tick, i.e. each hour, is longer/thicker) ---
    if show_ticks:
        for m in range(60):
            angle = m * 6.0
            is_hour_tick = (m % 5 == 0)
            outer = 0.95
            inner = 0.80 if is_hour_tick else 0.88
            lw = 2.5 if is_hour_tick else 1.0
            x1, y1 = _polar_to_xy(angle, inner)
            x2, y2 = _polar_to_xy(angle, outer)
            ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=lw,
                     solid_capstyle="round", zorder=2)

    # --- numbers 1-12 ---
    if show_numbers:
        for h in range(1, 13):
            angle = h * 30.0
            x, y = _polar_to_xy(angle, 0.66)
            ax.text(x, y, str(h), ha="center", va="center",
                    fontsize=17, fontweight="bold", color=edge_color, zorder=2)

    # --- hands ---
    hour_angle, minute_angle = hand_angles(hour, minute)
    hx, hy = _polar_to_xy(hour_angle, hour_hand_length)
    mx, my = _polar_to_xy(minute_angle, minute_hand_length)

    # Hour hand: shorter and thicker (drawn first, minute hand on top)
    ax.plot([0, hx], [0, hy], color=hand_color, linewidth=hour_hand_thickness,
             solid_capstyle="round", zorder=3)
    # Minute hand: longer and thinner
    ax.plot([0, mx], [0, my], color=hand_color, linewidth=minute_hand_thickness,
             solid_capstyle="round", zorder=4)

    # center pivot dot
    ax.add_patch(plt.Circle((0, 0), 0.03, facecolor=hand_color, zorder=5))

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _draw_and_record(hour, minute, filename, out_dir, style):
    """Draw one clock into `out_dir`, and return the metadata row for it.
    Shared by `generate_dataset` and `generate_balanced_dataset` so both
    write the exact same CSV schema."""
    generate_clock_image(hour, minute, os.path.join(out_dir, filename), **style)
    row = {"filename": filename, "hour": hour, "minute": minute}
    row.update(style)
    return row


def generate_dataset(
    n=500,
    out_dir="data",
    seed=42,
    hour_hand_length=0.50,
    minute_hand_length=0.85,
    hour_hand_thickness=9,
    minute_hand_thickness=4,
    show_numbers=True,
    show_ticks=True,
    face_color="white",
    hand_color="black",
    image_size=512,
):
    """Generate `n` clock images with random times (default style) and a CSV
    logging the true time + all rendering settings for each image.

    Returns the metadata DataFrame (also written to <out_dir>/data.csv).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    style = dict(
        hour_hand_length=hour_hand_length, minute_hand_length=minute_hand_length,
        hour_hand_thickness=hour_hand_thickness, minute_hand_thickness=minute_hand_thickness,
        show_numbers=show_numbers, show_ticks=show_ticks,
        face_color=face_color, hand_color=hand_color, image_size=image_size,
    )

    rows = []
    for i in range(n):
        hour = int(rng.randint(1, 13))     # 1-12 inclusive
        minute = int(rng.randint(0, 60))   # 0-59 inclusive
        rows.append(_draw_and_record(hour, minute, f"clock_{i:04d}.png", out_dir, style))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {n} clock images to '{out_dir}/' and metadata to '{csv_path}'")
    return df


def generate_balanced_dataset(
    images_per_minute=8,
    out_dir="data_balanced",
    seed=123,
    hour_hand_length=0.50,
    minute_hand_length=0.85,
    hour_hand_thickness=9,
    minute_hand_thickness=4,
    show_numbers=True,
    show_ticks=True,
    face_color="white",
    hand_color="black",
    image_size=512,
):
    """Generate a dataset with an EQUAL number of images at every minute
    value (0-59), each with a random hour. Used for probing: a regression
    probe needs even coverage of the full 0-360 degree angle range, which a
    plain random draw doesn't guarantee at small sample sizes.

    With the default `images_per_minute=8`, this makes 60 * 8 = 480 images.

    Returns the metadata DataFrame (also written to <out_dir>/data.csv).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    style = dict(
        hour_hand_length=hour_hand_length, minute_hand_length=minute_hand_length,
        hour_hand_thickness=hour_hand_thickness, minute_hand_thickness=minute_hand_thickness,
        show_numbers=show_numbers, show_ticks=show_ticks,
        face_color=face_color, hand_color=hand_color, image_size=image_size,
    )

    rows = []
    i = 0
    for minute in range(60):
        for _ in range(images_per_minute):
            hour = int(rng.randint(1, 13))  # random hour, so it isn't collinear with minute
            rows.append(_draw_and_record(hour, minute, f"clock_{i:04d}.png", out_dir, style))
            i += 1

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    n = len(df)
    print(f"Wrote {n} clock images ({images_per_minute} per minute value) to "
          f"'{out_dir}/' and metadata to '{csv_path}'")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic analog clock images.")
    parser.add_argument("--n", type=int, default=500, help="number of clocks to generate")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="output directory (default: 'data', or 'data_balanced' with --balanced)")
    parser.add_argument("--seed", type=int, default=None,
                         help="random seed (default: 42, or 123 with --balanced)")
    parser.add_argument("--balanced", action="store_true",
                         help="generate the balanced dataset (equal images per minute value) used for probing")
    parser.add_argument("--images_per_minute", type=int, default=8,
                         help="with --balanced: images per minute value (default 8 -> 480 images)")
    args = parser.parse_args()

    if args.balanced:
        generate_balanced_dataset(
            images_per_minute=args.images_per_minute,
            out_dir=args.out_dir or "data_balanced",
            seed=args.seed if args.seed is not None else 123,
        )
    else:
        generate_dataset(
            n=args.n,
            out_dir=args.out_dir or "data",
            seed=args.seed if args.seed is not None else 42,
        )
