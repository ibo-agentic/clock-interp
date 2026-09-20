"""
analyze.py -- STEP 1 behavior check: analyze the VLM's clock-reading results.

Reads the results CSV produced by eval_behavior.py and reports:
  - exact accuracy (hour AND minute both correct)
  - hour accuracy and minute accuracy separately
  - hand-swap rate (see `is_hand_swap` below for the exact heuristic), reported
    both including and excluding "overlap" clocks -- see `hands_overlap` --
    where the hour and minute hands sit close enough together (e.g. 12:00,
    1:05, 2:11) that a swap and a correct answer look almost identical
  - accuracy within +/- 5 minutes (circular distance on a 12h face)
  - accuracy broken down by minute-bucket (near :00 vs elsewhere) and by hour
  - example images of each error type, saved as PNG montages

Usage as a script:
    python analyze.py --results_csv results/results.csv --images_dir data \
        --out_dir analysis_output

Usage as a library (e.g. from the notebook):
    from analyze import run_analysis
    summary = run_analysis(results_csv="results/results.csv", images_dir="data")
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from clocks import hand_angles

# Hands within this many degrees of each other are considered "overlapping":
# at that point a swapped reading and a correct reading point at nearly the
# same place on the face, so a "hand swap" there is much less meaningful
# than one where the hands are clearly apart.
OVERLAP_THRESHOLD_DEG = 15


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def circular_minute_diff(true_hour, true_minute, pred_hour, pred_minute):
    """Absolute difference in minutes between two times on a 12-hour clock
    face, accounting for wraparound (e.g. 11:58 vs 12:02 is 4 minutes apart,
    not 718). Hours are normalized so 12 == 0.
    """
    t_true = (true_hour % 12) * 60 + true_minute
    t_pred = (pred_hour % 12) * 60 + pred_minute
    diff = abs(t_true - t_pred)
    return min(diff, 720 - diff)


def is_hand_swap(true_hour, true_minute, pred_hour, pred_minute):
    """Heuristic: did the model read the hour hand as the minute hand and
    vice versa?

    On an analog face, the hour hand's position also lines up with a minute
    tick (hour H sits at the same angle as minute H*5), and the minute
    hand's position lines up with an hour number (minute M sits at the same
    angle as hour round(M/5)). If the model's answer matches the time you'd
    get by reading the hands backwards this way, we call it a hand swap.

    This is an approximation, not a perfect ground truth -- e.g. for times
    near quarter-hours a swapped reading can coincide with other error
    types. It's intended to give a rough rate, not a guarantee.
    """
    swapped_hour = round(true_minute / 5) % 12
    if swapped_hour == 0:
        swapped_hour = 12
    swapped_minute = (true_hour % 12) * 5

    exact = (pred_hour == true_hour) and (pred_minute == true_minute)
    return (not exact) and (pred_hour == swapped_hour) and (pred_minute == swapped_minute)


def hands_overlap(hour, minute, threshold_deg=OVERLAP_THRESHOLD_DEG):
    """True if the hour and minute hands point within `threshold_deg` of each
    other on the true clock face (e.g. near 12:00, 1:05, 2:11, ...).

    Uses the exact same angle formulas as the renderer (`clocks.hand_angles`)
    so this lines up with what the image actually looks like.
    """
    hour_angle, minute_angle = hand_angles(hour, minute)
    diff = abs(hour_angle - minute_angle) % 360
    diff = min(diff, 360 - diff)
    return diff <= threshold_deg


def minute_bucket(minute):
    """Bucket a minute value into a 5-minute-wide label, e.g. 'near :00'."""
    bucket_start = (minute // 5) * 5
    if bucket_start == 0:
        return "near :00"
    return f":{bucket_start:02d}-:{bucket_start + 4:02d}"


def categorize_row(row):
    """Assign a single error-type label to a result row, in priority order."""
    if not row["parse_success"]:
        return "parse_failure"

    th, tm, ph, pm = row["true_hour"], row["true_minute"], row["pred_hour"], row["pred_minute"]
    ph, pm = int(ph), int(pm)

    if ph == th and pm == tm:
        return "correct"
    if is_hand_swap(th, tm, ph, pm):
        return "hand_swap"
    if ph == th:
        return "hour_only_correct"
    if pm == tm:
        return "minute_only_correct"
    if circular_minute_diff(th, tm, ph, pm) <= 5:
        return "within_5min"
    return "other_error"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(results_csv="results/results.csv", images_dir="data",
                  out_dir="analysis_output", examples_per_category=6,
                  overlap_threshold_deg=OVERLAP_THRESHOLD_DEG):
    df = pd.read_csv(results_csv)
    os.makedirs(out_dir, exist_ok=True)

    n_total = len(df)
    parsed = df[df["parse_success"]].copy()
    n_parsed = len(parsed)

    # Types coming back from CSV are floats if there were any NaNs; make the
    # parsed subset's prediction columns plain ints for comparisons.
    parsed["pred_hour"] = parsed["pred_hour"].astype(int)
    parsed["pred_minute"] = parsed["pred_minute"].astype(int)

    parsed["error_type"] = parsed.apply(categorize_row, axis=1)
    parsed["minute_diff"] = parsed.apply(
        lambda r: circular_minute_diff(r["true_hour"], r["true_minute"], r["pred_hour"], r["pred_minute"]),
        axis=1,
    )
    parsed["is_overlap"] = parsed.apply(
        lambda r: hands_overlap(r["true_hour"], r["true_minute"], overlap_threshold_deg), axis=1
    )

    exact_correct = (parsed["error_type"] == "correct").sum()
    hour_correct = (parsed["pred_hour"] == parsed["true_hour"]).sum()
    minute_correct = (parsed["pred_minute"] == parsed["true_minute"]).sum()
    hand_swaps = (parsed["error_type"] == "hand_swap").sum()
    within_5min = (parsed["minute_diff"] <= 5).sum()

    # Hand-swap rate is easy to inflate near-overlap times (e.g. 1:05), where
    # a swapped reading and a correct one point almost the same place on the
    # face -- so report it both ways: over every parsed clock, and over only
    # the clocks where the hands are clearly apart.
    non_overlap = parsed[~parsed["is_overlap"]]
    n_overlap = int(parsed["is_overlap"].sum())
    n_non_overlap = len(non_overlap)
    hand_swaps_non_overlap = (non_overlap["error_type"] == "hand_swap").sum()

    summary = {
        "n_total_images": n_total,
        "n_parsed": n_parsed,
        "parse_failure_rate": 1 - n_parsed / n_total if n_total else float("nan"),
        "exact_accuracy": exact_correct / n_parsed if n_parsed else float("nan"),
        "hour_accuracy": hour_correct / n_parsed if n_parsed else float("nan"),
        "minute_accuracy": minute_correct / n_parsed if n_parsed else float("nan"),
        "within_5min_accuracy": within_5min / n_parsed if n_parsed else float("nan"),
        "n_overlap_clocks": n_overlap,
        "overlap_rate": n_overlap / n_parsed if n_parsed else float("nan"),
        "hand_swap_rate_including_overlap": hand_swaps / n_parsed if n_parsed else float("nan"),
        "hand_swap_rate_excluding_overlap": (
            hand_swaps_non_overlap / n_non_overlap if n_non_overlap else float("nan")
        ),
    }

    # --- breakdown by hour ---
    # Named aggregation (rather than groupby().apply()) works the same across
    # older and newer pandas versions, which matters since Kaggle's pinned
    # pandas version can lag behind.
    by_hour = parsed.groupby("true_hour").agg(
        n=("error_type", "size"),
        exact_accuracy=("error_type", lambda s: (s == "correct").mean()),
        hand_swap_rate=("error_type", lambda s: (s == "hand_swap").mean()),
    ).reset_index()
    by_hour.to_csv(os.path.join(out_dir, "by_hour.csv"), index=False)

    # --- breakdown by minute bucket (near :00 vs elsewhere) ---
    parsed["minute_bucket"] = parsed["true_minute"].apply(minute_bucket)
    by_minute = parsed.groupby("minute_bucket").agg(
        n=("error_type", "size"),
        exact_accuracy=("error_type", lambda s: (s == "correct").mean()),
        hand_swap_rate=("error_type", lambda s: (s == "hand_swap").mean()),
    ).reset_index()
    by_minute.to_csv(os.path.join(out_dir, "by_minute_bucket.csv"), index=False)

    # --- print summary ---
    print("=== Clock-reading behavior summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1%}" if "rate" in k or "accuracy" in k else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # --- example images per error type ---
    examples_dir = os.path.join(out_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    error_categories = ["correct", "hand_swap", "hour_only_correct",
                         "minute_only_correct", "within_5min", "other_error"]
    for category in error_categories:
        subset = parsed[parsed["error_type"] == category]
        if len(subset) == 0:
            continue
        save_example_grid(subset.head(examples_per_category), images_dir,
                           os.path.join(examples_dir, f"{category}.png"), category)

    # Parse failures don't have a usable prediction to caption, but are still
    # worth a quick look.
    parse_fail_subset = df[~df["parse_success"]].head(examples_per_category)
    if len(parse_fail_subset) > 0:
        save_example_grid(parse_fail_subset, images_dir,
                           os.path.join(examples_dir, "parse_failure.png"), "parse_failure")

    print(f"\nBreakdowns and example images written to '{out_dir}/'.")
    return summary


def save_example_grid(rows, images_dir, save_path, category):
    """Save a small montage of example clock images with true/predicted
    time captions, for quick visual inspection of one error category."""
    n = len(rows)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows))
    axes = [axes] if n == 1 else axes.flatten()

    for ax, (_, row) in zip(axes, rows.iterrows()):
        img_path = os.path.join(images_dir, row["filename"])
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")

        true_str = f"{int(row['true_hour'])}:{int(row['true_minute']):02d}"
        if row["parse_success"]:
            pred_str = f"{int(row['pred_hour'])}:{int(row['pred_minute']):02d}"
        else:
            pred_str = f"(unparsed: {row['raw_answer']!r})"
        ax.set_title(f"true={true_str}  pred={pred_str}", fontsize=10)

    # hide any unused axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Error type: {category}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze VLM clock-reading results.")
    parser.add_argument("--results_csv", type=str, default="results/results.csv")
    parser.add_argument("--images_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="analysis_output")
    parser.add_argument("--examples_per_category", type=int, default=6)
    parser.add_argument("--overlap_threshold_deg", type=float, default=OVERLAP_THRESHOLD_DEG,
                         help="hands within this many degrees of each other count as 'overlapping'")
    args = parser.parse_args()

    run_analysis(
        results_csv=args.results_csv,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        examples_per_category=args.examples_per_category,
        overlap_threshold_deg=args.overlap_threshold_deg,
    )
