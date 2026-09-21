"""
compare_models.py -- build a single cross-model comparison table + figure
from each model's own results, so the finding can be checked against
"is this specific to Qwen2.5-VL-3B" (see run_model.sh, which produces the
inputs this script reads).

Every source file is optional -- a model missing one step (e.g. Experiment A
hasn't been run yet) still gets a row, with NaN for whatever's missing, and a
printed note saying exactly what's absent. Never crashes on partial results;
that would defeat the point of checking multiple models' progress at once.

Each script keeps its OWN established base directory (unchanged from before
multi-model support existed), just partitioned by model short_name beneath
it (see adapters.output_dir_for) -- so this reads from THREE different
roots, not one unified "outputs/":
  - results/<short>/results.csv                    (eval_behavior.py)    -> minute/hour/exact accuracy
  - results/<short>/describe.csv                   (describe_check.py)   -> self-consistency
  - results/<short>/text_only.csv                  (text_only_check.py)  -> text-only (hand->time) accuracy
  - probe_output/probe_results/<short>/summary_table.csv,
    per_layer_results.csv                          (probe.py)            -> best-layer R2, final-layer R2
  - intervene_output/<short>/experiment_a_per_layer_transfer.csv (intervene.py) -> readout curve vs. relative depth

Usage:
    python compare_models.py
    python compare_models.py --models qwen2.5-vl-3b,qwen2.5-vl-7b,gemma-3-4b-it
    (default: every model_short_name with a subfolder under --results_dir)
"""
import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"                        # eval_behavior.py / describe_check.py / text_only_check.py
INTERVENE_DIR = "intervene_output"              # intervene.py's Experiment A output
PROBE_RESULTS_DIR = "probe_output/probe_results"   # probe.py
PROBE_REPRESENTATION = "hidden_meanpool"   # matches Experiment B's steering representation
PROBE_HAND = "minute"


def minute_number(minute):
    """Same convention as probe.py/describe_check.py: which of the 12 face
    numbers a minute value is nearest to."""
    n = round(minute / 5) % 12
    return 12 if n == 0 else n


def discover_models(results_dir):
    if not os.path.isdir(results_dir):
        return []
    return sorted(d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)))


def behavior_metrics(results_dir, short_name, notes):
    path = os.path.join(results_dir, short_name, "results.csv")
    if not os.path.exists(path):
        notes.append(f"{short_name}: no results.csv (run eval_behavior.py) -- minute/hour/exact accuracy missing")
        return {}
    df = pd.read_csv(path)
    ok = df["parse_success"].astype(bool)
    return {
        "n_behavior": len(df),
        "minute_accuracy": float((ok & (df["pred_minute"] == df["true_minute"])).mean()),
        "hour_accuracy": float((ok & (df["pred_hour"] == df["true_hour"])).mean()),
        "exact_accuracy": float((ok & (df["pred_minute"] == df["true_minute"]) &
                                  (df["pred_hour"] == df["true_hour"])).mean()),
        "parse_rate": float(ok.mean()),
    }


def self_consistency_metric(results_dir, short_name, notes):
    """Does the model's STATED description of the minute hand's number agree
    with the number nearest its OWN stated final-answer minute -- independent
    of whether either is factually correct. This is what "self-consistency"
    means here: internal agreement between two things the model itself said,
    not agreement with ground truth (that's minute_accuracy above)."""
    path = os.path.join(results_dir, short_name, "describe.csv")
    if not os.path.exists(path):
        notes.append(f"{short_name}: no describe.csv (run describe_check.py) -- self-consistency missing")
        return {}
    df = pd.read_csv(path)
    usable = df["said_minute_number"].notna() & df["pred_minute"].notna()
    if usable.sum() == 0:
        return {"self_consistency": float("nan"), "n_self_consistency": 0}
    pred_number = df.loc[usable, "pred_minute"].apply(minute_number)
    agree = (df.loc[usable, "said_minute_number"] == pred_number)
    return {"self_consistency": float(agree.mean()), "n_self_consistency": int(usable.sum())}


def text_only_metric(results_dir, short_name, notes):
    path = os.path.join(results_dir, short_name, "text_only.csv")
    if not os.path.exists(path):
        notes.append(f"{short_name}: no text_only.csv (run text_only_check.py) -- text-only accuracy missing")
        return {}
    df = pd.read_csv(path)
    hand_to_time = df[df["case"] == "hand_to_time"]
    arithmetic = df[df["case"] == "arithmetic"]
    return {
        "text_only_accuracy": float(hand_to_time["correct"].mean()) if len(hand_to_time) else float("nan"),
        "text_only_arithmetic_accuracy": float(arithmetic["correct"].mean()) if len(arithmetic) else float("nan"),
    }


def probe_r2_metrics(probe_results_dir, short_name, notes):
    summary_path = os.path.join(probe_results_dir, short_name, "summary_table.csv")
    per_layer_path = os.path.join(probe_results_dir, short_name, "per_layer_results.csv")
    if not os.path.exists(summary_path) or not os.path.exists(per_layer_path):
        notes.append(f"{short_name}: no probe results under '{probe_results_dir}/{short_name}/' "
                     "(run probe.py --stage all) -- probe R2 missing")
        return {}
    summary = pd.read_csv(summary_path)
    row = summary[(summary["representation"] == PROBE_REPRESENTATION) & (summary["hand"] == PROBE_HAND)]
    best_r2 = float(row["r2_all"].iloc[0]) if len(row) else float("nan")
    best_layer = int(row["layer"].iloc[0]) if len(row) and pd.notna(row["layer"].iloc[0]) else None

    per_layer = pd.read_csv(per_layer_path)
    sub = per_layer[(per_layer["representation"] == PROBE_REPRESENTATION) & (per_layer["hand"] == PROBE_HAND) &
                     (per_layer["split"] == "all") & (per_layer["status"] != "skipped")]
    final_r2 = float("nan")
    if len(sub):
        final_layer = int(sub["layer"].max())
        final_row = sub[sub["layer"] == final_layer]
        final_r2 = float(final_row["r2"].iloc[0])
    return {"best_probe_r2": best_r2, "best_probe_layer": best_layer, "final_layer_r2": final_r2}


def readout_curve(intervene_dir, short_name, notes):
    """The Experiment A per-layer transfer curve (image_tokens, condition
    real_b), keyed by RELATIVE depth -- what plot_readout_curves compares
    across models. Returns a DataFrame (empty if missing) with columns
    layer, relative_depth, to_b_rate, stay_a_rate, other_rate, n_usable."""
    path = os.path.join(intervene_dir, short_name, "experiment_a_per_layer_transfer.csv")
    if not os.path.exists(path):
        notes.append(f"{short_name}: no experiment_a_per_layer_transfer.csv (run intervene.py --experiment a) "
                     "-- readout curve missing")
        return pd.DataFrame()
    df = pd.read_csv(path)
    sub = df[(df["condition"] == "real_b") & (df["position_set"] == "image_tokens") & (df["layer"] >= 0)]
    if "relative_depth" not in sub.columns:
        notes.append(f"{short_name}: experiment_a_per_layer_transfer.csv predates saving relative_depth -- "
                     "re-run the sweep to include this model in the cross-model readout-curve plot")
        return pd.DataFrame()
    return sub[["layer", "relative_depth", "to_b_rate", "stay_a_rate", "other_rate", "n_usable"]].sort_values("layer")


def build_comparison_table(models, results_dir, intervene_dir, probe_results_dir):
    notes = []
    rows = []
    curves = {}
    for short_name in models:
        row = {"model": short_name}
        row.update(behavior_metrics(results_dir, short_name, notes))
        row.update(self_consistency_metric(results_dir, short_name, notes))
        row.update(text_only_metric(results_dir, short_name, notes))
        row.update(probe_r2_metrics(probe_results_dir, short_name, notes))
        rows.append(row)
        curve = readout_curve(intervene_dir, short_name, notes)
        if len(curve):
            curves[short_name] = curve

    table = pd.DataFrame(rows)
    return table, curves, notes


def print_comparison_table(table):
    cols = ["model", "minute_accuracy", "hour_accuracy", "exact_accuracy", "self_consistency",
            "text_only_accuracy", "best_probe_r2", "best_probe_layer", "final_layer_r2"]
    cols = [c for c in cols if c in table.columns]
    pct_cols = {"minute_accuracy", "hour_accuracy", "exact_accuracy", "self_consistency", "text_only_accuracy"}
    lines = ["=== CROSS-MODEL COMPARISON ===", ""]
    header = "".join(f"{c:<22}" if c == "model" else f" {c:>19}" for c in cols)
    lines.append(header)
    for _, r in table.iterrows():
        parts = []
        for c in cols:
            v = r.get(c, float("nan"))
            if c == "model":
                parts.append(f"{v:<22}")
            elif pd.isna(v):
                parts.append(f" {'--':>19}")
            elif c == "best_probe_layer":
                parts.append(f" {int(v):>19}")
            elif c in pct_cols:
                parts.append(f" {v:>18.1%}")
            else:
                parts.append(f" {v:>18.3f}")
        lines.append("".join(parts))
    lines.append("")
    lines.append("minute/hour/exact_accuracy: eval_behavior.py, vs. ground truth. self_consistency: does the")
    lines.append("model's STATED description agree with its OWN stated final answer (describe_check.py),")
    lines.append("independent of correctness. text_only_accuracy: hand-position-in-words -> time, no image")
    lines.append("(text_only_check.py). best/final_layer_r2: hidden_meanpool/minute probe R^2 (probe.py).")
    text = "\n".join(lines)
    print("\n" + text)
    return text


def plot_readout_curves(curves, out_path):
    """The Experiment A readout curve (image-token to_B transfer rate) for
    every model, all on ONE figure, x-axis = RELATIVE depth (0=embeddings,
    1=final layer) so models with different layer counts are comparable --
    this is the whole point of item 3's relative-depth reporting."""
    if not curves:
        print("No models have a readout curve yet (run intervene.py --experiment a for at least one) -- "
              "skipping plot_readout_curves.")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for short_name, curve in curves.items():
        ax.plot(curve["relative_depth"], curve["to_b_rate"], marker="o", markersize=3, label=short_name)
    ax.axhline(0, color="lightgray", linewidth=1)
    ax.set_xlabel("relative depth (0 = embeddings, 1 = final decoder layer)")
    ax.set_ylabel("to_B transfer rate (image_tokens, real_b)")
    ax.set_title("Experiment A readout curve across models, by relative depth")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Readout-curve comparison plot saved to '{out_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Build a cross-model comparison table + figure.")
    parser.add_argument("--models", type=str, default=None,
                         help="comma-separated model short_names (default: every subfolder under --results_dir)")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                         help="base dir for eval_behavior.py/describe_check.py/text_only_check.py output")
    parser.add_argument("--intervene_dir", type=str, default=INTERVENE_DIR,
                         help="base dir for intervene.py's Experiment A output")
    parser.add_argument("--probe_results_dir", type=str, default=PROBE_RESULTS_DIR)
    parser.add_argument("--out_dir", type=str, default="outputs/compare",
                         help="where to write comparison_table.csv and readout_curves.png")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else discover_models(args.results_dir)
    if not models:
        raise SystemExit(f"No models found under '{args.results_dir}/' -- run run_model.sh for at least one "
                          "model first, or pass --models explicitly.")
    print(f"Comparing {len(models)} model(s): {models}")

    table, curves, notes = build_comparison_table(models, args.results_dir, args.intervene_dir,
                                                   args.probe_results_dir)
    text = print_comparison_table(table)

    if notes:
        print("\nMissing data (these don't stop the comparison -- see the table/plot for what's usable):")
        for n in notes:
            print(f"  - {n}")

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "comparison_table.csv"), index=False)
    with open(os.path.join(args.out_dir, "comparison_table.txt"), "w") as f:
        f.write(text + "\n")
        if notes:
            f.write("\nMissing data:\n" + "\n".join(f"  - {n}" for n in notes) + "\n")
    plot_readout_curves(curves, os.path.join(args.out_dir, "readout_curves.png"))
    print(f"\nComparison table + plot written to '{args.out_dir}/'.")


if __name__ == "__main__":
    main()
