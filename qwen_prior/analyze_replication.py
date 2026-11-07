"""
analyze_replication.py -- pair-level statistical comparison of Experiment A's
readout-window transfer rate between two (or more) models, for checking a
finding on FRESH pairs with the depth window fixed in advance (see
intervene.py's --exclude_pairs_from / --seed for how to actually generate
non-overlapping replication data).

WHAT THIS COMPUTES, precisely (read this before trusting the numbers):
  1. Loads each model's experiment_a_trials.csv and adds transfer columns
     via intervene.py's `compute_transfer_columns` (same logic
     summarize_per_layer_transfer_a uses).
  2. Restricts to condition=='real_b', position_set=='image_tokens' (image
     tokens are where this project's own prior finding says the minute is
     causally read out -- see intervene.py's module docstring), and layers
     with relative_depth in [--rel_depth_min, --rel_depth_max].
  3. Restricts further to pairs where THAT MODEL's OWN baseline minutes
     differ for A and the target (minute_baselines_differ == 1 -- the SAME
     restriction summarize_per_layer_transfer_a already uses, applied
     independently per model, NOT a cross-model condition). If you meant
     something else by "the two models' own baseline minutes differ",
     this is the one assumption in this script most likely to need
     correcting -- everything downstream depends on it.
  4. Within that restricted set, computes ONE number per (pair): the mean
     of minute_transfer ("to_B") across however many layers fall inside the
     window for that pair (NaN layers -- unparseable answers -- are
     skipped, not counted as 0; a pair with ALL layers NaN in the window is
     dropped entirely, since it contributes no signal).
  5. Compares the two models' pair-level means with a two-sided PERMUTATION
     test (pools both models' pair-level values, reshuffles the group
     labels many times, and asks how often a difference this large or
     larger appears by chance) -- appropriate here because each model's
     pairs are drawn independently (even where the same underlying images
     are used, which model ran on which image pair differs per model's own
     build_pairs call), not matched/paired samples. Also reports a
     percentile BOOTSTRAP confidence interval for each model's own mean and
     for the difference.

Usage:
    python analyze_replication.py \
        --model_a_name qwen2.5-vl-3b --model_a_csv intervene_output/qwen2.5-vl-3b/experiment_a_trials.csv \
        --model_b_name qwen2.5-vl-7b --model_b_csv intervene_output/qwen2.5-vl-7b/experiment_a_trials.csv \
        --rel_depth_min 0.60 --rel_depth_max 0.73
"""
import argparse

import numpy as np
import pandas as pd

from intervene import compute_transfer_columns


def pair_level_window_means(trials_df, rel_depth_min, rel_depth_max, position_set="image_tokens",
                             condition="real_b"):
    """One row per pair: the mean minute_transfer ("to_B") rate across the
    layers within [rel_depth_min, rel_depth_max] for that pair, restricted
    to pairs where THIS model's own baseline minutes differ (see module
    docstring, step 3). Returns a pandas Series indexed by pair id -- empty
    if nothing matches (e.g. this trials.csv predates saving relative_depth,
    or the window doesn't overlap any swept layer)."""
    if "relative_depth" not in trials_df.columns:
        raise ValueError(
            "This trials.csv has no 'relative_depth' column -- it predates the multi-model adapter "
            "rewrite (or --layers only swept a subset that happened to skip it). Re-run the sweep with "
            "the current intervene.py, or compute relative_depth manually as layer/num_layers before "
            "calling this function."
        )
    df = compute_transfer_columns(trials_df)
    sub = df[(df["condition"] == condition) & (df["position_set"] == position_set) &
              (df["relative_depth"] >= rel_depth_min) & (df["relative_depth"] <= rel_depth_max) &
              (df["minute_baselines_differ"] == 1.0)]
    pair_means = sub.groupby("pair")["minute_transfer"].mean()
    return pair_means.dropna()


def permutation_test(values_a, values_b, n_permutations=10000, seed=0):
    """Two-sided permutation test on the difference of means. Returns
    (observed_diff, p_value) where observed_diff = mean(a) - mean(b), and
    p_value is the fraction of label-shuffled differences at least as
    extreme (|.| >=) as the observed one."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    observed_diff = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        diff = pooled[:n_a].mean() - pooled[n_a:].mean()
        if abs(diff) >= abs(observed_diff):
            count += 1
    p_value = count / n_permutations
    return observed_diff, p_value


def bootstrap_ci(values, n_bootstrap=10000, seed=0, ci=0.95):
    """Percentile bootstrap CI for the mean of `values`. Returns
    (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_diff_ci(values_a, values_b, n_bootstrap=10000, seed=0, ci=0.95):
    """Percentile bootstrap CI for mean(a) - mean(b), resampling each
    group independently. Returns (diff, lo, hi)."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n_a, n_b = len(a), len(b)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sa = a[rng.randint(0, n_a, size=n_a)]
        sb = b[rng.randint(0, n_b, size=n_b)]
        diffs[i] = sa.mean() - sb.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def compare_models(name_a, trials_a, name_b, trials_b, rel_depth_min, rel_depth_max,
                    position_set="image_tokens", condition="real_b",
                    n_permutations=10000, n_bootstrap=10000, seed=0):
    """Runs the full pipeline and returns (report_text, result_dict)."""
    pairs_a = pair_level_window_means(trials_a, rel_depth_min, rel_depth_max, position_set, condition)
    pairs_b = pair_level_window_means(trials_b, rel_depth_min, rel_depth_max, position_set, condition)

    lines = ["=== REPLICATION: PAIR-LEVEL COMPARISON ===", ""]
    lines.append(f"condition={condition!r}  position_set={position_set!r}  "
                 f"relative_depth in [{rel_depth_min:.2f}, {rel_depth_max:.2f}]")
    lines.append(f"restriction: minute_baselines_differ==1 (this model's OWN A/target stated-minute "
                 "baselines differ), applied independently to each model")
    lines.append("")

    if len(pairs_a) == 0 or len(pairs_b) == 0:
        lines.append(f"{name_a}: {len(pairs_a)} usable pair(s)   {name_b}: {len(pairs_b)} usable pair(s)")
        lines.append("At least one model has ZERO usable pairs in this window/restriction -- cannot run the "
                     "permutation test or bootstrap CIs. Check the window overlaps swept layers and that "
                     "--exclude_pairs_from wasn't so aggressive it excluded everything.")
        text = "\n".join(lines)
        print("\n" + text)
        return text, {"n_a": len(pairs_a), "n_b": len(pairs_b)}

    mean_a, lo_a, hi_a = bootstrap_ci(pairs_a.to_numpy(), n_bootstrap=n_bootstrap, seed=seed)
    mean_b, lo_b, hi_b = bootstrap_ci(pairs_b.to_numpy(), n_bootstrap=n_bootstrap, seed=seed + 1)
    diff, p_value = permutation_test(pairs_a.to_numpy(), pairs_b.to_numpy(),
                                      n_permutations=n_permutations, seed=seed)
    diff_check, lo_diff, hi_diff = bootstrap_diff_ci(pairs_a.to_numpy(), pairs_b.to_numpy(),
                                                      n_bootstrap=n_bootstrap, seed=seed + 2)

    lines.append(f"{name_a:<20} n_pairs={len(pairs_a):>4}  mean to_B={mean_a:.1%}  "
                 f"95% bootstrap CI=[{lo_a:.1%}, {hi_a:.1%}]")
    lines.append(f"{name_b:<20} n_pairs={len(pairs_b):>4}  mean to_B={mean_b:.1%}  "
                 f"95% bootstrap CI=[{lo_b:.1%}, {hi_b:.1%}]")
    lines.append("")
    lines.append(f"difference ({name_a} - {name_b}): {diff:+.1%}   "
                 f"95% bootstrap CI=[{lo_diff:+.1%}, {hi_diff:+.1%}]")
    lines.append(f"permutation test (two-sided, {n_permutations} shuffles): p = {p_value:.4f}")
    lines.append("")
    lines.append("Pair-level means are computed per pair as the mean minute_transfer (\"to_B\") rate across")
    lines.append("whatever layers fall inside the relative-depth window for that pair -- NaN (unparseable)")
    lines.append("layers are skipped, not counted as 0; a pair with every window layer NaN is dropped.")

    text = "\n".join(lines)
    print("\n" + text)
    result = {
        "n_a": len(pairs_a), "n_b": len(pairs_b), "mean_a": mean_a, "mean_b": mean_b,
        "ci_a": (lo_a, hi_a), "ci_b": (lo_b, hi_b), "diff": diff, "ci_diff": (lo_diff, hi_diff),
        "p_value": p_value,
    }
    return text, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_a_name", type=str, required=True)
    parser.add_argument("--model_a_csv", type=str, required=True)
    parser.add_argument("--model_b_name", type=str, required=True)
    parser.add_argument("--model_b_csv", type=str, required=True)
    parser.add_argument("--rel_depth_min", type=float, default=0.60)
    parser.add_argument("--rel_depth_max", type=float, default=0.73)
    parser.add_argument("--position_set", type=str, default="image_tokens")
    parser.add_argument("--condition", type=str, default="real_b")
    parser.add_argument("--n_permutations", type=int, default=10000)
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_txt", type=str, default=None, help="optional path to also save the report text")
    args = parser.parse_args()

    trials_a = pd.read_csv(args.model_a_csv)
    trials_b = pd.read_csv(args.model_b_csv)
    print(f"Loaded {len(trials_a)} row(s) from '{args.model_a_csv}' ({args.model_a_name}), "
          f"{len(trials_b)} row(s) from '{args.model_b_csv}' ({args.model_b_name}).")

    text, _ = compare_models(
        args.model_a_name, trials_a, args.model_b_name, trials_b,
        args.rel_depth_min, args.rel_depth_max, position_set=args.position_set, condition=args.condition,
        n_permutations=args.n_permutations, n_bootstrap=args.n_bootstrap, seed=args.seed)

    if args.out_txt:
        with open(args.out_txt, "w") as f:
            f.write(text + "\n")
        print(f"\nReport saved to '{args.out_txt}'.")


if __name__ == "__main__":
    main()
