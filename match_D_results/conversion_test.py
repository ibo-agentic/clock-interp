"""conversion_test.py -- how much does an attached IMAGE change the answer to the
SAME text prompt that text_only_v2.py used?

Prompt (every condition, identical text): text_only_v2's headline phrasing b2,
"The long minute hand is pointing at the number {N}, and the short hour hand is
pointing at the number {H}. What time does the clock show? Answer only in
HH:MM format.", with the first 3 lead-ins of text_only_v2.LEAD_INS.
N = 1..12, H = 1..12, 3 lead-ins -> 432 prompts per condition per model.

Conditions (only the image argument differs):
  text_only        no image (must reproduce text_only_v2's b2 rows; checked)
  matching_clock   a REAL clock drawn at H:(N*5 mod 60) with clocks.py defaults
                   (dataset style). NOTE the real-clock hour-hand creep: the
                   hour hand advances 0.5 deg/min, so at e.g. H:55 it sits almost
                   on H+1 while the text says "pointing at H". Kept on purpose
                   (dataset fidelity); the minute is the headline metric.
  blank            a plain white image, same size as the clocks (512x512)
  mismatched_clock a real clock at (H', M') with H' != H at circular hour
                   distance >= 2, and M' a multiple-of-5 minute whose clock
                   NUMBER is at circular distance >= 3 from N. One (H', M') is
                   drawn per (N, H) -- shared across lead-ins and across models
                   (fixed seed) -- so every model sees identical pixels.

The image goes through adapter.build_inputs(image_path_or_None, prompt) -- the
exact call eval_behavior.py uses -- so the chat template, system prompt and
image position are unchanged.  Greedy decoding, set_seed(0), max_new_tokens=32.

Scoring is against the TEXT description's time (H : (N mod 12)*5).  Headline
metric = minute_correct.  For mismatched_clock, matches_image_minute/hour/full
say whether the answer followed the image instead of the text.  Because the
minute positions differ by >= 3, "text minute" and "image minute" never
coincide, so following the text and following the image are exclusive.

USAGE (one process per model; --aggregate needs no GPU):
    python conversion_test.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --smoke
    python conversion_test.py --model_id Qwen/Qwen2.5-VL-3B-Instruct
    python conversion_test.py --model_id Qwen/Qwen2.5-VL-7B-Instruct
    python conversion_test.py --aggregate
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from text_only_v2 import (LEAD_INS, MAX_NEW_TOKENS, PHRASINGS, SEED, git_commit, score, summarize,
                          template_check, truth_minute, with_lead_in, wilson)

PHRASING = "b2"
N_LEAD_INS = 3
POSITIONS = list(range(1, 13))
HOURS = list(range(1, 13))
CONDITIONS = ["text_only", "matching_clock", "blank", "mismatched_clock"]
IMAGE_SIZE = 512            # clocks.generate_clock_image default
IMAGE_SEED = 0
MIN_MINUTE_POS_DIST = 3     # mismatched: circular distance (in clock numbers) from N
MIN_HOUR_DIST = 2           # mismatched: circular hour distance from H
HOUR_HAND_CREEP_NOTE = (
    "matching_clock and mismatched_clock are REAL clocks drawn by clocks.generate_clock_image with default "
    "style: the hour hand creeps 0.5 deg per minute, so at large minute values it sits close to the NEXT "
    "hour number while the text says the hour hand points at H. Deliberate (dataset-style images); the "
    "minute is the headline metric.")


def circ_dist(a, b, mod=12):
    d = abs(a - b) % mod
    return min(d, mod - d)


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

def mismatch_for(n, h, image_seed=IMAGE_SEED):
    """Deterministic (H', M_pos') for text (N, H). Independent of which other
    trials are run (own RandomState per (N, H)), so smoke and full runs, and
    both models, get identical images."""
    rng = np.random.RandomState(image_seed * 100003 + n * 100 + h)
    m_opts = [m for m in POSITIONS if circ_dist(m, n) >= MIN_MINUTE_POS_DIST]
    h_opts = [x for x in HOURS if circ_dist(x, h) >= MIN_HOUR_DIST]
    return int(rng.choice(h_opts)), int(rng.choice(m_opts))


def image_spec(condition, n, h, images_dir, image_seed=IMAGE_SEED):
    """-> (path or None, image_hour, image_minute)."""
    if condition == "text_only":
        return None, None, None
    if condition == "blank":
        return os.path.join(images_dir, "blank.png"), None, None
    if condition == "matching_clock":
        return os.path.join(images_dir, f"match_N{n:02d}_H{h:02d}.png"), h, (n % 12) * 5
    if condition == "mismatched_clock":
        h2, m2 = mismatch_for(n, h, image_seed)
        return os.path.join(images_dir, f"mismatch_N{n:02d}_H{h:02d}.png"), h2, (m2 % 12) * 5
    raise ValueError(condition)


def full_manifest(images_dir, image_seed=IMAGE_SEED):
    rows = [{"file": "blank.png", "kind": "blank", "text_N": None, "text_H": None,
             "image_hour": None, "image_minute": None}]
    for n in POSITIONS:
        for h in HOURS:
            for cond, kind in (("matching_clock", "match"), ("mismatched_clock", "mismatch")):
                path, ih, im = image_spec(cond, n, h, images_dir, image_seed)
                rows.append({"file": os.path.basename(path), "kind": kind, "text_N": n, "text_H": h,
                             "image_hour": ih, "image_minute": im})
    return pd.DataFrame(rows)


def render_image(path, hour, minute):
    from clocks import generate_clock_image
    generate_clock_image(hour, minute, path)    # defaults == original dataset style, 512x512


def ensure_image(condition, n, h, images_dir, image_seed=IMAGE_SEED):
    path, ih, im = image_spec(condition, n, h, images_dir, image_seed)
    if path is None:
        return path, ih, im
    if condition == "blank":
        if not os.path.exists(path):
            from PIL import Image
            Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white").save(path)
    else:
        render_image(path, ih, im)
    return path, ih, im


# ---------------------------------------------------------------------------
# trials
# ---------------------------------------------------------------------------

def build_groups():
    """One group = one prompt in all 4 conditions (interleaved so a partial
    run is balanced). Returns a list of (lead_id, N, H)."""
    return [(lead, n, h) for lead in range(N_LEAD_INS) for n in POSITIONS for h in HOURS]


def prompt_for(lead, n, h):
    return with_lead_in(LEAD_INS[lead], PHRASINGS["b"][PHRASING].format(N=n, H=h))


def add_image_metrics(row, cond, image_hour, image_minute):
    ok = row["parse_ok"]
    if cond == "mismatched_clock":
        pm = row["parsed_answer"]
        mm = bool(ok) and int(pm.split(":")[1]) == image_minute
        mh = bool(ok) and (row["parsed_hour"] % 12) == (image_hour % 12)
        row.update({"matches_image_minute": mm, "matches_image_hour": mh, "matches_image_full": mm and mh})
    else:
        row.update({"matches_image_minute": None, "matches_image_hour": None, "matches_image_full": None})
    return row


def reproduction_check(df, short_name, ref_dir="results"):
    """Compare this run's text_only rows to text_only_v2.csv's b2 rows (same prompts)."""
    p = os.path.join(ref_dir, short_name, "text_only_v2.csv")
    if not os.path.exists(p):
        return {"status": "NOT CHECKED", "reason": f"{p} not found"}
    ref = pd.read_csv(p)
    ref = ref[(ref.task == "b") & (ref.phrasing == PHRASING) & (ref.lead_in_id < N_LEAD_INS)]
    mine = df[df.condition == "text_only"]
    m = mine.merge(ref[["prompt", "raw_output", "minute_correct"]], on="prompt", suffixes=("", "_v2"))
    if m.empty:
        return {"status": "NOT CHECKED", "reason": "no overlapping prompts"}
    exact = (m.raw_output.astype(str) == m.raw_output_v2.astype(str))
    agree = (m.minute_correct.astype(bool) == m.minute_correct_v2.astype(bool))
    return {"status": "CHECKED", "n_compared": int(len(m)), "exact_raw_match_rate": float(exact.mean()),
            "minute_correct_agreement": float(agree.mean()),
            "n_raw_mismatch": int((~exact).sum()),
            "mismatch_examples": [{"prompt": r.prompt, "now": r.raw_output, "text_only_v2": r.raw_output_v2}
                                  for r in m[~exact].head(10).itertuples()]}


def run(model_id=None, adapter_name=None, seed=SEED, smoke=False, max_new_tokens=MAX_NEW_TOKENS,
        image_seed=IMAGE_SEED, results_dir="results", images_dir=None, text_only_dir=None):
    import torch
    import transformers
    from tqdm import tqdm
    from transformers import set_seed
    from adapters import get_adapter, output_dir_for
    set_seed(seed)

    adapter = get_adapter(model_id=model_id, adapter_name=adapter_name)
    adapter.load(model_id)
    resolved_model_id = model_id or adapter.default_model_id
    out_base = output_dir_for(results_dir, adapter)
    suffix = "_smoke" if smoke else ""
    out_csv = os.path.join(out_base, f"conversion_test{suffix}.csv")
    cfg_path = os.path.join(out_base, f"conversion_test_run_config{suffix}.json")
    images_dir = images_dir or ("conversion_images_smoke" if smoke else "conversion_images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(out_base, exist_ok=True)

    manifest = full_manifest(images_dir, image_seed)
    manifest_hash = hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()
    manifest.to_csv(os.path.join(images_dir, "manifest.csv"), index=False)

    groups = build_groups()
    n_groups_full = len(groups)
    if smoke:   # 5 prompts x 4 conditions = 20 generations
        step = max(1, len(groups) // 5)
        groups = groups[::step][:5]

    rows, text_only_raw = [], {}
    total = len(groups) * len(CONDITIONS)
    pbar = tqdm(total=total, desc=f"conversion test [{adapter.short_name}]")
    for lead, n, h in groups:
        prompt = prompt_for(lead, n, h)
        trial = {"task": "b", "hour": h, "truth_minute": truth_minute("b", n)}
        for cond in CONDITIONS:
            path, ih, im = ensure_image(cond, n, h, images_dir, image_seed)
            inputs = adapter.build_inputs(path, prompt)          # path=None -> text-only; else image first, then text
            raw = adapter.generate_answer(inputs, max_new_tokens=max_new_tokens)
            row = {"model": adapter.short_name, "condition": cond, "position": n, "hour": h,
                   "lead_in_id": lead, "prompt": prompt, "image_path": path,
                   "image_hour": ih, "image_minute": im, "truth_minute": trial["truth_minute"],
                   "raw_output": raw, **score(trial, raw), "max_new_tokens": max_new_tokens}
            if cond == "text_only":
                text_only_raw[(lead, n, h)] = raw
                row["changed_vs_text_only"] = None
            else:
                row["changed_vs_text_only"] = raw != text_only_raw[(lead, n, h)]
            rows.append(add_image_metrics(row, cond, ih, im))
            pbar.update(1)
            if len(rows) % 200 == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)   # partial flush
    pbar.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    repro = reproduction_check(df, adapter.short_name, text_only_dir or results_dir)

    config = {
        "script": "conversion_test.py", "model_id": resolved_model_id, "adapter": type(adapter).__name__,
        "short_name": adapter.short_name,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "python_version": sys.version.split()[0], "git_commit": git_commit(),
        "seed": seed, "image_seed": image_seed, "max_new_tokens": max_new_tokens,
        "decoding": {"do_sample": False, "strategy": "greedy", "use_cache": True},
        "dtype": "float16", "device_map": "auto",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "smoke": smoke, "n_generations_run": len(df), "n_prompts_full_design": n_groups_full,
        "n_generations_full_design": n_groups_full * len(CONDITIONS),
        "phrasing": PHRASING, "phrasing_template": PHRASINGS["b"][PHRASING],
        "lead_ins_used": LEAD_INS[:N_LEAD_INS],
        "conditions": {
            "text_only": "no image",
            "matching_clock": "real clock at H:(N*5 mod 60), clocks.generate_clock_image defaults",
            "blank": f"plain white {IMAGE_SIZE}x{IMAGE_SIZE} RGB",
            "mismatched_clock": (f"real clock at (H', M'): circular hour distance(H,H') >= {MIN_HOUR_DIST}; M' a "
                                 f"multiple-of-5 minute whose clock number is at circular distance >= "
                                 f"{MIN_MINUTE_POS_DIST} from N; one draw per (N,H), seeded by image_seed"),
        },
        "image_size": IMAGE_SIZE, "images_dir": images_dir, "image_manifest_sha256": manifest_hash,
        "hour_hand_creep_note": HOUR_HAND_CREEP_NOTE,
        "chat_template_check": template_check(adapter, "<PROMPT>"),
        "chat_path": "adapter.build_inputs(image_path_or_None, prompt) -> adapter.generate_answer(); "
                     "identical call to eval_behavior.py (image first, then text)",
        "text_only_reproduction_vs_text_only_v2": repro,
    }
    notes = getattr(adapter, "run_config_notes", None)    # only families that define it (InternVL); Qwen configs unchanged
    if callable(notes):
        config["adapter_notes"] = notes()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved {len(df)} generations to '{out_csv}', config to '{cfg_path}'.")
    print(f"text_only reproduction vs text_only_v2: {repro['status']}"
          + (f" -- {repro['reason']}" if "reason" in repro else
             f" -- n={repro['n_compared']}, exact raw match {repro['exact_raw_match_rate']:.1%}, "
             f"minute_correct agreement {repro['minute_correct_agreement']:.1%}"))
    print(f"parse_ok: {df.parse_ok.mean():.1%}")
    for cond, g in df.groupby("condition"):
        extra = f"  matches_image_minute {g.matches_image_minute.astype(float).mean():.1%}" if cond == "mismatched_clock" else ""
        print(f"  {cond:<17} minute_correct {g.minute_correct.mean():.1%} (n={len(g)}){extra}")

    if smoke:
        print("\n=== 10 RAW OUTPUTS (check parsing and image handling before the full run) ===")
        picks = []
        for cond, k in (("matching_clock", 3), ("mismatched_clock", 3), ("blank", 2), ("text_only", 2)):
            picks.append(df[df.condition == cond].head(k))
        for _, r in pd.concat(picks).iterrows():
            print(f"[{r.condition} N={r.position} H={r.hour}] text truth minute={r.truth_minute}"
                  + (f" | image shows {int(r.image_hour)}:{int(r.image_minute):02d}" if pd.notna(r.image_minute) else ""))
            print(f"   image  : {r.image_path}")
            print(f"   prompt : {r.prompt!r}")
            print(f"   raw    : {r.raw_output!r}")
            print(f"   parsed : {r.parsed_answer!r} parse_ok={r.parse_ok} minute_correct={r.minute_correct} "
                  f"correct={r.correct} matches_image_minute={r.matches_image_minute}")
    return df


# ---------------------------------------------------------------------------
# paired tests
# ---------------------------------------------------------------------------

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: binomial test of b vs c discordant
    pairs under p=0.5.  b + c == 0 -> p = 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm step-down adjusted p-values, same order as input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [None] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_tests(df, metric="minute_correct"):
    rows = []
    key = ["lead_in_id", "position", "hour"]
    for model, gm in df.groupby("model"):
        base = gm[gm.condition == "text_only"][key + [metric]].rename(columns={metric: "t"})
        for cond in ("matching_clock", "blank", "mismatched_clock"):
            img = gm[gm.condition == cond][key + [metric]].rename(columns={metric: "i"})
            m = base.merge(img, on=key)
            m["t"], m["i"] = m["t"].astype(bool), m["i"].astype(bool)

            def counts(d):
                return (int((d.t & d.i).sum()), int((d.t & ~d.i).sum()),
                        int((~d.t & d.i).sum()), int((~d.t & ~d.i).sum()))

            per_pos = []
            for pos in POSITIONS:
                d = m[m.position == pos]
                if len(d):
                    bc, ot, oi, bw = counts(d)
                    per_pos.append({"model": model, "comparison": f"text_only_vs_{cond}", "position": pos,
                                    "n": len(d), "both_correct": bc, "only_text_correct": ot,
                                    "only_image_correct": oi, "both_wrong": bw,
                                    "p_value": mcnemar_exact(ot, oi)})
            for r, ph in zip(per_pos, holm([r["p_value"] for r in per_pos]) if per_pos else []):
                r["p_holm"] = ph
            bc, ot, oi, bw = counts(m)
            overall = {"model": model, "comparison": f"text_only_vs_{cond}", "position": "all", "n": len(m),
                       "both_correct": bc, "only_text_correct": ot, "only_image_correct": oi, "both_wrong": bw,
                       "p_value": mcnemar_exact(ot, oi)}
            overall["p_holm"] = overall["p_value"]   # single test: no multiplicity correction applies
            rows.extend([overall] + per_pos)
    cols = ["model", "comparison", "position", "n", "both_correct", "only_text_correct",
            "only_image_correct", "both_wrong", "p_value", "p_holm"]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate(models, results_dir="results", out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or os.path.join(results_dir, "conversion_test")
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for m in models:
        p = os.path.join(results_dir, m, "conversion_test.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f"NOTE: {p} not found -- skipping model '{m}'.")
    if not frames:
        raise SystemExit("No per-model conversion_test.csv found. Run the generation step first.")
    df = pd.concat(frames, ignore_index=True)
    for c in ("minute_correct", "correct", "parse_ok"):
        df[c] = df[c].astype(bool)
    df["position_all"] = "all"

    parts = []
    for metric, cond_filter in [("minute_correct", None), ("correct", None),
                                ("matches_image_minute", "mismatched_clock"),
                                ("matches_image_hour", "mismatched_clock"),
                                ("matches_image_full", "mismatched_clock"),
                                ("changed_vs_text_only", "!text_only")]:
        d = df
        if cond_filter == "!text_only":
            d = d[d.condition != "text_only"]
        elif cond_filter:
            d = d[d.condition == cond_filter]
        d = d.copy()
        d[metric] = d[metric].astype(bool)
        for pos_col in ("position", "position_all"):
            s = summarize(d, ["model", "condition", pos_col], metric).rename(columns={pos_col: "position"})
            parts.append(s)
    summary = pd.concat(parts, ignore_index=True)
    summary["position"] = summary["position"].astype(str)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    tests = paired_tests(df)
    tests.to_csv(os.path.join(out_dir, "paired_tests.csv"), index=False)

    (df.groupby(["model", "condition"]).agg(n=("parse_ok", "size"), parse_ok=("parse_ok", "mean"))
       .reset_index().to_csv(os.path.join(out_dir, "parse_rates.csv"), index=False))

    # ---- plot ----
    models_present = list(df.model.unique())
    ncols = 2 if len(models_present) > 1 else 1
    nrows = math.ceil(len(models_present) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5 * nrows), sharey=True, squeeze=False)
    for extra_ax in [a for row in axes for a in row][len(models_present):]:
        extra_ax.axis("off")
    colors = {"text_only": "tab:gray", "matching_clock": "tab:green", "blank": "tab:blue",
              "mismatched_clock": "tab:red"}
    offsets = {c: (i - 1.5) * 0.06 for i, c in enumerate(CONDITIONS)}
    for ax, model in zip([a for row in axes for a in row], models_present):
        for cond in CONDITIONS:
            g = summary[(summary.model == model) & (summary.condition == cond) &
                        (summary.metric == "minute_correct") & (summary.position != "all")].copy()
            g["position"] = g.position.astype(int)
            g = g.sort_values("position")
            if g.empty:
                continue
            ax.errorbar(g.position + offsets[cond], g.accuracy,
                        yerr=[g.accuracy - g.ci95_lo, g.ci95_hi - g.accuracy], marker="o", capsize=2,
                        color=colors[cond], label=f"{cond} (minute matches TEXT)")
        g = summary[(summary.model == model) & (summary.condition == "mismatched_clock") &
                    (summary.metric == "matches_image_minute") & (summary.position != "all")].copy()
        if not g.empty:
            g["position"] = g.position.astype(int)
            g = g.sort_values("position")
            ax.errorbar(g.position + 0.12, g.accuracy, yerr=[g.accuracy - g.ci95_lo, g.ci95_hi - g.accuracy],
                        marker="s", ls=":", capsize=2, color="darkred", alpha=0.8,
                        label="mismatched_clock (minute matches IMAGE)")
        ax.set_title(model)
        ax.set_xlabel("minute-hand position N in the text (clock number)")
        ax.set_xticks(POSITIONS)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    for row in axes:
        row[0].set_ylabel("minute accuracy vs TEXT (95% Wilson CI)")
    fig.suptitle("Conversion test: same text prompt, different images (n=36 per point)", fontsize=12)
    fig.text(0.5, 0.005, "Caveat: CIs treat the 36 prompts per position (12 hours x 3 lead-ins) as independent; "
             "they are crossed, so CIs are somewhat optimistic. Paired McNemar tests: paired_tests.csv.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "minute_accuracy_vs_position.png"), dpi=150)

    print(f"Wrote summary.csv, paired_tests.csv, parse_rates.csv, minute_accuracy_vs_position.png to '{out_dir}'.")
    print("\n=== OVERALL (n=432 per cell) ===")
    ov = summary[(summary.position == "all")]
    for r in ov.sort_values(["model", "metric", "condition"]).itertuples():
        print(f"{r.model:<15} {r.condition:<17} {r.metric:<21} {r.accuracy:6.1%} [{r.ci95_lo:.1%}, {r.ci95_hi:.1%}] (n={r.n})")
    print("\n=== PAIRED McNEMAR, text_only vs image condition, overall ===")
    for r in tests[tests.position == "all"].itertuples():
        print(f"{r.model:<15} {r.comparison:<32} only_text={r.only_text_correct:>3} only_image={r.only_image_correct:>3} "
              f"both_ok={r.both_correct:>3} both_wrong={r.both_wrong:>3}  p={r.p_value:.4g}")
    return summary, tests


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--image_seed", type=int, default=IMAGE_SEED)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--smoke", action="store_true", help="5 prompts x 4 conditions, print 10 raw outputs")
    ap.add_argument("--aggregate", action="store_true", help="CPU only: summary, paired tests, plot")
    ap.add_argument("--models", type=str, default="qwen2.5-vl-3b,qwen2.5-vl-7b,internvl3-2b,internvl3-8b")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--images_dir", type=str, default=None)
    ap.add_argument("--text_only_dir", type=str, default=None,
                    help="folder holding <model_short>/text_only_v2.csv for the reproduction check "
                         "(default: --results_dir)")
    args = ap.parse_args()
    if args.aggregate:
        aggregate([m.strip() for m in args.models.split(",")], results_dir=args.results_dir)
    else:
        run(model_id=args.model_id, adapter_name=args.adapter, seed=args.seed, smoke=args.smoke,
            max_new_tokens=args.max_new_tokens, image_seed=args.image_seed, results_dir=args.results_dir,
            images_dir=args.images_dir, text_only_dir=args.text_only_dir)
