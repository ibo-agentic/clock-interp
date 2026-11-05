"""text_only_v2.py -- GATE TEST: can Qwen2.5-VL convert clock-hand positions to
minutes in TEXT-ONLY mode (no image anywhere in the input)?

Three tasks, every one a pure-text prompt sent through the SAME code path as
eval_behavior.py's image prompts: adapter.build_inputs(image, prompt) +
adapter.generate_answer(...). With image=None that path builds one user
message through the processor's own chat template (Qwen's default system
prompt comes from that template; nothing is overridden), so the follow-up
image experiment can add the image to these exact prompts and change nothing
else. run_config.json records the rendered template (text-only vs. with an
image placeholder) so that claim can be checked, not assumed.

  (a) minute only   "The minute hand points at {N}. How many minutes past the hour is it? ..."
  (b) full time     "The hour hand is just past {H} and the minute hand points at {N}. What time is it? ..."
  (c) arithmetic    "What is {N} times 5? Answer with a number only."   (control)

Truth: clock minute = (N mod 12) * 5, so position 12 -> 0.  Arithmetic truth
= N * 5, so N=12 -> 60 (deliberately NOT mod 12: that difference is the point
of the control).  Task (b) HEADLINE metric = minute_correct (the hour is
ambiguous at large N when the prompt says "just past H"); full-time
correctness is logged and summarized as a secondary metric.  HEADLINE
phrasing for (b) = b2, which mirrors the wording of the model's own hand
descriptions in describe_check.py outputs.

Greedy decoding is deterministic, so a repeated prompt adds no information:
every trial is a DISTINCT prompt.  Per-position n comes from crossing
phrasings x lead-in sentences (a, c) or x hours x lead-ins (b).  Wilson CIs
treat these as independent, which they are not fully (crossed factors) --
see the caveat printed on the plot and the per-phrasing table.

USAGE (one process per model; --aggregate needs no GPU):
    python text_only_v2.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --smoke
    python text_only_v2.py --model_id Qwen/Qwen2.5-VL-3B-Instruct
    python text_only_v2.py --model_id Qwen/Qwen2.5-VL-7B-Instruct
    python text_only_v2.py --aggregate
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import subprocess
import sys

import pandas as pd

from eval_behavior import parse_time_answer   # same parser as the image runs

SEED = 0
MAX_NEW_TOKENS = 32   # NOT eval_behavior.py's 16: chosen for this script AND the later image runs
POSITIONS = list(range(1, 13))
HOURS = list(range(1, 13))
HEADLINE_B = "b2"

PHRASINGS = {
    "a": {
        "a1": "The minute hand points at {N}. How many minutes past the hour is it? Answer with a number only.",
        # mirrors describe_check.py outputs: "The long minute hand is pointing at the number 12, ..."
        "a2": "The long minute hand is pointing at the number {N}. How many minutes past the hour does this represent? Answer with a number only.",
        "a3": "On an analog clock, the minute hand is on the {N}. What is the minute value? Answer with a number only.",
    },
    "b": {
        "b1": "The hour hand is just past {H} and the minute hand points at {N}. What time is it? Answer only in HH:MM format.",
        "b2": "The long minute hand is pointing at the number {N}, and the short hour hand is pointing at the number {H}. What time does the clock show? Answer only in HH:MM format.",
        "b3": "On an analog clock the short hand is just past the {H} and the long hand is on the {N}. What time is it? Answer only in HH:MM format.",
    },
    "c": {
        "c1": "What is {N} times 5? Answer with a number only.",
        "c2": "Compute {N} multiplied by 5. Answer with a number only.",
        "c3": "{N} x 5 = ? Answer with a number only.",
    },
}

# 10 neutral lead-ins (id 0 = none). Used for (a) and (c) in full; (b) uses the
# first --b_lead_ins of them (default 3) so the headline phrasing b2 still has
# >= 30 trials per position.
LEAD_INS = [
    "",
    "Consider a standard 12-hour analog clock.",
    "Here is a question about reading a clock.",
    "Answer the following question.",
    "Think of a round clock face with the numbers 1 to 12.",
    "This is a quick quiz.",
    "Please read the question carefully.",
    "Imagine a wall clock.",
    "Here is a short problem.",
    "Use standard conventions.",
]

INT_RE = re.compile(r"-?\d+")


def with_lead_in(lead, prompt):
    return f"{lead} {prompt}" if lead else prompt


def truth_minute(task, n):
    return n * 5 if task == "c" else (n % 12) * 5


def build_trials(b_lead_ins=3):
    trials = []
    for task in ("a", "b", "c"):
        n_leads = b_lead_ins if task == "b" else len(LEAD_INS)
        for phrasing, template in PHRASINGS[task].items():
            for lead_id in range(n_leads):
                for n in POSITIONS:
                    hours = HOURS if task == "b" else [None]
                    for h in hours:
                        base = template.format(N=n, H=h)
                        trials.append({
                            "task": task, "phrasing": phrasing, "position": n, "hour": h,
                            "lead_in_id": lead_id, "prompt": with_lead_in(LEAD_INS[lead_id], base),
                            "truth_minute": truth_minute(task, n),
                        })
    return trials


def score(trial, raw):
    task = trial["task"]
    out = {"parsed_answer": None, "parsed_hour": None, "parse_ok": False, "format_ok": False,
           "correct": False, "minute_correct": False, "hour_correct": None}
    if task == "b":
        h12, minute, ok = parse_time_answer(raw)
        out["parse_ok"] = ok
        if ok:
            out["parsed_answer"] = f"{h12:02d}:{minute:02d}"
            out["parsed_hour"] = h12
            out["minute_correct"] = minute == trial["truth_minute"]
            out["hour_correct"] = (h12 % 12) == (trial["hour"] % 12)
            out["correct"] = bool(out["minute_correct"] and out["hour_correct"])
        out["format_ok"] = bool(re.fullmatch(r"\d{1,2}:\d{2}", raw.strip().rstrip(".")))
    else:
        stripped = raw.strip().rstrip(".")
        m = INT_RE.search(raw)
        out["format_ok"] = bool(re.fullmatch(r"-?\d+", stripped))
        if m:
            out["parse_ok"] = True
            out["parsed_answer"] = int(m.group(0))
            out["correct"] = out["minute_correct"] = out["parsed_answer"] == trial["truth_minute"]
    return out


def headline_correct(row):
    """The one boolean per trial the headline plots use: minute_correct for
    (b), plain correctness for (a)/(c)."""
    return bool(row["minute_correct"])


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def template_check(adapter, prompt):
    """Render the chat template for a text-only vs. an image-bearing message
    so the claim 'only the image placeholder differs' is inspectable."""
    info = {}
    try:
        proc = adapter.processor
        txt = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                                       tokenize=False, add_generation_prompt=True)
        img = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                                       tokenize=False, add_generation_prompt=True)
        info = {"text_only_rendered": txt, "with_image_rendered": img,
                "system_prompt_present": "<|im_start|>system" in txt}
    except Exception as e:   # adapters without .processor / different template API
        info = {"error": f"{type(e).__name__}: {e}"}
    return info


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def run(model_id=None, adapter_name=None, out_csv=None, seed=SEED, smoke=False, b_lead_ins=3,
        max_new_tokens=MAX_NEW_TOKENS):
    import torch
    import transformers
    from tqdm import tqdm
    from transformers import set_seed
    from adapters import get_adapter, output_dir_for
    set_seed(seed)

    adapter = get_adapter(model_id=model_id, adapter_name=adapter_name)
    adapter.load(model_id)
    resolved_model_id = model_id or adapter.default_model_id
    base = output_dir_for("results", adapter)
    if out_csv is None:
        out_csv = os.path.join(base, "text_only_v2_smoke.csv" if smoke else "text_only_v2.csv")
    cfg_path = os.path.join(os.path.dirname(out_csv), "run_config_smoke.json" if smoke else "run_config.json")

    trials = build_trials(b_lead_ins)
    n_total = len(trials)
    if smoke:   # ~20 prompts evenly spread over every task/phrasing/position
        step = max(1, len(trials) // 20)
        trials = trials[::step][:20]

    rows = []
    for i, t in enumerate(tqdm(trials, desc=f"text-only gate [{adapter.short_name}]")):
        inputs = adapter.build_inputs(None, t["prompt"])        # image=None: pure text, same path as image runs
        raw = adapter.generate_answer(inputs, max_new_tokens=max_new_tokens)
        row = {"model": adapter.short_name, **t, "raw_output": raw, **score(t, raw),
               "max_new_tokens": max_new_tokens}
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    config = {
        "script": "text_only_v2.py", "model_id": resolved_model_id, "adapter": type(adapter).__name__,
        "short_name": adapter.short_name,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "python_version": sys.version.split()[0], "git_commit": git_commit(),
        "seed": seed, "max_new_tokens": max_new_tokens,
        "decoding": {"do_sample": False, "strategy": "greedy", "use_cache": True},
        "dtype": "float16", "device_map": "auto",
        "note_max_new_tokens": "32 here and in later image runs; eval_behavior.py's original runs used 16",
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "smoke": smoke, "n_trials_run": len(df), "n_trials_full_design": n_total, "b_lead_ins": b_lead_ins,
        "lead_ins": LEAD_INS, "phrasings": PHRASINGS, "headline_phrasing_b": HEADLINE_B,
        "prompts_sha256": hashlib.sha256("\n".join(t["prompt"] for t in build_trials(b_lead_ins)).encode()).hexdigest(),
        "chat_template_check": template_check(adapter, "<PROMPT>"),
        "chat_path": "adapter.build_inputs(image_or_None, prompt) -> adapter.generate_answer(); identical to eval_behavior.py",
    }
    notes = getattr(adapter, "run_config_notes", None)    # only families that define it (InternVL); Qwen configs unchanged
    if callable(notes):
        config["adapter_notes"] = notes()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved {len(df)} trials to '{out_csv}' and config to '{cfg_path}'.")
    print(f"parse_ok: {df.parse_ok.mean():.1%}   format_ok (bare answer): {df.format_ok.mean():.1%}")
    for task, g in df.groupby("task"):
        print(f"  task {task}: headline-correct {g.minute_correct.mean():.1%} (n={len(g)})")

    if smoke:
        pick = df.iloc[[int(round(k * (len(df) - 1) / 9)) for k in range(10)]]
        print("\n=== 10 RAW OUTPUTS (check parsing before the full run) ===")
        for _, r in pick.iterrows():
            print(f"[{r.task}/{r.phrasing} N={r.position} H={r.hour}] truth_min={r.truth_minute}")
            print(f"   prompt : {r.prompt!r}")
            print(f"   raw    : {r.raw_output!r}")
            print(f"   parsed : {r.parsed_answer!r} parse_ok={r.parse_ok} minute_correct={r.minute_correct} "
                  f"hour_correct={r.hour_correct}")
    return df


# ---------------------------------------------------------------------------
# aggregation: Wilson CIs, summary tables, plot
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(df, group_cols, metric):
    rows = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n, k = len(g), int(g[metric].astype(bool).sum())
        lo, hi = wilson(k, n)
        rows.append({**dict(zip(group_cols, keys)), "metric": metric, "n": n, "k": k,
                     "accuracy": k / n if n else float("nan"), "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows)


def aggregate(models, results_dir="results", out_dir=None, file_name="text_only_v2.csv"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or os.path.join(results_dir, "text_only_v2")
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for m in models:
        p = os.path.join(results_dir, m, file_name)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f"NOTE: {p} not found -- skipping model '{m}'.")
    if not frames:
        raise SystemExit("No per-model CSVs found. Run the generation step first.")
    df = pd.concat(frames, ignore_index=True)
    df["hour_correct"] = df["hour_correct"].astype("boolean")
    df["full_correct"] = df["correct"].astype(bool)

    parts = []
    # pooled over all phrasings
    for task, metric in [("a", "minute_correct"), ("c", "minute_correct"),
                         ("b", "minute_correct"), ("b", "full_correct")]:
        s = summarize(df[df.task == task], ["model", "task", "position"], metric)
        s.insert(3, "scope", "pooled_all_phrasings")
        parts.append(s)
    # headline scope: b2 only for (b); (a)/(c) have no single headline phrasing -> pooled
    for metric in ("minute_correct", "full_correct"):
        s = summarize(df[(df.task == "b") & (df.phrasing == HEADLINE_B)], ["model", "task", "position"], metric)
        s.insert(3, "scope", f"headline_{HEADLINE_B}")
        parts.append(s)
    summary = pd.concat(parts, ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    by_ph = []
    for metric in ("minute_correct", "full_correct"):
        s = summarize(df, ["model", "task", "phrasing", "position"], metric)
        by_ph.append(s[~((s.task != "b") & (s.metric == "full_correct"))])
    pd.concat(by_ph, ignore_index=True).to_csv(os.path.join(out_dir, "summary_by_phrasing.csv"), index=False)

    parse = (df.groupby(["model", "task", "phrasing"])
               .agg(n=("parse_ok", "size"), parse_ok=("parse_ok", "mean"), format_ok=("format_ok", "mean"))
               .reset_index())
    parse.to_csv(os.path.join(out_dir, "parse_rates.csv"), index=False)

    # ---- plot ----
    panels = [
        ("a", "pooled_all_phrasings", "minute_correct", "(a) minutes past the hour\n(all phrasings x 10 lead-ins)"),
        ("b", f"headline_{HEADLINE_B}", "minute_correct", f"(b) full time, HEADLINE: minute correct\n(phrasing {HEADLINE_B} only)"),
        ("c", "pooled_all_phrasings", "minute_correct", "(c) control: N x 5 (truth = 5N; N=12 -> 60)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (task, scope, metric, title) in zip(axes, panels):
        for model, g in summary[(summary.task == task) & (summary.scope == scope) & (summary.metric == metric)].groupby("model"):
            g = g.sort_values("position")
            ax.errorbar(g.position, g.accuracy, yerr=[g.accuracy - g.ci95_lo, g.ci95_hi - g.accuracy],
                        marker="o", capsize=3, label=f"{model} (n={int(g.n.iloc[0])}/pos)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("minute-hand position (clock number)")
        ax.set_xticks(POSITIONS)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("accuracy (95% Wilson CI)")
    fig.suptitle("Text-only gate test: clock-hand position -> minutes (no image)", fontsize=12)
    fig.text(0.5, 0.005, "Caveat: CIs treat each position's prompts as independent; phrasing/lead-in/hour factors are "
             "crossed, so they are somewhat optimistic. See summary_by_phrasing.csv.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "accuracy_vs_position.png"), dpi=150)

    print(f"Wrote summary.csv, summary_by_phrasing.csv, parse_rates.csv, accuracy_vs_position.png to '{out_dir}'.")
    print("\n=== accuracy per position (headline), 95% Wilson CI ===")
    for task, scope, metric, _ in panels:
        t = summary[(summary.task == task) & (summary.scope == scope) & (summary.metric == metric)]
        for model, g in t.groupby("model"):
            cells = "  ".join(f"{int(r.position)}:{r.accuracy:.0%}[{r.ci95_lo:.0%},{r.ci95_hi:.0%}]"
                              for r in g.sort_values("position").itertuples())
            print(f"task {task} {model}: {cells}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None,
                    help="default: results/<model_short_name>/text_only_v2.csv (text_only_v2_smoke.csv with --smoke)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--b_lead_ins", type=int, default=3,
                    help="how many of the 10 lead-ins task (b) is crossed with (default 3 -> b2 has 36 trials/position)")
    ap.add_argument("--smoke", action="store_true", help="run ~20 prompts, print 10 raw outputs, write *_smoke files")
    ap.add_argument("--aggregate", action="store_true", help="CPU only: merge per-model CSVs -> summary + plot")
    ap.add_argument("--models", type=str, default="qwen2.5-vl-3b,qwen2.5-vl-7b,internvl3-2b,internvl3-8b")
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()
    if args.aggregate:
        aggregate([m.strip() for m in args.models.split(",")], results_dir=args.results_dir)
    else:
        run(model_id=args.model_id, adapter_name=args.adapter, out_csv=args.out_csv, seed=args.seed,
            smoke=args.smoke, b_lead_ins=args.b_lead_ins, max_new_tokens=args.max_new_tokens)
