"""
eval_behavior.py -- STEP 1 behavior check: ask a VLM what time each clock shows.

Loads Qwen2.5-VL-3B-Instruct in float16, shows it each clock image generated
by clocks.py, asks it to read the time, and logs every answer (parsed and
raw) to a results CSV. This script does NOT do any interpretability -- it
only records model behavior for later analysis (analyze.py).

Usage as a script:
    python eval_behavior.py --data_csv data/data.csv --images_dir data \
        --out_csv results/results.csv

Usage as a library (e.g. from the notebook):
    from eval_behavior import run_eval
    results_df = run_eval(data_csv="data/data.csv", images_dir="data",
                           out_csv="results/results.csv")
"""

import argparse
import os
import re

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

PROMPT = "What time does this clock show? Answer only in HH:MM format."
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SEED = 0

# Matches things like "3:45", "03:45", "12:05" anywhere in the model's reply.
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")


def parse_time_answer(text):
    """Extract (hour, minute) ints from a model reply string.

    Returns (hour, minute, success). success=False (hour/minute=None) if no
    HH:MM pattern was found, or if the numbers are out of a valid clock range.
    """
    match = TIME_RE.search(text)
    if not match:
        return None, None, False

    hour, minute = int(match.group(1)), int(match.group(2))

    # Sanity-check the range. Models sometimes output 24h-style hours (e.g.
    # "13:45") or garbage; treat those as parse failures too since our
    # ground truth is always a 12-hour analog reading (hour in 1-12).
    if not (0 <= hour <= 23) or not (0 <= minute <= 59):
        return None, None, False
    hour_12 = hour if 1 <= hour <= 12 else (hour - 12 if hour > 12 else 12)

    return hour_12, minute, True


def load_model(model_id=MODEL_ID):
    """Load the VLM + processor in float16. Uses device_map='auto' so it
    places the model on GPU if available (works on a Kaggle T4)."""
    print(f"Loading {model_id} in float16 ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


@torch.no_grad()
def ask_model(model, processor, image_path, prompt=PROMPT, max_new_tokens=16):
    """Send one clock image + prompt to the model, return its raw text reply."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Build the chat-formatted prompt text, then let the processor turn the
    # PIL image + text into model inputs (this handles image tokenization).
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Strip the prompt tokens off the front so we only decode the new reply.
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    reply = processor.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    return reply.strip()


def run_eval(
    data_csv="data/data.csv",
    images_dir="data",
    out_csv="results/results.csv",
    model_id=MODEL_ID,
    max_new_tokens=16,
    max_images=None,
    seed=SEED,
):
    """Run the behavior check over every clock in `data_csv` and save results.

    Generation is deterministic: `do_sample=False` (greedy decoding) in
    `ask_model`, plus a fixed seed set here for full reproducibility (greedy
    decoding shouldn't need one, but this also pins any other randomness,
    e.g. in kernel selection). `max_new_tokens` is kept small since the
    expected answer is just "HH:MM".

    Writes:
      - out_csv: one row per image with the true time, the model's raw text
        reply, the parsed prediction, and a parse_success flag.
      - <out_csv dir>/parse_failures.csv: just the rows that failed to parse,
        for quick inspection.

    Returns the results DataFrame.
    """
    set_seed(seed)

    df = pd.read_csv(data_csv)
    if max_images is not None:
        df = df.head(max_images)

    model, processor = load_model(model_id)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating clocks"):
        image_path = os.path.join(images_dir, row["filename"])
        raw_reply = ask_model(model, processor, image_path, max_new_tokens=max_new_tokens)
        pred_hour, pred_minute, ok = parse_time_answer(raw_reply)

        rows.append({
            "filename": row["filename"],
            "true_hour": row["hour"],
            "true_minute": row["minute"],
            "raw_answer": raw_reply,
            "pred_hour": pred_hour,
            "pred_minute": pred_minute,
            "parse_success": ok,
        })

    results_df = pd.DataFrame(rows)

    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(out_csv, index=False)

    failures = results_df[~results_df["parse_success"]]
    failures_path = os.path.join(out_dir, "parse_failures.csv")
    failures.to_csv(failures_path, index=False)

    n_fail = len(failures)
    print(f"Done. {len(results_df)} images evaluated, {n_fail} parse failures "
          f"({n_fail / len(results_df):.1%}).")
    print(f"Results saved to '{out_csv}', parse failures to '{failures_path}'.")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a VLM's clock-reading behavior.")
    parser.add_argument("--data_csv", type=str, default="data/data.csv",
                         help="CSV of clock metadata produced by clocks.py")
    parser.add_argument("--images_dir", type=str, default="data",
                         help="directory containing the clock PNGs")
    parser.add_argument("--out_csv", type=str, default="results/results.csv",
                         help="where to write the results CSV")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_images", type=int, default=None,
                         help="only evaluate the first N images (useful for a quick test run)")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed for reproducibility")
    args = parser.parse_args()

    run_eval(
        data_csv=args.data_csv,
        images_dir=args.images_dir,
        out_csv=args.out_csv,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_images=args.max_images,
        seed=args.seed,
    )
