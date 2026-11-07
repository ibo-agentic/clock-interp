"""
describe_check.py -- The key experiment.

The model often NAMES the right hand positions out loud, then computes the
wrong time. This measures, over many images, how often:
  (a) the described minute-hand number is correct
  (b) the described hour-hand number is correct
  (c) the final answer is correct
  (d) the description is right BUT the final answer is wrong  <- the finding
"""
import argparse, os, re
import pandas as pd
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"   # this project's original model -- see adapters.py for others
PROMPT = ("Look at the clock. Which number is the long minute hand pointing at? "
          "Then which number is the short hour hand pointing at? "
          "Finally give the time as HH:MM.")

MIN_RE  = re.compile(r"minute hand is pointing at (?:the )?(?:number )?(\d{1,2})", re.I)
HOUR_RE = re.compile(r"hour hand is pointing at (?:the )?(?:number )?(\d{1,2})", re.I)
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")

def true_minute_number(minute):
    """Which clock number the minute hand is nearest (0-59 min -> 1-12)."""
    n = round(minute / 5) % 12
    return 12 if n == 0 else n

def true_hour_number(hour):
    return hour % 12 if hour % 12 != 0 else 12

def parse_reply(text):
    m = MIN_RE.search(text); h = HOUR_RE.search(text); t = TIME_RE.search(text)
    return (int(m.group(1)) if m else None,
            int(h.group(1)) if h else None,
            (int(t.group(1)), int(t.group(2))) if t else None)

# ask(): used to be hardcoded Qwen chat-template + generate + trim -- now
# just adapter.build_inputs/generate_answer (see adapters.py's module
# docstring for why: this script asks a model something, and every script
# that does that shares one interface now instead of three near-duplicates).

def main(data_csv="data/data.csv", images_dir="data",
         out_csv=None, model_id=None, adapter_name=None, max_images=None, seed=0):
    from transformers import set_seed
    from adapters import get_adapter, output_dir_for
    set_seed(seed)
    df = pd.read_csv(data_csv)
    if max_images: df = df.head(max_images)

    adapter = get_adapter(model_id=model_id, adapter_name=adapter_name)
    adapter.load(model_id)
    if out_csv is None:
        out_csv = os.path.join(output_dir_for("results", adapter), "describe.csv")

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Describing clocks"):
        inputs = adapter.build_inputs(os.path.join(images_dir, r["filename"]), PROMPT)
        reply = adapter.generate_answer(inputs, max_new_tokens=120)
        said_min_num, said_hour_num, said_time = parse_reply(reply)
        tmn, thn = true_minute_number(r["minute"]), true_hour_number(r["hour"])
        rows.append({
            "filename": r["filename"], "true_hour": r["hour"], "true_minute": r["minute"],
            "true_minute_number": tmn, "true_hour_number": thn,
            "said_minute_number": said_min_num, "said_hour_number": said_hour_num,
            "pred_hour": said_time[0] if said_time else None,
            "pred_minute": said_time[1] if said_time else None,
            "raw_answer": reply,
            # allow +/-1 on the minute-hand number: the hand often sits between numbers
            "minute_desc_ok": (said_min_num is not None
                               and min((said_min_num - tmn) % 12, (tmn - said_min_num) % 12) <= 1),
            "hour_desc_ok": (said_hour_num is not None and said_hour_num == thn),
            "answer_ok": (said_time is not None
                          and said_time[0] % 12 == r["hour"] % 12 and said_time[1] == r["minute"]),
        })

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    res.to_csv(out_csv, index=False)

    n = len(res)
    both_desc_ok = res["minute_desc_ok"] & res["hour_desc_ok"]
    print("\n=== DESCRIPTION vs ANSWER ===")
    print(f"  n images:                              {n}")
    print(f"  minute-hand number described right:    {res['minute_desc_ok'].mean():.1%}")
    print(f"  hour-hand number described right:      {res['hour_desc_ok'].mean():.1%}")
    print(f"  both described right:                  {both_desc_ok.mean():.1%}")
    print(f"  final answer right:                    {res['answer_ok'].mean():.1%}")
    if both_desc_ok.sum():
        print(f"  >> described BOTH right but answered WRONG: "
              f"{(both_desc_ok & ~res['answer_ok']).sum()}/{both_desc_ok.sum()} "
              f"({(~res.loc[both_desc_ok,'answer_ok']).mean():.1%} of those)")
    print(f"  parse failures (no numbers found):     {res['said_minute_number'].isna().mean():.1%}")
    print(f"\nSaved to {out_csv}")
    return res

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--out_csv", type=str, default=None,
                    help="default: results/<model_short_name>/describe.csv")
    p.add_argument("--data_csv", type=str, default="data/data.csv")
    p.add_argument("--images_dir", type=str, default="data")
    p.add_argument("--model_id", type=str, default=None)
    p.add_argument("--adapter", type=str, default=None)
    a = p.parse_args()
    main(data_csv=a.data_csv, images_dir=a.images_dir, model_id=a.model_id, adapter_name=a.adapter,
         max_images=a.max_images, out_csv=a.out_csv)
