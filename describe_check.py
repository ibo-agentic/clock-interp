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
import pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
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

@torch.no_grad()
def ask(model, processor, path, max_new_tokens=120):
    img = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

def main(data_csv="data/data.csv", images_dir="data",
         out_csv="results/describe.csv", max_images=None, seed=0):
    set_seed(seed)
    df = pd.read_csv(data_csv)
    if max_images: df = df.head(max_images)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Describing clocks"):
        reply = ask(model, processor, os.path.join(images_dir, r["filename"]))
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
    p.add_argument("--out_csv", type=str, default="results/describe.csv")
    p.add_argument("--data_csv", type=str, default="data/data.csv")
    p.add_argument("--images_dir", type=str, default="data")
    a = p.parse_args()
    main(data_csv=a.data_csv, images_dir=a.images_dir,
         max_images=a.max_images, out_csv=a.out_csv)
