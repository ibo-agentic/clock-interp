"""text_only_check.py -- Can the model do the 'number on face -> minutes'
conversion when there is NO image? If yes, perception and arithmetic are each
fine on their own, and the failure is in linking them.

Model loading/asking is behind adapters.py's ModelAdapter interface (see its
module docstring) -- pass --model_id/--adapter to run this on a different
model; defaults to this project's original Qwen2.5-VL-3B.
"""
import argparse
import os
import re

TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
NUM_RE = re.compile(r"(\d{1,2})")

# (minute-hand number, hour-hand number, correct answer)
CASES = [
    (10, 8,  "8:50"),
    (2,  7,  "7:10"),
    (6,  5,  "5:30"),
    (4,  10, "10:20"),
    (12, 3,  "3:00"),
    (9,  1,  "1:45"),
    (7,  11, "11:35"),
    (3,  6,  "6:15"),
]

ARITHMETIC_CASES = [2, 4, 6, 7, 9, 10, 12]


def run(model_id=None, adapter_name=None, out_csv=None, seed=0):
    from transformers import set_seed
    from adapters import get_adapter, output_dir_for
    import pandas as pd
    set_seed(seed)

    adapter = get_adapter(model_id=model_id, adapter_name=adapter_name)
    adapter.load(model_id)
    if out_csv is None:
        out_csv = os.path.join(output_dir_for("results", adapter), "text_only.csv")

    def ask_text(prompt, max_new_tokens=80):
        inputs = adapter.build_inputs(None, prompt)   # image=None -> text-only (see adapters.py)
        return adapter.generate_answer(inputs, max_new_tokens=max_new_tokens)

    rows = []
    print(f"=== TEXT ONLY (model: {adapter.short_name}): no image, just the hand positions "
          "described in words ===")
    for m_num, h_num, correct in CASES:
        p = (f"On an analog clock, the long minute hand is pointing at the {m_num} "
             f"and the short hour hand is pointing at the {h_num}. "
             f"What time is it? Answer only in H:MM format.")
        reply = ask_text(p)
        m = TIME_RE.search(reply)
        correct_h, correct_m = (int(x) for x in correct.split(":"))
        got_ok = bool(m) and int(m.group(1)) % 12 == correct_h % 12 and int(m.group(2)) == correct_m
        rows.append({"case": "hand_to_time", "minute_hand_num": m_num, "hour_hand_num": h_num,
                     "correct_answer": correct, "raw_answer": reply, "correct": got_ok})
        print(f"minute-hand@{m_num}, hour-hand@{h_num} -> correct {correct:>6} | model: {reply!r}")

    print("\n=== PURE ARITHMETIC: does it know number -> minutes at all? ===")
    for m_num in ARITHMETIC_CASES:
        p = (f"On an analog clock face, if the minute hand points exactly at the number {m_num}, "
             f"how many minutes past the hour is it? Answer with just the number.")
        reply = ask_text(p, 20)
        correct_val = (m_num % 12) * 5
        m = NUM_RE.search(reply)
        got_ok = bool(m) and int(m.group(1)) == correct_val
        rows.append({"case": "arithmetic", "minute_hand_num": m_num, "hour_hand_num": None,
                     "correct_answer": str(correct_val), "raw_answer": reply, "correct": got_ok})
        print(f"number {m_num:>2} -> correct {correct_val:>2} | model: {reply!r}")

    results_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    hand_to_time_acc = results_df[results_df["case"] == "hand_to_time"]["correct"].mean()
    arithmetic_acc = results_df[results_df["case"] == "arithmetic"]["correct"].mean()
    print(f"\nhand-to-time accuracy: {hand_to_time_acc:.1%}   arithmetic accuracy: {arithmetic_acc:.1%}")
    print(f"Saved to '{out_csv}'.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None,
                         help="default: results/<model_short_name>/text_only.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(model_id=args.model_id, adapter_name=args.adapter, out_csv=args.out_csv, seed=args.seed)
