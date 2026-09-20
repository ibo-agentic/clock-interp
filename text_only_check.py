"""
text_only_check.py -- Can the model do the 'number on face -> minutes'
conversion when there is NO image? If yes, perception and arithmetic are each
fine on their own, and the failure is in linking them.
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

set_seed(0)
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()

# (minute-hand number, hour-hand number, correct answer)
cases = [
    (10, 8,  "8:50"),
    (2,  7,  "7:10"),
    (6,  5,  "5:30"),
    (4,  10, "10:20"),
    (12, 3,  "3:00"),
    (9,  1,  "1:45"),
    (7,  11, "11:35"),
    (3,  6,  "6:15"),
]

@torch.no_grad()
def ask_text(prompt, max_new_tokens=80):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

print("=== TEXT ONLY: no image, just the hand positions described in words ===")
for m_num, h_num, correct in cases:
    p = (f"On an analog clock, the long minute hand is pointing at the {m_num} "
         f"and the short hour hand is pointing at the {h_num}. "
         f"What time is it? Answer only in H:MM format.")
    print(f"minute-hand@{m_num}, hour-hand@{h_num} -> correct {correct:>6} | model: {ask_text(p)!r}")

print("\n=== PURE ARITHMETIC: does it know number -> minutes at all? ===")
for m_num in [2, 4, 6, 7, 9, 10, 12]:
    p = (f"On an analog clock face, if the minute hand points exactly at the number {m_num}, "
         f"how many minutes past the hour is it? Answer with just the number.")
    print(f"number {m_num:>2} -> correct {(m_num % 12) * 5:>2} | model: {ask_text(p, 20)!r}")
