| metric | qwen2.5-vl-3b | qwen2.5-vl-7b | internvl3-2b | internvl3-8b |
|---|---|---|---|---|
| Behavior: exact accuracy (data/, parsed replies) | 1.8% [0.9%, 3.4%] n=500 | 12.4% [9.8%, 15.6%] n=500 | 8.8% [6.6%, 11.7%] n=498 | 9.8% [7.5%, 12.7%] n=500 |
| Behavior: hour accuracy | 26.4% [22.7%, 30.4%] n=500 | 70.4% [66.3%, 74.2%] n=500 | 58.0% [53.7%, 62.3%] n=498 | 63.2% [58.9%, 67.3%] n=500 |
| Behavior: minute accuracy | 2.4% [1.4%, 4.1%] n=500 | 16.4% [13.4%, 19.9%] n=500 | 11.4% [8.9%, 14.5%] n=498 | 11.2% [8.7%, 14.3%] n=500 |
| Behavior: parse rate | 100.0% [99.2%, 100.0%] n=500 | 100.0% [99.2%, 100.0%] n=500 | 99.6% [98.6%, 99.9%] n=500 | 100.0% [99.2%, 100.0%] n=500 |
| Describe [PRIMARY]: minute hand described EXACTLY right | 58.5% [54.1%, 62.9%] n=480 | 83.8% [80.2%, 86.8%] n=480 | 52.3% [47.8%, 56.7%] n=480 | 69.8% [65.5%, 73.7%] n=480 |
| Describe [PRIMARY]: hour hand described right (exact) | 31.2% [27.3%, 35.5%] n=480 | 59.4% [54.9%, 63.7%] n=480 | 52.7% [48.2%, 57.1%] n=480 | 66.9% [62.5%, 70.9%] n=480 |
| Describe [PRIMARY]: BOTH hands described exactly right | 23.1% [19.6%, 27.1%] n=480 | 50.4% [46.0%, 54.9%] n=480 | 30.0% [26.1%, 34.2%] n=480 | 45.0% [40.6%, 49.5%] n=480 |
| Describe [PRIMARY]: final answer WRONG given both hands described EXACTLY right | 55.0% [45.7%, 63.9%] n=111 | 0.0% [0.0%, 1.6%] n=242 | 0.7% [0.1%, 3.8%] n=144 | 7.4% [4.6%, 11.7%] n=216 |
| Describe [secondary]: minute hand described right (+/-1 number) | 82.3% [78.6%, 85.4%] n=480 | 85.4% [82.0%, 88.3%] n=480 | 59.8% [55.3%, 64.1%] n=480 | 84.2% [80.6%, 87.2%] n=480 |
| Describe [secondary]: final answer WRONG given both hands described right (+/-1 number) | 66.0% [58.0%, 73.2%] n=147 | 3.2% [1.6%, 6.2%] n=250 | 8.7% [5.3%, 14.1%] n=161 | 26.5% [21.6%, 32.0%] n=272 |
| Describe: reply fully parsed (parse-fit check, original regexes) | 100.0% [99.2%, 100.0%] n=480 | 98.8% [97.3%, 99.4%] n=480 | 100.0% [99.2%, 100.0%] n=480 | 95.8% [93.7%, 97.3%] n=480 |
| Describe [robustness: tolerant hour parse]: hour hand described right | 31.2% [27.3%, 35.5%] n=480 | 59.8% [55.3%, 64.1%] n=480 | 52.7% [48.2%, 57.1%] n=480 | 70.6% [66.4%, 74.5%] n=480 |
| Describe [robustness: tolerant hour parse]: BOTH hands exactly right | 23.1% [19.6%, 27.1%] n=480 | 50.8% [46.4%, 55.3%] n=480 | 30.0% [26.1%, 34.2%] n=480 | 48.8% [44.3%, 53.2%] n=480 |
| Describe [robustness: tolerant hour parse]: final answer WRONG given both hands EXACTLY right | 55.0% [45.7%, 63.9%] n=111 | 0.0% [0.0%, 1.6%] n=244 | 0.7% [0.1%, 3.8%] n=144 | 6.8% [4.3%, 10.8%] n=234 |
| Describe [robustness: tolerant hour parse]: WRONG given both right (+/-1 number) | 66.0% [58.0%, 73.2%] n=147 | 3.2% [1.6%, 6.1%] n=252 | 8.7% [5.3%, 14.1%] n=161 | 24.8% [20.2%, 30.1%] n=290 |
| Describe [robustness: tolerant hour parse]: reply fully parsed | 100.0% [99.2%, 100.0%] n=480 | 99.6% [98.5%, 99.9%] n=480 | 100.0% [99.2%, 100.0%] n=480 | 100.0% [99.2%, 100.0%] n=480 |
| Text-only gate: b2 minute accuracy (no image) | 67.6% [63.0%, 71.8%] n=432 | 97.0% [94.9%, 98.2%] n=432 | 49.8% [45.1%, 54.5%] n=432 | 94.9% [92.4%, 96.6%] n=432 |
| Conversion: minute acc., no image (re-run of text-only) | 67.6% [63.0%, 71.8%] n=432 | 97.0% [94.9%, 98.2%] n=432 | 49.8% [45.1%, 54.5%] n=432 | 94.9% [92.4%, 96.6%] n=432 |
| Conversion: minute acc., matching clock | 61.8% [57.1%, 66.3%] n=432 | 99.3% [98.0%, 99.8%] n=432 | 71.8% [67.3%, 75.8%] n=432 | 100.0% [99.1%, 100.0%] n=432 |
| Conversion: minute acc., blank image | 72.0% [67.6%, 76.0%] n=432 | 100.0% [99.1%, 100.0%] n=432 | 45.6% [41.0%, 50.3%] n=432 | 98.4% [96.7%, 99.2%] n=432 |
| Conversion: minute acc., mismatched clock (follows TEXT) | 66.7% [62.1%, 70.9%] n=432 | 99.8% [98.7%, 100.0%] n=432 | 43.8% [39.1%, 48.5%] n=432 | 96.3% [94.1%, 97.7%] n=432 |
| Conversion: mismatched clock, answer minute follows IMAGE | 2.8% [1.6%, 4.8%] n=432 | 0.0% [0.0%, 0.9%] n=432 | 15.3% [12.2%, 19.0%] n=432 | 2.5% [1.4%, 4.5%] n=432 |
| Conversion: mismatched clock, answer hour follows IMAGE | 0.7% [0.2%, 2.0%] n=432 | 0.0% [0.0%, 0.9%] n=432 | 7.9% [5.7%, 10.8%] n=432 | 0.7% [0.2%, 2.0%] n=432 |
| Conversion: parse rate (all conditions) | 100.0% [99.8%, 100.0%] n=1728 | 100.0% [99.8%, 100.0%] n=1728 | 99.8% [99.5%, 99.9%] n=1728 | 99.1% [98.6%, 99.5%] n=1728 |
| McNemar p, no-image vs matching clock (minute) | p=0.02 (only-text-correct=66 / only-image-correct=41) | p=0.013 (only-text-correct=2 / only-image-correct=12) | p=3.4e-15 (only-text-correct=29 / only-image-correct=124) | p=4.8e-07 (only-text-correct=0 / only-image-correct=22) |
| McNemar p, no-image vs blank (minute) | p=0.0013 (only-text-correct=7 / only-image-correct=26) | p=0.00024 (only-text-correct=0 / only-image-correct=13) | p=0.054 (only-text-correct=48 / only-image-correct=30) | p=0.0059 (only-text-correct=6 / only-image-correct=21) |
| McNemar p, no-image vs mismatched clock (minute) | p=0.77 (only-text-correct=54 / only-image-correct=50) | p=0.00049 (only-text-correct=0 / only-image-correct=12) | p=0.048 (only-text-correct=93 / only-image-correct=67) | p=0.41 (only-text-correct=15 / only-image-correct=21) |

Rates are shown as value [95% Wilson CI] n=count. Wilson CIs treat trials as independent; in the conversion and text-only tests the prompts are crossed (hours x lead-ins), so those CIs are somewhat optimistic.

Known protocol differences (not hidden): InternVL3's chat template injects NO default system prompt while Qwen2.5-VL's does; InternVL images are one 448x448 tile = 256 tokens, Qwen's 512x512 clocks use ~324 tokens; max_new_tokens is 16 (behavior) and 120 (describe) for every model; for text-only / conversion it is 32 for every model EXCEPT any run whose run_config.json says otherwise (InternVL3-8B was re-run at 64 after 32 cut off replies; no Qwen or InternVL3-2B reply reached 32 tokens, so those results are unaffected); dtype is float16 for all four (see each run_config.json).
Describe rows: the +/-1-number tolerance and the regex parsing are Qwen-tuned (describe_check.py); see the 'reply fully parsed' row for how well they fit each model's wording.
