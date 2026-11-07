# FINDINGS (living summary; paper numbers come from here)

**Last updated:** 2026-11-07 (after D2'', notebook Q, D3-IVL). **Update rule:** after every run, edit the tables and the "Status" block, add one line to the update log, and never change a number without its source file.
**Headline decoding:** plain greedy, `repetition_penalty = 1.0`, all four models (decided 2026-11-06; deviations log). Qwen checkpoints ship 1.05; the old Qwen results are the *sensitivity* set (`*_rp105*`).
**Provenance:** tables = `results/comparison/four_model_table.md`; sensitivity = `results/comparison/sensitivity_rp1.0_vs_rp1.05.csv`; mechanism = `results/mech_e3/Q_D3IVL_report.txt`, `D2prime_D4_report.txt`. Pre-registration + deviations log: `PREREG_MECHANISM.md`.

## 1. Behaviour at greedy decoding (rp 1.0; n = 432 text/conversion trials, 480 describe images, 500 behaviour images)
| | Qwen-3B | Qwen-7B | InternVL3-2B | InternVL3-8B |
|---|---|---|---|---|
| Image behaviour: exact HH:MM | 1.8% | 12.6% | 8.8% | 9.8% |
| Describe: both hands exactly right | 25.4% | 50.6% | 30.0% | 45.0% |
| Final answer wrong given both hands exactly right | **70.5%** (n=122) | 0.0% (n=243) | 0.7% (n=144) | 7.4% (n=216) |
| Minute acc.: text only (no image) | 64.4% | 97.9% | 49.8% | 97.9% |
| ... matching clock | 54.2% | 99.5% | 71.8% | 100.0% |
| ... blank image | 66.2% | 100.0% | 45.6% | 98.8% |
| ... mismatched clock, follows text / follows image (minute) | 56.0% / 3.9% | 100% / 0.0% | 43.8% / 15.3% | 96.3% / 2.5% |
| Hour->minute leaks in text-only (of 432) | 40 | 6 | 117 | 2 |
| McNemar, text-only vs matching clock (only-text-right / only-image-right) | p=4.6e-5 (79/35, image hurts) | p=0.065 (2/9) | p=3e-15 (29/124, image helps) | p=0.004 (0/9) |
| Text-light pilot (E2 floor >= 0.15) | **0.093 (fails)** | 0.637 | 0.500 | 0.750 |

## 2. Confirmatory results, exactly as pre-registered
| Test | Result | Pre-registered verdict |
|---|---|---|
| **E3e Qwen-3B** (verbatim description, no image) | 122 clocks; wrong w/o image **57.4%**, with image 70.5%; paired diff -13.1 pts, CI [-23.8, -3.3] | **MIXED** (>=40% wrong, but CI not inside +/-15 pts; >25%, so not image-interference). Sensitivity rp 1.05: 111 clocks, 54.1% vs 55.0%, diff -0.9 [-10.8, +9.0] -> *language-side* |
| **E3d InternVL3-2B** (block answer->hour-token, all layers) | leaks 117 -> 28 (-76%); accuracy +22.5 pts CI [+17.4, +27.5]; random-span control accuracy **+3.0** [+1.2, +5.1], leaks 113 | **NOT SUPPORTED** (`M_bind_supported=false`): thresholds (>=30%, >=4 pts, CIs) met, control clause (< 2 pts) violated |
| E3d Qwen-3B (bar: leaks 40 -> <= 28, acc +4) | pending (notebook D3-Q) | - |
| E3d Qwen-7B (specificity control, `min` mode) | pending | - |
| E3b N-span (minute formed from the N token), InternVL3-2B | shift +11 to +13 nats at 8/9 layers; controls <= 1.9 | SUPPORTED (statement about the N token, not the leak) |
| E3c restoration, InternVL3-2B | best cell 48.8% (layer 15, hour token) vs control 23.8% | NOT SUPPORTED (below 50% bar; exploratory) |
| E1 (windowed knockout, 2B), E2 (patching) | not run | - |

## 3. Exploratory / descriptive
- **E3d control-adjusted (2B, exploratory):** H-block minus random-span block = accuracy **+19.4 pts** (CI +14.4, +24.5), leak rate **-19.7 pts** of trials (CI -23.6, -16.0). Effect sits in layers 16-23 (windows 16-20: 87 leaks, 20-24: 37; cumulative-to-end 28). Blocking the N token collapses accuracy to 7%.
- **E3a (2B logit lens):** gold-minus-leak log-odds at the final layer: right +5.8, leak-error -2.7, other-wrong +1.2, hour 12 -4.9; leak-error trials favour the leak minute from about layer 6.
- **E3e other models (descriptive):** 2B 66.7% wrong w/o image vs 0.7% with (its own description is not enough: it needs the image); 7B 0.8% vs 0.0% (ceiling); 8B 13.4% vs 7.4%.
- **E3f hour 12 (minute accuracy, H=12 vs H!=12 gap, numeral -> word form):** Qwen-3B 33.0 -> 37.6 pts (gap stays: "12 = 00"); Qwen-7B 21.6 -> **7.7** (numeral/tokenisation-form rule SUPPORTED at rp 1.0; at rp 1.05: 19.6 -> 13.2, not); InternVL3-2B 40.0 -> 23.4 (gap stays).
- **Gates:** all pass. InternVL3-2B G2-own = 0.95 exactly (deterministic across two runs); Qwen at rp 1.0: 3B G2-own 0.97 / G8b 1.0, 7B 1.0 / 1.0. At rp 1.05 the Qwen gates failed (G8b 0.72 / 0.94) because `generate` applied the penalty and the no-cache loop did not.

## 4. rp 1.0 vs rp 1.05 sensitivity (Qwen); flagged = changes > 3 pts or flips significance
- Qwen-3B: wrong-given-both-exact **70.5% vs 55.0%** (set 122 vs 111); text-only 64.4 vs 67.6; matching clock 54.2 vs 61.8; blank 66.2 vs 72.0; mismatched 56.0 vs 66.7. McNemar vs text-only: blank p=0.23 (was 0.001), mismatched p=0.001 (was 0.77) - both **flip**. E3e verdict flips language-side -> mixed. H=12 collapse is much weaker at 1.0 (accuracy 39% vs 19%). Leaks 40 (was 46).
- Qwen-7B: all behaviour numbers within 1 pt; matching-clock McNemar p=0.065 vs 0.013 (11 discordant trials); E3f word-form verdict flips (above).
- Not flagged: image behaviour (exact 1.8/12.6), describe hour/minute rates, conversion follows-image rates.

## 5. Status and next
Done: E0, E3a-f, text-light pilot, InternVL3-8B 128-token re-run, Qwen at rp 1.0 (notebook Q), D3-IVL. Built, not yet run: D3-Q (Qwen E3a-d), E (E1: InternVL3-2B confirmatory; 7B/8B all-layers specificity; 3B exploratory), F1/F2 (E2: 2B confirmatory, 7B/8B descriptive, 3B exploratory - fails the 0.15 floor). Analysis scripts ready: analysis/analyze_E1.py, analyze_E2.py.

## Update log
- 2026-11-07: created; D2'', Q and D3-IVL incorporated.
- 2026-11-07: E1/E2 code, notebooks and analysis scripts built (no results yet).
