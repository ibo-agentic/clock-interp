# FINDINGS (living summary; paper numbers come from here)

**Last updated:** 2026-11-08 (after D3-Q and notebook E). **Update rule:** after every run, edit the tables and the "Status" block, add one line to the update log, and never change a number without its source file.
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
| **E3d Qwen-3B** (bar: leaks 40 -> <= 28, acc +4; random-span < 2 pts) | leaks 40 -> 13 (-67.5%); accuracy +9.3 pts CI [+5.8, +13.0]; random-span control -0.9 pts (leaks 44); destruction control 10.9% accuracy | **SUPPORTED** (`M_bind_supported=true`) |
| E3d Qwen-7B (specificity control, `min` mode) | leaks 6 -> 2; accuracy +1.4 pts CI [+0.5, +2.5]; random-span 0.0 | not supported (below the 4-pt bar; ceiling model, tiny but non-zero effect) |
| E3d family, Holm over {3B, 2B, 7B} | adjusted p (accuracy) 0.0003 / 0.0003 / 0.031; (leaks) 0.0003 / 0.0003 / 0.12 | no verdict changes |
| E3b N-span, Qwen-3B | supported at 11/12 layers, shift up to +12.3 nats (layer 21); H-span 0/12 | SUPPORTED (about the N token, not the leak) |
| E3c restoration, Qwen-3B | 1 of 48 cells: layer 21, hour token, 52.8% of 85 wrong runs vs best control 16.7% | SUPPORTED (single cell; the 2B is not supported) |
| E3b N-span (minute formed from the N token), InternVL3-2B | shift +11 to +13 nats at 8/9 layers; controls <= 1.9 | SUPPORTED (statement about the N token, not the leak) |
| E3c restoration, InternVL3-2B | best cell 48.8% (layer 15, hour token) vs control 23.8% | NOT SUPPORTED (below 50% bar; exploratory) |
| **E1 InternVL3-2B**, machinery check | blocked-all vs text-only: query set B 6.7 pts, A 12.5 pts (both > 5) | **INVALID** => every 2B E1 result is EXPLORATORY ONLY |
| E1 H-E1-1 / H-E1-2 / H-E1-3 (2B, exploratory status) | best window w16-20: drop 10.6 pts (48% of T=22.0), below the 11 pts / 50% line => evidence distributed; mismatched follow-image 54 -> 36 (-33%) => not met; final-4 window -1.2 pts => within 3 | not supported / not supported / within 3 pts (all exploratory) |
| E1 Qwen-7B, InternVL3-8B (H-E1-3 specificity) | gate G5 failed (block changed 0/12 and 3-7/12 generations); no cells ran | not evaluated |
| E2 (patching): F1a 2B, F1b Qwen-7B, F2 8B | not run | - |

## 3. Exploratory / descriptive
- **E3d control-adjusted (2B, exploratory):** H-block minus random-span block = accuracy **+19.4 pts** (CI +14.4, +24.5), leak rate **-19.7 pts** of trials (CI -23.6, -16.0). Effect sits in layers 16-23 (windows 16-20: 87 leaks, 20-24: 37; cumulative-to-end 28). Blocking the N token collapses accuracy to 7%.
- **E3a (2B logit lens):** gold-minus-leak log-odds at the final layer: right +5.8, leak-error -2.7, other-wrong +1.2, hour 12 -4.9; leak-error trials favour the leak minute from about layer 6.
- **E3e other models (descriptive):** 2B 66.7% wrong w/o image vs 0.7% with (its own description is not enough: it needs the image); 7B 0.8% vs 0.0% (ceiling); 8B 13.4% vs 7.4%.
- **E3f hour 12 (minute accuracy, H=12 vs H!=12 gap, numeral -> word form):** Qwen-3B 33.0 -> 37.6 pts (gap stays: "12 = 00"); Qwen-7B 21.6 -> **7.7** (numeral/tokenisation-form rule SUPPORTED at rp 1.0; at rp 1.05: 19.6 -> 13.2, not); InternVL3-2B 40.0 -> 23.4 (gap stays).
- **Gates:** all pass. InternVL3-2B G2-own = 0.95 exactly (deterministic across two runs); Qwen at rp 1.0: 3B G2-own 0.97 / G8b 1.0, 7B 1.0 / 1.0. At rp 1.05 the Qwen gates failed (G8b 0.72 / 0.94) because `generate` applied the penalty and the no-cache loop did not.

- **E3d depth profile (exploratory):** Qwen-3B leaks are removed by layers 24-28 of 36 (window removes 24 of 27 removable leaks; blocking only layers 24-36 gives the same 13 leaks as blocking all 36), relative depth 0.67-0.78. InternVL3-2B: layers 16-24 of 28 (relative depth 0.57-0.86; window 20-24 alone removes 80 of 89). Both are late-middle layers; the 3B window is narrower. 3B control-adjusted (H_all - random_all): accuracy +10.2 pts [+6.9, +13.4], leak rate -7.2 pts of trials [-10.2, -4.4]. 7B: +1.4 [+0.5, +2.5].
- **E3a Qwen-3B (descriptive):** final-layer gold-minus-leak log-odds right +6.9, leak-error +2.0 (still favours gold on average; the 2B is -2.7), other-wrong +3.5, hour 12 -0.04 (vs +6.1 for other hours).
- **E1 (all exploratory/descriptive; see the machinery caveat above).** InternVL3-2B, query set B: intact 71.8%, text-only 49.8%, blank 45.6%. Drop by 4-layer window: 0-4 6.9, 4-8 4.2, 8-12 3.9, 12-16 8.1, **16-20 10.6**, 20-24 4.2 (n.s.), 24-28 -1.2 pts; blocking the last 4 layers does nothing. Late half (14-28) 12.7 pts, early half 30.8. With B the all-layer block gives identical correctness for matching, blank and mismatched images (432/432), so the 6.7-pt gap to text-only is not leftover image content. With query set A the question tokens still read the image, so A cannot match text-only by construction. E1 w16-20 overlaps the E3d hour-token layers (16-23) but E3d is strongest at 20-24, where the E1 window is not significant. Qwen-3B (machinery valid): the image lowers accuracy (54.2% vs 64.4%); blocking layers 16-20 / 24-28 raises it by 6.7 / 5.8 pts, blocking layers 0-4 lowers it 5.3 pts; the 24-28 window coincides with the E3d hour-token window.

## 4. rp 1.0 vs rp 1.05 sensitivity (Qwen); flagged = changes > 3 pts or flips significance
- Qwen-3B: wrong-given-both-exact **70.5% vs 55.0%** (set 122 vs 111); text-only 64.4 vs 67.6; matching clock 54.2 vs 61.8; blank 66.2 vs 72.0; mismatched 56.0 vs 66.7. McNemar vs text-only: blank p=0.23 (was 0.001), mismatched p=0.001 (was 0.77) - both **flip**. E3e verdict flips language-side -> mixed. H=12 collapse is much weaker at 1.0 (accuracy 39% vs 19%). Leaks 40 (was 46).
- Qwen-7B: all behaviour numbers within 1 pt; matching-clock McNemar p=0.065 vs 0.013 (11 discordant trials); E3f word-form verdict flips (above).
- Not flagged: image behaviour (exact 1.8/12.6), describe hour/minute rates, conversion follows-image rates.

## 5. Status and next
Done: E0, E3a-f, text-light pilot, InternVL3-8B 128-token re-run, Qwen at rp 1.0 (notebook Q), D3-IVL, D3-Q, E1 for InternVL3-2B and Qwen-3B, **E2 for InternVL3-2B (F1a, section 5a)**, Gemma G0b (section 6). Open: E1 for Qwen-7B and InternVL3-8B (verify gate G5 failed; not re-run; H-E1-3 7B/8B part not evaluated). Built, not yet run: E2 F1b (Qwen-7B, descriptive) and F2 (InternVL3-8B, descriptive); Qwen-3B dropped from E2 (deviations log). Gemma: 4B works in bf16 and fp32; 12B bf16 hit a vision-tower out-of-memory on GPU 0 (untested fix); Gemma pipeline runs not started, awaiting owner decision. Analysis scripts: analysis/analyze_E1.py, analyze_E2.py, analyze_F1a_extra.py, analyze_G0b.py.

### 5a. E2, InternVL3-2B (notebook F1a; 100 pairs; gates V1-V4 pass; 2.79 h)
- Standard prompt (confirmatory): the pre-registered "minute-hand region" rule is NOT supported (minute-hand patch +10 pts, Holm p 0.04 at layers 0 and 14, but not above the other-region patch: difference +1 to +4 pts, CIs include 0); the cue/consistency rule is also not supported (minute-hand upper bound 16 pts). All-token patch +16 pts. Neither pre-registered account is picked out by the standard variant.
- Text-light (minute only in the image; positive control): all-token +34-36 pts, minute-hand tokens alone +19-21 pts at layers 0-17 (Holm p 0.006) with 13 of 256 tokens, about 0.55-0.60 of the all-token effect; descriptive minute-hand minus other-region +10 to +13 pts. Effects vanish at layer 24. This is direct, concentrated minute evidence in the hand tokens when the text does not supply the minute.
- E2 does not test the binding-cue/leak-suppression account directly (that is E3d); it shows the image tokens causally carry minute content, not that the cue account is wrong. Overlays to inspect: `results/mech_e2/internvl3-2b/overlays/`.

## 6. Gemma 3 smoke test (notebook G0, 2026-11-08; fp16, T4 x2, transformers 5.20.2)
| | gemma-3-4b-it | gemma-3-12b-it |
|---|---|---|
| Verdict | **FAIL** | **FAIL** |
| Image tokens / pixel_values | 256 / (1,3,896,896) OK | 256 / (1,3,896,896) OK |
| Single BOS, token_type_ids consistent (adapter checks) | no problem reported | no problem reported |
| Logits finite | **NO**, for all 3 image prompts AND all 3 text-only prompts | **NO**, same |
| Raw answers | all empty (''), 11/11 prompts | all empty, 11/11 |
| Describe-regex parse | 0/5 (empty replies) | 0/5 |
| Peak GPU memory (cuda:0 / cuda:1) | 5.9 / 2.7 GiB | 13.6 / 9.7 GiB (fits, ~1.4 GiB headroom on cuda:0) |
| Step wall time incl. download + 11 generations | 2.5 min | 5.6 min |
Reading: non-finite logits already on text-only prompts, so the problem is in the language model path, not the vision tower. The cause (true fp16 activation overflow vs an attention-implementation / masking NaN) is NOT yet diagnosed. No Gemma behaviour numbers exist. Decision (2026-11-08): test bf16 (4B, 12B) and fp32 (4B only) plus an fp16 diagnostic in notebook G0b before choosing; Gemma would run in native bf16 while Qwen/InternVL stay fp16 (deviations log, third entry of 2026-11-08).
**Paper note:** InternVL3-2B/8B-hf use a Qwen2 language backbone (config: model_type qwen2, 28 layers, hidden 1536 / 3584 = Qwen2.5-1.5B / 7B), so three of the four current models share the Qwen2.5 LLM lineage; Gemma 3 (independent LLM) is the real cross-family generality test.

### 6b. G0b result (2026-11-08)
- fp16 diagnostic (4B): true fp16 overflow in the language model, not an attention problem: `sdpa` and `eager` give identical results; the residual stream reaches 51392 at decoder layer 4 (fp16 max 65504) and decoder layer 5 produces the first non-finite value (1 inf, no nan) while its own sub-module outputs are finite (the interpretation, not directly shown: the residual addition overflows).
- 4B bf16 PASS, 4B fp32 PASS (table in `results/gemma/G0b_report.txt`): bf16 7.8/2.7 GiB, 0.77 s text-only, 4.69 s with image; fp32 11.5/5.4 GiB, 0.72 s, 3.53 s. bf16 is emulated on the T4 and slower for the image prefill. 12B bf16: CUDA OOM in the SigLIP vision tower on GPU 0 (no result).
- describe_check.py regexes parse 1/5 (bf16) or 2/5 (fp32) Gemma replies (markdown bold); tolerant-parse columns need approval.

### 6c. G0c result (2026-11-09; zip `mech_G0c_results.zip`, unzipped to `match_G0c_results/`)
- gemma-3-12b-it bf16 PASSED on the FIRST cap, `0:9.5GiB,1:13.5GiB` (ladder tries 2 and 3 were not needed). Runtime 7.2 min (guess was 15-40); load 198 s (includes the first download).
- Device map: {GPU0: 19 module groups, GPU1: 32, **disk: 3**}. Three module groups were offloaded to DISK ("Some parameters are on the meta device because they were offloaded to the disk"). The log records only counts, not which modules. Peak memory: cuda:0 11.3 GiB, cuda:1 13.5 GiB (= the cap).
- All logits finite (3 image prompts, 3 text-only prompts). Raw answers: image prompts `10:10`->'10:10', `3:45`->'09:20', `7:20`->'07:20'; text-only: '09:20', '35', '35' (7x5). Smoke clocks at 6 timings: text-only '04:35', with image '06:00' (identical across repeats).
- Seconds per generation (5 new tokens, 3 repeats): text-only 2.06 s (prefill 0.77 s), with image 7.87 s (prefill 6.84 s). 4B bf16 for comparison (G0b): 0.77 s and 4.69 s.
- Parsing: the 5 describe replies are markdown-formatted ("Minute hand:** The long hand is pointing at the number 10"). Re-parsed with the current repo parser (markdown normalization on): time 5/5, hour (tolerant) 5/5, but strict `parse_reply` minute/hour numbers only 1/5 (12:00). Cause: the strict regexes need "minute hand is pointing at", Gemma writes "Minute hand: The long hand is pointing at ...". The markdown stripping does NOT fix this wording mismatch; the log line "1/5 fully parsed" was printed by the zip's `gemma_diag.py` and is the same with the normalizer on. No tolerant MINUTE parse exists yet (only the hour one). Not changed; needs a decision.

### 6d. Describe parser (2026-11-09)
- Added a tolerant minute-hand parse (`parse_minute_tolerant`, columns `said_minute_number_tolerant`, `minute_desc_ok_tolerant`) beside the tolerant hour columns; strict columns unchanged and still primary.
- Switching all models to the tolerant parser is NOT a no-op (`results/describe_parser_noop_check.txt`): 88 cell differences over 12 saved describe CSVs, all hour parses ("pointing just past 12"); headline file affected: InternVL3-8B hour_desc_ok 66.9% -> 70.6%. Qwen-3B/7B and InternVL3-2B headline files identical. **Decision (2026-11-09, project owner): tolerant parse = PRIMARY for all six models, strict = sensitivity table** (deviations log, sixth entry; made before any Gemma describe result was seen). `compare_families.py` updated; InternVL3-8B: hour right 66.9% -> 70.6%, wrong-given-both-exact 7.4% -> 6.8%; other three headline files identical. E3 clock-set selection and its scripts stay on the strict columns. 20 recovered InternVL3-8B replies: `results/describe_recovered_internvl3-8b.txt` (18/20 hour numbers correct).
- G2a/G2b notebooks (12B) built, not run; launch after G1's gate and first steps. Disk-offload device map goes to run_config.json.

## Update log
- 2026-11-09: tolerant describe parse made PRIMARY for all six models (sixth deviations entry); compare_families.py updated (strict = sensitivity rows); --recompute_tolerant covers minute columns.
- 2026-11-09: tolerant minute parse added (additive); parser-switch no-op check = NOT a no-op (section 6d); G2a/G2b built; fifth deviations entry written before any G1 describe result.
- 2026-11-09: G0c incorporated (12B bf16 passes at cap 0:9.5GiB,1:13.5GiB with 3 module groups on disk); section 6c added. G2 plan drafted, not built.
- 2026-11-08: decisions logged (E2 verdicts as computed; Gemma bf16 + 4B fp32 cross-check; markdown-stripping parse step for all models, verified a no-op on 54,810 stored replies; G0c and G1 built, not run; priority G0c, G1 before F1b, F2).
- 2026-11-08: F1a (E2, InternVL3-2B) and G0b (Gemma precision) incorporated; sections 5a and 6b added.
- 2026-11-08: notebook G0b (Gemma precision: fp16 diagnostic, bf16 4B/12B, fp32 4B) and `gemma_diag.py` built; `CLOCK_DTYPE` for Gemma adapters; no results yet.
- 2026-11-08: G0 Gemma smoke: both models fail in fp16 (non-finite logits).
- 2026-11-08: E2 split into F1a/F1b/F2, Qwen-3B dropped, V3 informational for 7B/8B (deviations log); Gemma 3 adapter built, notebook G0 added (no results).
- 2026-11-08: D3-Q (E3d Qwen-3B supported; 7B control) and E (E1: 2B machinery check invalid; 7B/8B gate failed) incorporated.
- 2026-11-07: created; D2'', Q and D3-IVL incorporated.
- 2026-11-07: E1/E2 code, notebooks and analysis scripts built (no results yet).
