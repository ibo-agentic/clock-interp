# Plan: E1 (windowed image knockout) and E2 (cross-clock image-token patching)

Status: BUILT 2026-11-07 (mech_e1.py, mech_e2.py, notebooks E/F1/F2, analysis/analyze_E1.py + analyze_E2.py); decisions D1-D4 resolved and logged (deviations log 2026-11-07 third + fourth entries). Not yet run on real weights. Pre-registration = `PREREG_MECHANISM.md` Sections 5-6; anything below that differs from it goes into the deviations log BEFORE the run.
Headline decoding = greedy, `repetition_penalty 1.0` (Qwen: `--rep_penalty 1.0` / `CLOCK_REP_PENALTY=1.0`). Run times are guesses.

## 0. Models and roles (after your decision)
| | E1 | E2 |
|---|---|---|
| InternVL3-2B | **confirmatory** (H-E1-1/2/3) | **confirmatory** (minute-hand-region rule, positive control for the text-light variant) |
| Qwen-2.5-VL-7B, InternVL3-8B | all-windows cell only (H-E1-3 specificity, < 3 pts) | descriptive, same procedure as the 2B (no pre-registered rule for them; to be logged as descriptive) |
| Qwen-2.5-VL-3B | exploratory (9 windows) | **exploratory** (text-light floor 0.093 < 0.15, so the pre-declared floor rule applies) |

Note for the 3B at rp 1.0: the image effect is negative (matching clock 54.2% vs text-only 64.4%), so the prereg numbers "text-only 67.6%, blocked >= 62.6%" are stale; the machinery check is re-stated on the rp 1.0 text-only run (64.4%, tolerance 5 pts). Nothing else changes.

## 1. What can be reused (so the new code is small)
- `knockout.py` (existing, Qwen-3B only): attention-knockout hook for text-query -> image-key blocking that works for **prefill and cached decode steps**. Plan: generalise through `adapters.py` (`image_token_positions`, `decoder_layers`) and add the query sets, windows and fp32 eager (`mech_lib.eager_attention`).
- `mech_lib.py`: fp32 eager attention (fixed D2'), cluster bootstrap, sign-flip, Holm, label taxonomy, `capture/patch_residuals` (multi-GPU safe).
- `intervene.py` (Experiment A): `build_pairs` (seed 0, min gap), `position_masks`, `make_matched_noise`, `make_random_noise_image`, `teacher_forced_minute_logprob`, same-minute partner lookup, `image_features_patched` (embedding-level all-token patch for the gate).
- `mech_e3.py` pattern: Ctx, verify gate first, status/zip notebook skeleton, `--rep_penalty`.
New files: `mech_e1.py`, `mech_e2.py` (+ small additions to `mech_lib.py`), `tests/test_e1_offline.py`, `tests/test_e2_offline.py`, notebooks E (E1) and F (E2) built like D3-IVL.

## 2. E1 design (notebook E)
- **Cells.** matching_clock: 4-wide sliding windows (28 layers -> 7; 3B 36 layers -> 9), cumulative-from-L-to-end, all layers; blank and mismatched_clock: all layers, early half, late half, final 4. Two query sets, each cell run for both. 432 trials per cell (same trials as the conversion test).
- **Query sets (needs your confirmation, see D2):** B = every position after the last image token (question text + generated tokens), A = the last prompt token and the generated tokens only (the positions that produce the answer). Image tokens are never blocked as queries (image->image attention intact).
- **Decoding.** Cached generation with the hook active in prefill and every decode step (as `knockout.py`), max_new_tokens 32, greedy, rp 1.0, eager fp32 inside the context. Faster than the no-cache loop (which would re-run the vision tower every step).
- **Gate (per model, before any windowed cell):** (1) empty block set == intact generations exactly (50/50); (2) hook reaches every decoder layer (calls == n_layers); (3) eager-vs-default agreement (see D1); (4) full-prompt guard; (5) **machinery check** (blocked-all matching_clock vs the model's own text-only run, 432 trials, tolerance 5 pts, per query set): outcome reported whether it passes; a fail makes all E1 results of that model/query set exploratory.
- **Analysis (script `analysis/analyze_E1.py`):** per window drop vs intact on matching_clock, T = intact - text-only (2B ~22.2 pts), Holm over the 7 windows, cluster bootstrap on (N,H) cells; H-E1-1 (a window removes >= 50% of T, drop >= 11 pts, CI excl. 0); H-E1-2 (the best matching_clock window lowers the mismatched follow-image rate, baseline 54 trials = taxonomy label `follows_image`, checked: 54 of 432 for the 2B, by >= 50% relative, CI excl. 0); H-E1-3 (final-4 window < 3 pts; 7B/8B all-windows < 3 pts); cumulative curves descriptive.
- **Compute (guess):** ~46 cells x 432 trials ~ 20k generations for the 2B, ~0.2-0.3 s each cached -> 1.5-2.5 h; 7B/8B/3B add ~2-3 h in total -> ~4 h as in the prereg guess. First step of the notebook times 20 generations and prints the extrapolation.
- **Offline tests:** tiny InternVL (images work: random vision tower) for the full path incl. cached-decode blocking vs an uncached reference; the Qwen image path (processor, mrope + `mm_token_type_ids`) CANNOT be tested offline with the tiny Qwen (no processor) - covered by the hook-level test and by the gate on Kaggle.

## 3. E2 design (notebook F)
- **Pairs.** Text fixed to clock A; image A shown; image-token residuals of A replaced by those of clock B at layer L. B: same hour number, minute number >= 3 away from N_A (circular), deterministic (seed 0, `build_pairs`). **Pair count N = 100 per model (proposal, D3);** layers = embedding output + 7 layers at 1/8..7/8 of depth (fixed now, before any run).
- **Token sets:** all image tokens, minute-hand tokens, hour-hand tokens, all other image tokens. Regions: rasterise the hand segments of A and of B (from `clocks.hand_angles` + drawing geometry, dilated by half a token) onto the token grid (InternVL 16x16, Qwen 18x18 at 512 px); union of A and B.
- **Controls:** (c1) same-minute B (another render, same N) = noise floor; (c2) norm-matched noise (`make_matched_noise`); (c3) text-light variant (hour number in the text, minute only in the image) = confirmatory E2 test for the 3B only if its floor passed (it does not: exploratory) and the positive control for the 2B.
- **Metrics:** P(answer minute = B's), P(stays A's), teacher-forced log p(B minute) - log p(A minute). Generation uses cached `generate`, rp 1.0.
- **Gate:** identity patch (B=A) changes 0/100 generations; embedding-level all-token patch reproduces the existing mismatched_clock follow-image rate within 3 pts per model (2B: 54/432 = 12.5%); **region masks overlaid on 20 images saved as PNGs for your eyes** (cannot be judged automatically).
- **Decisions rules** as pre-registered (2B: minute-hand patch >= +5 pts over same-minute control, CI excl. 0, and > other regions, Holm over layers x regions; cue account: only all/non-hand patching moves the answer, minute-hand upper bound < 3 pts; 3B text-light: +10 pts, floor rule).
- **Compute (guess):** per model 100 pairs x 8 layers x 4 token sets (~3,200 patched forwards) + controls ~ 1-1.5 h for the 2B, similar for 7B/8B -> ~3 h total with the prereg guess.
- **Offline tests:** tiny InternVL end-to-end (pairs, masks, patching, identity patch = 0 changes).

## 4. Decisions needed from you before any code (will be logged pre-run)
- **D1 (important):** the prereg E1 gate says "eager and SDPA attention agree to < 1e-3 on logits". In fp16 this cannot hold: measured max |delta log-prob| 0.05-0.08 on all models (G8c) with identical argmax. Proposal: replace with the G8c form (finite, max |delta log-prob| < 0.5 nat, argmax equal on >= 90% of 10 prompts), logged as a pre-run deviation. Without this the E1 gate would fail by construction.
- **D2:** confirm the query-set definitions (A = last prompt token + generated tokens; B = everything after the last image token).
- **D3:** E2 pair count 100 per model, and 7B/8B run as descriptive (no prereg rule).
- **D4:** order: E1 first (prereg), then E2; they are independent, so D3-Q, E and F notebooks can also run in parallel on separate Kaggle sessions.

## 5. Sequence
1. Log D1-D3 (pre-run) -> 2. write `mech_e1.py` + offline test + notebook harness -> 3. you run notebook E (gate output first: send the gate table before the long run, as with D2) -> 4. analysis script + `FINDINGS.md` update -> 5. same for E2.
