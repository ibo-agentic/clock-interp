# PREREG_MECHANISM.md — pre-registration of the causal mechanism experiments (E1, E2, E3)

Status: written BEFORE any mechanism experiment was run on real weights. The author commits this file before the first
Kaggle run; the commit timestamp is the proof of order. Anything not stated here is exploratory and will be labelled so.
Later changes go ONLY in the "Deviations log" at the end (append-only, dated), never as edits above it.

## 0. What was already known when this was written (honest disclosure)

Seen already (so NOT confirmatory evidence for the hypotheses they motivated):
- Behavior, describe, text-only (`text_only_v2`) and conversion-test (`conversion_test`) results for Qwen2.5-VL-3B/7B and
  InternVL3-2B/8B (`results/comparison/four_model_table.md`), and the linear probes for Qwen-3B/7B and InternVL3-2B.
- E0 error taxonomy (`analysis/error_taxonomy.py`, outputs in `results/mechanism_E0/`), run on the existing CSVs. Counts of
  text-only errors in the 432-trial conversion test (the first number is the hour->minute "leak", minute = 5*(H mod 12)):

  | model | text-only correct / leak / adjacent / 0-30 default / other | matching clock: correct / leak | mismatched: follows-image |
  |---|---|---|---|
  | Qwen-3B | 292 / 46 / 35 / 30 / 29 | 267 / 19 | 9 (2.1%) |
  | Qwen-7B | 419 / 4 / 4 / 4 / 1 | 429 / 0 | 0 |
  | InternVL3-2B | 215 / 117 / 24 / 32 / 44 | 310 / 34 | 54 (12.5%) |
  | InternVL3-8B | 410 / 2 / 4 / 0 / 3 (+13 unparsed) | 432 / 0 | 7 |

  (Follows-image here is stricter than the table's 15.3%/2.8% because the taxonomy gives "leak" priority; both numbers are reported.)
- Qwen-3B and InternVL3-2B single-hand conversion is much better than the two-hand prompt (Qwen-3B a1 98%, a2 95% vs b2 68%).
- H=12 anomaly (text-only, minute accuracy): Qwen-3B 19% (vs 72% at H!=12), InternVL3-2B 19% (vs 53%), Qwen-7B 78% (vs 99%),
  8B 92% (vs 95%). The failure looks different per model: Qwen-3B gets the hour wrong too (hour acc 28%) and answers "X:00";
  InternVL3-2B writes "12:00" 32/36 times (hour right, minute collapsed to 00 = the leak value 12*5 mod 60); Qwen-7B writes "00:xx" for
  hour 12 in 105 replies — the scorer counts these as hour-correct (`hour_correct` compares hour mod 12; verified: all 105 are hour-correct,
  so this is NOT a scoring bug and NOT a model error) — but in 13/36 text-only H=12 trials it writes hour "01" (18 replies over all conditions),
  which is a real hour error (consistent with reading the digit-split "12" as "1"). Qwen-3B's 26 hour errors at H=12 (text-only) include 20 where
  the hour slot holds the MINUTE number N (base rate for H!=12: 33/396 = 8%), i.e. a binding error in the opposite direction (minute->hour), together
  with minute 00. So H=12 mixes leak and swap-type binding errors. All four tokenizers split "12" as [' ', '1', '2'] (3 tokens), identical
  to " 10" and " 11", which are NOT anomalous -> token length alone cannot explain H=12.
- Qwen-3B: clock pixels hurt vs blank (61.8% vs 72.0%, p=1.6e-5), blank helps vs text-only (72.0 vs 67.6, p=0.0013).

NOT seen: any result of attention knockout, activation patching, logit lens, swap patching, edge knockout or description
transplant on any model. `knockout.py` and `intervene.py` have only ever been run on fake models.

Consequence for claims: "the leak exists and the image reduces it" is descriptive (known). The experiments test WHERE and HOW.

## 1. Hypotheses (two mechanisms, not two routes — framing not locked)

- **M-bind (language-side binding failure):** in the two-hand prompt, small models mis-bind numbers to output slots: the hour number
  leaks into the minute slot (minute = 5*H), with 00/30 defaults and an H=12 collapse. It is present without any image.
- **M-img (image effects on binding/readout):** an image can (a) act as a consistency cue that suppresses the leak and/or (b) supply direct
  minute evidence. For InternVL3-2B both may operate; for Qwen-3B the matching clock adds new errors ("other" 29->71).
- The earlier labels "visual route" / "weak conversion" are NOT assumed; E1–E3 decide the final framing.

## 2. Common rules (all experiments)

- Models: primary contrast InternVL3-2B vs Qwen2.5-VL-3B; controls Qwen2.5-VL-7B and InternVL3-8B (expected to be image-robust).
- Prompt/data: the 432-trial conversion set (N=1..12 x H=1..12 x 3 lead-ins; `conversion_test.py`); same images (`conversion_images`), same
  scoring (`minute_correct`), greedy decoding, fp16, max_new_tokens 32 (8B: 128 after the re-run below).
- Unit of analysis = the (N,H) cell (144 cells; mean over its 3 lead-ins), because lead-ins are not independent replicates.
  Primary tests: paired sign-flip permutation test on cell-level differences (10,000 permutations, seed 0) and a cluster bootstrap over cells
  (10,000 resamples, seed 0) for 95% CIs. Exact McNemar on trials is reported as secondary. Holm correction within each experiment's
  family of primary comparisons (families listed below). Alpha = 0.05 two-sided.
- Unparsed replies count as WRONG in the primary analysis; a sensitivity analysis drops every (N,H,lead-in) key with any unparsed reply.
- "Equivalent" = 95% CI of the difference lies inside +/-5 percentage points (stated per use). A bare non-significant result is never
  reported as "no effect"; null results are reported with the upper 95% confidence bound.
- Every experiment has a **verify gate** (below) run first on real weights. If a gate fails, the experiment's results are labelled
  "uninterpretable until fixed" and not used for claims.
- Exploratory analyses must be labelled "exploratory" in the paper; none may be used to reinterpret a pre-registered decision rule.

## 3. E0 (done before this file; descriptive)

Label priority (fixed): correct > unparsed > leak > follows_image > adjacent (+/-5 minutes of truth) > default (00/30) > other.
Ambiguities acknowledged: at H=12 leak and the "00" default coincide (leak wins); at N=H the leak value equals the truth (correct wins).

## 4. E3 — text-side analysis (runs first; notebook D, also contains the InternVL3-8B 128-token re-run)

Models: Qwen-3B and InternVL3-2B (both targets for E3d), Qwen-7B (specificity control), text-only prompts unless stated.

**E3a logit lens.** Teacher-force the gold hour prefix ("HH:") for every trial; at the position predicting the minute, apply the final norm
+ unembedding to each layer's residual. Metric per layer: log-prob(gold minute string) - log-prob(leak minute string), via
`intervene.teacher_forced_minute_logprob`-style scoring (digit-split aware). Compare trials the model gets right vs wrong, cluster bootstrap.
Descriptive; no decision rule. (Exploratory.)

**E3b swap patching.** Pairs of prompts P(N,H) and P(H,N) with equal token length only (so both numbers occupy identical positions; pairs that
include 10, 11, 12 are paired only with each other when both are 3-token numbers; reported separately). Patch the residual of a source prompt into
the target prompt, per layer, at: N-token span, H-token span, between-span tokens, last token. Metric: shift of the answer minute toward the
source prompt's minute (log-prob difference, source minute minus target minute). Controls: identity patch (must be 0 +/- numerical noise), patch from
a same-answer prompt, random-position patch.
Decision rule (pre-registered): the "minute value is formed from the N-token" statement is supported if, for some layer window, patching the
N-span moves the answer toward the source minute with a shift whose cluster-bootstrap 95% CI excludes 0 and exceeds every control by >= 1 nat.

**E3c restoration pairs.** Same N, different H; one prompt answered right, the other wrong (model-specific, defined from existing outputs).
Patch the right run's residual into the wrong run, per layer/position. Controls: right->right, wrong->wrong, random position, identity.
Metric: fraction of wrong runs restored to the correct minute. Supported localisation: restoration >= 50% of the wrong runs at a single
layer-position cell AND >= 25 percentage points above the best control. (Exploratory below that.)

**E3d edge knockout (the key causal test of M-bind).** Block attention from the minute-slot positions (the tokens that form the answer's minute,
i.e. all answer positions) to the H-token span, by layer window (4 wide, plus cumulative-to-end, plus all layers). Controls: same-size knockout
of a random prompt span ("What time does the clock show" tokens); knockout to the N-token span (destruction control: must collapse accuracy).
Primary metric: leak rate and minute accuracy in text-only, evaluated SEPARATELY and with IDENTICAL thresholds for the two target models, Qwen-3B and
InternVL3-2B (the 2B has 117 leaks vs 46 for the 3B, so it is the better-powered test; the 3B alone may be underpowered). Qwen-7B is the specificity
control (expected: no effect). Decision rules (primary comparison per model: all layers vs intact; Holm over the family {3B, 2B, 7B}; thresholds are
relative to each model's own intact baseline):
 - **M-bind supported (interference), per model:** leak count falls by >= 30% relative to that model's intact count (3B: 46 -> <= 32; 2B: 117 -> <= 82)
   AND minute accuracy rises by >= 4 percentage points, both CIs excluding 0, and the random-span control changes accuracy by < 2 points.
 - If supported in the 2B but not in the 3B, the 3B result is reported with its power caveat (smallest detectable leak drop at 46 leaks), not as evidence against M-bind.
 - **Weak-conversion account supported instead:** leak falls but accuracy gain < half of the leak decrease (errors change type, not disappear).
 - Otherwise: reported as inconclusive.

**E3e description transplant (answers "describes both hands exactly right yet answers wrong 55%").** Take the 111 Qwen-3B clocks whose describe reply
had both hands EXACTLY right (and the analogous sets for 2B (144), 7B (242), 8B (216)). Build a text-only prompt containing the model's own
correct description sentence (extracted values N,H) followed by the standard question, no image. Compare the wrong-answer rate with the
original with-image describe-then-answer rate (paired by clock).
 - **Language-side failure supported** (for Qwen-3B): wrong rate without image >= 40% (>= about 70% of the 55% observed) and the paired difference to the
   with-image rate has a CI inside +/-15 points.
 - **Image interference supported:** wrong rate without image <= 25%, with the paired difference CI excluding 0.
 - Between 25% and 40%: reported as mixed.

**E3f H=12 anomaly.** (i) Re-score E0 for H=12 vs H=10/11 (done: 12 differs, 10/11 do not; scorer verified to treat hour 00 == 12 mod 12). Hour-slot and
minute-slot errors at H=12 are analysed separately (hour copied from N; hour "01"; minute 00). (ii) Prompt variants that change only the form of H: "twelve",
"12 o'clock", "the number twelve", and N likewise. (iii) Logit lens/E3b at H=12 cells. Decision rule: a "numeral/tokenisation form" explanation is
supported for a model if the word form ("twelve") raises H=12 minute accuracy to within 10 points of its H!=12 accuracy while the numeral
keeps the gap; a "semantic 12=00 convention" explanation is supported if the gap remains for all forms. Exploratory beyond these two rules.

**Verify gate (E3):** (1) teacher-forced scoring reproduces the free-generation correctness on >= 95% of 100 random trials; (2) identity patch changes log-probs
by < 1e-3 nats; (3) knockout with an empty block set reproduces the intact generations exactly; (4) a full-prompt knockout of all tokens before the answer is
rejected (guard).

**Text-light pilot (same notebook):** intact text-light minute accuracy (hour in text, minute from the matching clock only), 432 trials, all four models; feeds the E2 floor rule (Section 6).

**8B 128-token re-run (same notebook):** `text_only_v2` + `conversion_test` for InternVL3-8B at max_new_tokens=128; success = `cap_check` reports 0 cap hits
in conversion_test and <= 5 in text_only_v2; otherwise report the residual and keep the 64-token numbers as primary with this as sensitivity.

## 5. E1 — attention knockout (which layers carry image evidence; notebook E)

Block attention from text positions to image-token keys by layer window: 4-wide sliding windows (2B/7B/8B: 7 windows over 28 layers; 3B: 9 over 36),
cumulative-from-L-to-end, and all layers. Two query sets: (A) answer positions only, (B) all text positions after the image. Conditions: matching_clock
(all cells), blank and mismatched_clock (all layers, early half, late half, final 4 layers only).

**Verify gate (E1):** eager and SDPA attention agree to < 1e-3 on logits; empty block set reproduces intact generations exactly; the hook reaches every decoder
layer of both families (counts logged).

**Machinery check (NOT a hypothesis test).** Blocking all text->image attention at all layers must reproduce text-only behaviour by construction, whichever
hypothesis is true, so this cell carries no evidential weight about mechanisms and is NOT part of any prediction below. It only tests whether the knockout
code disturbs the model. Rule (binding): for each model and query set separately, compare minute accuracy of "image blocked at ALL layers" (matching_clock) with
that model's real text-only run (same 432 trials). If they differ by MORE THAN 5 percentage points (absolute), the knockout is INVALID for that model/query set
(it disturbs the model beyond removing the image) and ALL E1 results for it are reported as EXPLORATORY ONLY. The outcome is reported whether or not it passes,
and it is checked before any windowed cell is interpreted. (For Qwen-3B, text-only is 67.6% and matching clock 61.8%; passing means blocked accuracy >= 62.6%.)

**Confirmatory windowed hypothesis tests — InternVL3-2B only** (total image effect 22 points: 49.8% text-only vs 71.8% matching clock; model/query set must pass
the machinery check). Primary query set = B; query set A is reported alongside, not used for decisions. All on matching_clock, comparing each window with the intact run;
Holm over the 7 sliding windows; effect sizes are relative to the total image effect T = intact minus text-only accuracy (expected ~22 points).
 - **H-E1-1 (localised evidence):** at least one sliding window removes >= 50% of T (accuracy drop >= 11 points, CI excluding 0). If no window does, the evidence is
   reported as DISTRIBUTED across layers (a result, not a failure).
 - **H-E1-2 (replication on independent trials):** the window with the largest matching_clock drop (selected on matching_clock only) lowers the mismatched_clock
   follow-image rate (54 trials at baseline, taxonomy label) by >= 50% relative, CI excluding 0. Selection and test use different image conditions (different trials).
 - **H-E1-3 (specificity):** the final-4-layers window changes matching_clock accuracy by < 3 points, and the 7B/8B all-windows cells change accuracy by < 3 points.
   Violation weakens the "image-robust" statement for 7B/8B; it does not affect H-E1-1/2.
 - Cumulative-from-L curves (how late the image can still be removed) are descriptive; the onset layer is reported but not tested.

**Qwen2.5-VL-3B: E1 windowed results are EXPLORATORY by pre-declaration.** The whole image effect is only ~6 points (61.8% vs 67.6%; the machinery-check headroom
is ~1 point above the intact value), so confirmatory power is not available. Windowed results are reported and clearly labelled exploratory; no E1 statement about
the 3B is used as a confirmatory claim. Qwen-3B's confirmatory causal evidence is E3 (text side: E3d, E3e) and the E2 text-light variant (Section 6), subject to
the floor rule stated there.

## 6. E2 — cross-clock image-token patching (sufficiency and content; notebook F)

Text fixed to clock A ("minute hand at N_A, hour hand at H_A"); image of clock A is shown; image-token residuals of A are replaced by those of clock
B at layer L. B has the same hour number and a minute number >= 3 away from N_A (circular), chosen deterministically (seed 0). Because image
positions precede the text and are causal, a patch at post-layer L equals supplying image B to layers > L while text positions at layers <= L still
see image A's keys/values; this limitation is stated in the paper.
Layers: ~8 spread over the depth (chosen before running: embedding output, then layers at 1/8..7/8 of depth); token sets: all image tokens, minute-hand
tokens, hour-hand tokens, all other tokens (regions from the drawing geometry of A union B).
Controls: (c1) same-minute B (a different render with the same N) -> noise floor; (c2) norm-matched noise (`intervene.make_matched_noise`);
(c3) the text-light variant (hour number in text, minute only in the image). For Qwen-3B the text-light variant is the CONFIRMATORY E2 test (the standard prompt
leaves it too little image dependence); for InternVL3-2B it is the positive control.
**Floor rule (pre-declared):** a pilot in notebook D measures intact text-light minute accuracy (matching clock, 432 trials) for all four models. For any model
where it is < 15%, the text-light E2 test is declared UNINFORMATIVE (floor) and that model's E2 results are exploratory. (Reason: Qwen-3B's plain behaviour accuracy on
minute is only 2.4%, so reading the minute from pixels may be too weak for a patch to move the answer; this is a risk to the E2 text-light plan, not an assumption.)

**Verify gate (E2):** identity patch (B=A) changes generations in 0 of 100 trials; the embedding-level all-token patch reproduces the existing
mismatched_clock follow-image rates within 3 points for each model; the region masks overlap the drawn hands (checked on 20 images by overlay).

Metrics: P(answer minute = B's), P(stays A's), teacher-forced log-prob(B's minute) - log-prob(A's minute).
Decision rules (Holm over layers x regions per model):
 - **InternVL3-2B reads minute evidence from the minute-hand region:** at some layer, minute-hand-token patching raises follow-B by >= 5 points
   over the same-minute control (CI excluding 0) AND the effect exceeds the other-region patch at that layer.
 - **Cue/consistency account:** only all-token (or non-hand) patching moves the answer; minute-hand-only does not (upper bound < 3 points).
 - **Qwen-3B (confirmatory test = text-light variant, only if the floor rule is passed):** image tokens carry minute evidence if minute-hand-token patching
   raises follow-B by >= 10 points over the same-minute control (CI excluding 0) at some layer, Holm over layers x regions. A claim of "no image-driven minute
   shift" requires the all-token patch's upper 95% bound on follow-B < 5 points AND the 2B (positive control) showing >= 20 points under the same procedure.
   Otherwise "inconclusive". Standard-prompt E2 results for the 3B are exploratory.

## 7. Compute and ordering (guesses, not measurements)

Order: E0 (done) -> E3 (notebook D) -> E1 (notebook E) -> E2 (notebook F). Each: Kaggle T4x2, verify gate first, step logging, zip at end,
and an offline tiny-model test before upload. Guessed GPU time: D 3-4 h (incl. 8B re-run ~1 h), E ~4 h, F ~3 h. These are
guesses; real times will be reported afterwards.

## 8. What would change the framing

- E3d + E3e both support language-side failure -> "binding failure in text; images modulate it" is primary; "visual route" is dropped as a name.
- E1/E2 show image-token dependence only through non-hand tokens -> the image is a cue, not a minute source for 2B.
- E1/E2 show hand-token minute evidence for 2B -> report 2B as using both: partial visual readout plus leak suppression.
- Any gate failure -> that experiment is re-run after a documented fix (log it below), and results from before the fix are kept and labelled.

## Deviations log (append only; date, what, why)

(none yet)
