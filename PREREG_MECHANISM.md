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

**2026-11-05 (PRE-RUN: logged before any notebook D run; no real-weight result of any kind has been seen).** Implementation choices the
prereg left open, for `mech_e3.py` / notebook D. None alters a hypothesis, threshold or decision rule except item 5, which fixes which E3e variant the
existing thresholds apply to.

1. **Hour prefix (E3a/b/c).** Teacher-forced scoring forces a zero-padded gold hour prefix ("01:") before the minute digits, because InternVL3-2B often
   writes unpadded hours in free generation.
2. **Layer subset.** Patching in E3b and E3c uses every 3rd layer (`range(0, L-1, 3)`), not all layers. E3a (logit lens) uses all layers.
3. **E3c restoration.** Scored as the argmax over all 12 minute candidates with the target's gold hour forced. "Right" and "wrong" runs come from the
   prior free-generation outputs (`conversion_test.csv`); up to 100 right->wrong pairs, plus wrong->wrong, right->right (cap/2), random-position and
   identity controls.
4. **E3b same-answer control.** Source = P(N,H') with the same minute answer, patched at the H-span, in-between tokens and last token. Control metric =
   absolute change in the target minute's log-prob (the swap metric is zero by construction there). Other controls: identity, random-position.
5. **E3e primary variant (changed per user instruction, pre-run).** PRIMARY = the VERBATIM description: the model's own words from `describe.csv`,
   cut before its answer sentence (first of "therefore/finally/thus/so the time/the time is/shown"), followed by the standard conversion question, no
   image. This follows the prereg wording "feed the model's own correct description". SECONDARY = template sentence ("The long minute hand is
   pointing at the number N, and the short hour hand is pointing at the number H.") + conversion question. EXPLORATORY = template + describe-style
   closing question. The decision thresholds in Section 4 E3e are unchanged (>=40% wrong with paired-difference CI within +/-15 points =
   language-side; <=25% with CI excluding 0 = image interference; between = mixed) and are applied to the verbatim variant. Clocks whose reply has
   no extractable description before the answer (or <=20 characters) are absent from the primary variant; the number of clocks per variant is
   reported (`e3e_summary.csv: n_clocks`) and the paired comparison uses only those clocks. The thresholds are also printed for the other variants
   but those verdicts are descriptive.
6. **E3f forms.** "the number 12", "the number twelve", "the 12 o'clock position", over the cells where N>=10 or H>=10; conditions
   num/num, num/word, num/oclock, word/num, oclock/num.
7. **E3d baseline.** The intact baseline is the same eager no-KV-cache greedy loop with no block, so "empty block = intact" holds exactly; G8b
   separately reports agreement of that loop with `adapter.generate` (fp16 numerics, >=95% required).

Also logged: per-model verify gate runs inside each model's first step (not one global gate step); HF-cache purge steps between model groups; the
InternVL3-8B E3e/pilot step runs with `--no_gate` (no hooks used); Qwen-7B E3d runs in `min` mode (intact / H-span all layers / random span).

**2026-11-05 (post-run, notebook D first run) -- gate failure; no mechanism result produced.** On Kaggle (T4 x2, device_map='auto') the verify
gate crashed in `part_verify` (G4 logit-lens check, `mech_lib.lens_logprobs`) with a cross-device tensor error for Qwen-3B, InternVL3-2B and Qwen-7B
(indices on one GPU, activations on the other). G3-G9 were never evaluated and no `verify.json` was written, so by the gate rule every dependent part
(E3d, E3b, E3c) was refused (rc=3). Separately, G2 (teacher-forced vs free-generation agreement, threshold 0.95) FAILED for Qwen-3B (0.85;
teacher-forced acc 0.74, free acc 0.67, n=100) and InternVL3-2B (0.83; 0.51 / 0.54); Qwen-7B passed (0.99). Consequently E3a, E3b, E3c, E3d, E3f, the
Qwen/InternVL-2B E3e and the Qwen/InternVL-2B text-light pilots were NOT run, and nothing is claimed for them. What did run: the InternVL3-8B
128-token re-run (text_only_v2, conversion_test, cap_check) and the 8B E3e + text-light pilot (`--no_gate`, no hooks, as pre-logged). Per the prereg
("any gate failure -> re-run after a documented fix, log it"), the fix and re-run will be logged here when made; this run's outputs are kept.
The 8B E3e verbatim variant retained all 216 clocks (none dropped). The 8B E3e verdict string in `e3e_summary.csv` is descriptive only: the
Section 4 thresholds were written for the Qwen-3B claim and no confirmatory 8B claim exists. The `describe_style` variant is uninformative at
max_new_tokens=128 (216/216 replies are cut-off chain-of-thought text) and is not interpreted.

**2026-11-06 (PRE-RUN: logged before ANY gate-only or re-run result exists; supersedes nothing above except where it says so).**
Changes after notebook D's gate crash. No hypothesis, threshold or decision rule of Sections 4-6 is changed.
1. **Multi-GPU fix.** `mech_lib.lens_logprobs`, `score_candidates` and `patch_residuals` now move every tensor to the device of the tensor it
   is combined with (hidden state -> final-norm device -> lm_head device; target ids -> logits device). Pure bug fix; no numerical change
   (checked on a single device and on an emulated 2-GPU split).
2. **Machinery gate for G2 = G2-own.** G2-own = teacher-force the model's OWN free-generation prefix (the exact string up to and including its
   first ":", e.g. "1:" or "01:") and compare the argmax-correctness over the 12 minute candidates with the free-generation correctness, on 100 sampled
   trials. **Gate: G2-own agreement >= 0.95.** A token-level exact-match rate (generated minute digits = teacher-forced argmax) is logged too.
   `--g2_gate own` is the default.
3. **G2-gold is always reported, never a gate.** G2-gold = the same comparison with the forced gold zero-padded hour prefix ("01:"), reported with
   a per-category breakdown of how the model's own prefix relates to the gold prefix (exact match / same hour unpadded / wrong hour padded /
   wrong hour unpadded / other text / no colon) and a per-trial file (`verify_g2_trials.csv`). A mismatch under the gold prefix is interpreted as a
   real prefix-dependence effect (consistent with the hour->minute leak: the minute the model writes depends on the hour it wrote), NOT as a
   machinery failure. (Notebook D's G2-gold values were 0.85 for Qwen-3B and 0.83 for InternVL3-2B; the own-prefix values were not measured.)
4. **If G2-own < 0.95 for any model, no experiment part runs for that model until the cause is diagnosed** (logged here before any re-run).
   All other gates (G1, G3-G9) are unchanged and still gate.
5. **E3a/E3b/E3c are reported under the pre-registered gold prefix, with the caveat of item 3, AND conditioned on trials where the model's own
   free-generation hour equals the gold hour (mod 12)** (`own_hour_eq_gold` column; extra rows in `e3a_summary.csv` [groups `own_hour_eq_gold*`],
   `e3b_summary.csv` [subset `own_hour_eq_gold`], and `e3c_summary_own_hour_eq_gold.csv`). The conditioned analysis is a robustness check, not a
   replacement of the pre-registered one; where the two disagree, both are reported and the disagreement is a finding.
6. **A crash inside one check is recorded as that check's FAILURE** (with the error text) and the gate continues; `verify.json` is always written;
   informational checks (non-gating G2 variant) carry `"gating": false`.
7. **Run structure.** A gate-only notebook (D2: verify for Qwen-3B, InternVL3-2B, Qwen-7B, nothing else) runs first; the full re-run (D3: notebook D
   minus the InternVL3-8B steps, which are complete) is launched only after all three gates pass.

**2026-11-06, second entry (PRE-RUN for the re-check D2' and for D4; written before any result of either exists).**
Context: the first gate-only run (D2) showed (numbers as reported by the user): Qwen-3B G2-own 0.93, G8b 0.72; Qwen-7B G8b 0.94, G7b 1/20, G8 0.0,
G9 0/20; InternVL3-2B G7b 0/20, G8 0.0, G9 0/20; G2-gold < G2-own for all models (the prefix hypothesis of the previous entry held). No
threshold of any gate is changed by this entry; no hypothesis, threshold or decision rule of Sections 4-6 is changed.
1. **Attention-blocking path (G7b/G8/G9).** Observation: G8 = 0.0 means default-attention output and eager output NEVER agree, while G5 (eager vs eager
   with an empty hook) passes - so eager does take effect, but its output is different from the default path's. Working hypothesis (NOT yet
   confirmed on real weights): the stock eager function multiplies q.k^T in fp16, overflows for some heads, and returns NaN/garbage, whereas the fused
   SDPA kernels accumulate in fp32. This also explains G9 (garbage never contains ":") and G7b (a NaN log-prob difference was counted as "no change"
   by the old check). Fix: inside `mech_lib.eager_attention()` the modeling module's `eager_attention_forward` is replaced by a copy that computes
   q.k^T, mask, softmax and p.v in float32 (same math; restored afterwards). Reproduced and fixed on a tiny fp16 Qwen2.5-VL with exploding q.k
   (stock eager non-finite, new path finite and equal to SDPA); that proves the failure mode exists, not that it is the cause on the real models.
2. **New or hardened checks (all use the unchanged thresholds where one exists).** G8a (gating): every decoder attention module's own config reads
   "eager" inside the context, and the per-layer implementation (default and inside the context) is saved to verify.json. G8c (gating): teacher-forced
   logits under eager are finite, max |logprob diff| to the default path < 0.5 nat, argmax equal on >= 90% of 10 trials. G7d (gating): baseline and
   blocked log-probs are finite. G7b/G7c/G8/G8b: a NaN counts as a failure, and two unparseable outputs no longer count as agreeing.
3. **Qwen no-cache loop (G8b, and G2-own 0.93).** The hypothesis "stale `rope_deltas` contaminates the no-cache loop" was checked first: Qwen2.5-VL
   stores `rope_deltas` on the model, but for a text-only prompt it is a constant added to all three position axes, and RoPE is relative, so a
   stale value cannot change the logits (verified numerically on the tiny model: identical logits with a stale value of 11). `rope_deltas` is
   nevertheless reset before every no-cache forward (defensive, no effect expected). New informational diagnostics decide between the remaining
   candidates (fp16 cached-vs-uncached numerics near ties vs a real position/cache error): G10 (cached incremental decode vs full forward on the same
   tokens: max logit diff, argmax agreement), the no-cache greedy loop is compared with cached `generate` per trial, and every G2-own disagreement is
   classified (free minute off the 12-candidate grid / near-tie < 0.5 nat / clear) and listed by prefix class in `verify_g2_trials.csv`.
   **Rule:** the G2-own >= 0.95 gate and the G8b >= 0.95 gate are NOT relaxed; if either still fails after these fixes it is diagnosed before use.
4. **Hook-free parts may run without the hook gates.** E3e (text transplant), E3f (H=12 forms) and the text-light pilot use no residual capture/patching,
   no attention blocking and no logit lens (checked in the code: only `generate` and ordinary forward passes). They may run for Qwen-3B, InternVL3-2B and
   Qwen-7B with `--no_gate`, exactly as the InternVL3-8B steps already did. They are NOT conditioned on G2-own or on any hook gate. E3a/E3b/E3c/E3d and
   any later E1/E2 still require a passing gate for that model.
5. **Run structure.** D4 (`notebook_D4_hookfree.ipynb`: E3e, E3f, pilot for the three models, no gate) can run now; D2' (the updated gate-only
   notebook) re-checks items 1-3; the hook-dependent re-run (D3) waits for D2' to pass for the model.

**2026-11-06, third entry (POST-RUN for D2' and D4; PRE-RUN for the next gate re-check).**
Observed (full tables: `results/mech_e3/D2prime_D4_report.txt`, script `analysis/analyze_D2prime_D4.py`):
1. **Attention-blocking fix worked.** With the fp32 eager path, G7b = 20/20, G9 = 20/20, G8 = 1.0 and G8a/G8c/G7d pass for InternVL3-2B, Qwen-3B and Qwen-7B.
   InternVL3-2B passes every gating check (G2-own 0.95, G8b 1.0). Verdict per the pre-registered gate: InternVL3-2B is cleared for the hook-based parts.
2. **Qwen-3B fails two gating checks that the fix did not touch:** G2-own 0.93 and G8b 0.72. Qwen-7B fails G8b (0.94) only (G2-own 0.98 passes).
   Not relaxed. No hook-based part runs for either Qwen model until this is diagnosed and the gate passes.
3. **Diagnosis so far (not yet a confirmed cause).** The cached-vs-uncached KV logits agree (G10: max |logit diff| 0.04-0.06, argmax agreement 1.0), so there is no
   position/cache bug. But Qwen's cached `generate` differs from the no-cache greedy loop on 26/100 (3B) and 3/100 (7B) trials, in 18/26 (3B) already in the HOUR.
   InternVL3-2B: 0/100. The Qwen checkpoints ship `repetition_penalty = 1.05` in `generation_config.json` (InternVL3-2B-hf ships none); `generate` applies it, the
   no-cache loop and the teacher-forced scoring do not. This is the leading explanation and it also means **every Qwen cached-generation output in this project
   (all behaviour tables, the E3 priors, E3e/E3f/pilot) was produced with that penalty on prompt tokens**. This is a property of the checkpoint's default decoding,
   not edited here. New informational check G11 (penalty forced to 1.0 vs checkpoint penalty) and a `--rep_penalty` option in `mech_e3.py` (default unset = unchanged
   behaviour) were added BEFORE any result of them exists; G2-own >= 0.95 and G8b >= 0.95 stay the gates.
4. **D4 (hook-free, run under the 2026-11-06 second entry).** All steps OK. E3e Qwen-3B, primary verbatim variant: 111 clocks, wrong without image 54.1%, paired
   difference to the with-image rate -0.9 pts (CI -10.8 to +9.0) -> by the pre-registered rule language-side failure is supported. Caveat: generation used the
   checkpoint's repetition penalty (item 3) in both arms; a penalty-free sensitivity run is pending and will be reported next to this result, not instead of it.
   E3e for the other models, the secondary/exploratory variants, E3f and the pilot are descriptive. 7B verbatim sits at a ceiling (0 wrong in both arms), 2B has ~0.7% wrong with image.
   Pilot: Qwen-3B 0.109 < 0.15 floor; InternVL3-2B 0.500, Qwen-7B 0.692, InternVL3-8B 0.750 clear it.

**2026-11-06, fourth entry (PRE-RUN: written before any rp=1.0 result exists). DECISION: headline decoding = plain greedy, repetition_penalty = 1.0, for ALL four models.**
Reason (decided by the project owner): (a) the Qwen checkpoints' default `repetition_penalty = 1.05` acts on prompt tokens, so it penalises copying numbers from the prompt
and is entangled with exactly the hour->minute leak under study; (b) InternVL3 ships no penalty, so the families must be decoded the same way to be comparable.
1. **Headline vs sensitivity.** Every Qwen-3B / Qwen-7B result used so far (text-only, conversion test, describe, image behaviour, E3 priors, E3e/E3f/pilot of D4) was generated with
   the checkpoint default (1.05). Those become the **SENSITIVITY analysis ("models' default generation config")** and keep a `_rp105` suffix; they are never overwritten.
   The HEADLINE for Qwen is the set regenerated at `repetition_penalty = 1.0` (notebook Q). InternVL3-2B/8B results are unchanged (no penalty exists), so they serve both.
2. **Mechanism of the change.** `CLOCK_REP_PENALTY=1.0` (environment variable read by `adapters.QwenVLAdapter.generate_answer`; unset = old behaviour byte-for-byte) for text_only_v2,
   conversion_test, describe_check, eval_behavior; `--rep_penalty 1.0` for `mech_e3.py` (its `ctx.gen`). The rp=1.0 value is recorded in each run_config. Max-new-token settings are unchanged.
3. **Gate re-check (D2'').** Qwen-3B and Qwen-7B are re-gated with `--rep_penalty 1.0` for generation; the no-cache loop and teacher forcing never had a penalty. Thresholds unchanged
   (G2-own >= 0.95, G8b >= 0.95). If they pass at 1.0 that supports the penalty as the cause of the D2' failures; if they still fail they are diagnosed before any hook-based Qwen part runs.
   A gate is an inference about the machinery for one decoding setup: hook-based Qwen experiments run only after the rp=1.0 gate passes.
4. **Quantities that will change and are re-derived, not edited above.** The per-model counts quoted in Section 4 (E3e sets 111/144/242/216, leak counts 46/117, the 55% with-image rate)
   were computed from rp=1.05 Qwen outputs. All decision rules keep their form and thresholds, evaluated on the rp=1.0 data (relative thresholds use each model's own rp=1.0 baseline).
   The E3e Qwen-3B confirmatory verdict is **re-computed on the rp=1.0 set (headline)**; the D4 verdict (language-side supported at rp=1.05) is reported as sensitivity. If the two disagree,
   both are reported and the disagreement is a finding about decoding, not hidden. The text-light pilot and E2's 15% floor are re-evaluated on rp=1.0.
5. **Order.** D2'' gate (Qwen only) -> notebook Q (regenerate Qwen results at 1.0, incl. E3e/E3f/pilot, `--no_gate`, hook-free) -> only then hook-based Qwen E3a-d, E1, E2 on the rp=1.0 priors.
   InternVL3-2B E3a-d (its gate passed in D2') runs now as notebook D3-IVL with the unchanged InternVL priors.

**2026-11-07 (POST-RUN for D2'', notebook Q, D3-IVL; no rule or threshold changed). Full tables: `results/mech_e3/Q_D3IVL_report.txt`; script `analysis/analyze_Q_D3IVL.py`.**
1. **Penalty was the cause of the Qwen gate mismatch.** At `--rep_penalty 1.0`: Qwen-3B G2-own 0.93 -> 0.97, G8b 0.72 -> 1.0; Qwen-7B G2-own 0.98 -> 1.0, G8b 0.94 -> 1.0; every gating check
   passes for both (G11: no-cache loop equals penalty-free generation on 100/100 trials, but only 74/100 (3B) and 97/100 (7B) with the checkpoint's 1.05). Both Qwen models are cleared for the hook-based
   parts, to be run on rp=1.0 priors. InternVL3-2B's gate passed again in D3-IVL with identical numbers (G2-own exactly 0.95; the run is deterministic).
2. **Headline re-derivation (rp 1.0).** Both-hands-exact describe sets: Qwen-3B 122 (was 111), Qwen-7B 243 (242), InternVL3-2B 144, InternVL3-8B 216. Conversion-test text-only leak counts (of 432):
   Qwen-3B 40 (was 46), Qwen-7B 6, InternVL3-2B 117, InternVL3-8B 2. (The counting method reproduces 111/46/242 on the rp 1.05 snapshot.) Relative thresholds that were stated as "46 -> <= 32" for the 3B
   now read 40 -> <= 28; 2B is unchanged (117 -> <= 82).
3. **E3e, Qwen-3B, verbatim variant (confirmatory) at rp 1.0: 122 clocks; wrong without image 57.4%, with image 70.5%; paired difference -13.1 pts, CI [-23.8, -3.3].** The wrong-rate condition
   (>= 40%) holds but the CI is not inside +/-15 pts, and 57.4% is above the 25% image-interference bar, so neither rule is met: pre-registered label = **mixed**. The earlier D4 result at rp 1.05
   (111 clocks, 54.1% vs 55.0%, diff -0.9, CI [-10.8, +9.0]: language-side) is the sensitivity result. The two differ and both are reported; the disagreement is a finding about decoding.
4. **E3d, InternVL3-2B (confirmatory), pre-registered verdict: `M_bind_supported = false`.** Leaks 117 -> 28 (-76%), accuracy +22.5 pts (CI [+17.4, +27.5]); leak and accuracy CIs exclude 0 and the
   thresholds (>= 30%, >= 4 pts) are met, but the **random-span control raised accuracy by +3.0 pts (CI [+1.2, +5.1]) which violates the pre-registered clause "< 2 pts"**, so the rule as written is not
   satisfied. This is reported as such; the effect is ~7x the control's, leaks are unchanged under the control (113 vs 117), and the effect is localised to layers 16-23 (windows 16-20: 87 leaks,
   20-24: 37, cumulative-to-end 28). The Holm correction over the family {3B, 2B, 7B} is still open (E3d for Qwen not yet run).
5. **E3b (N-span) supported at 8 of 9 tested layers; E3c not supported (best cell 48.8% restoration at layer 15, H-span, vs the 50% bar; best control 23.8%); E3a exploratory.**
6. **Pilot / E2 floor at rp 1.0:** Qwen-3B 0.093 (fails 0.15), Qwen-7B 0.637, InternVL3-2B 0.500, InternVL3-8B 0.750 (pass).
7. **Housekeeping.** All pre-Q Qwen per-model result files were renamed `*_rp105*` (never overwritten); the rp 1.05 state is also snapshotted under normal names in `results/_sens_rp105/` for the
   sensitivity comparison; the aggregates that were regenerated are backed up in `results/_backup_pre_Q/`. The frozen top-level files (`results/results.csv`, `describe*.csv`) are untouched.

**2026-11-07, second entry (decision of the project owner; no rule or threshold changed).**
1. **E3d InternVL3-2B: the pre-registered verdict stays exactly as computed (`M_bind_supported = false`)**, the cause being that the random-span control changed accuracy by +3.0 points against the pre-registered
   bound of < 2 points. In addition, a **control-adjusted comparison is reported, labelled EXPLORATORY** (`analysis/e3d_control_adjusted.py`, `e3d_control_adjusted.csv`; paired per-trial contrasts, cluster bootstrap over
   (N,H) cells, 10000 resamples): blocking the H-span minus blocking the random span = accuracy +19.4 pts (95% CI +14.4 to +24.5), leak rate -19.7 pts of trials (CI -23.6 to -16.0); the random-span control alone:
   accuracy +3.0 (CI +1.2 to +5.1), leak rate -0.9 (CI -2.8 to +0.7). It is not used to change the verdict.
2. **Priors for the Qwen hook-based parts = the rp 1.0 files** (headline decoding, entry of 2026-11-06 fourth). `kaggle/upload_D/clock-prior` now holds the rp 1.0 Qwen files; the previous (rp 1.05) version is kept
   as `kaggle/upload_D/clock-prior_rp105` for sensitivity. InternVL3 priors are unchanged.
3. **Qwen E3a-d (notebook D3-Q), all at `--rep_penalty 1.0`:** Qwen-3B confirmatory E3d bar re-stated on the rp 1.0 baseline: leaks 40 -> <= 28 (>= 30% relative drop) AND accuracy rise >= 4 pts, both CIs excluding
   0, random-span control < 2 pts; Qwen-7B (6 leaks at baseline) is the specificity control (E3d `min` mode, expected no effect), not a confirmatory test. Holm over the family {3B, 2B, 7B} as pre-registered.

**2026-11-07, third entry (PRE-RUN for E1 and E2; decisions of the project owner; written before any E1/E2 code has run).**
1. **E1 gate: the "eager and SDPA attention agree to < 1e-3 on logits" rule (Section 5) is replaced by the G8c form:** with eager attention (fp32-accumulating, `mech_lib.eager_attention`) the logits on 10 prompts are finite, the
   maximum absolute log-prob difference to the default attention path is < 0.5 nat, and the argmax is equal on >= 90% of the 10 prompts. Reason: 1e-3 is numerically impossible in fp16 - the measured differences were 0.05-0.08 nat
   on every model with identical argmax (G8c in the D2'/D2''/D3-IVL runs) - discovered before any E1 run. All other E1 gate items (empty block set reproduces intact generations exactly; hook reaches every decoder layer;
   machinery check within 5 points) are unchanged.
2. **E1 query sets (confirmed):** A = the last prompt token plus the generated tokens (the positions that produce the answer); B = all text positions after the last image token (question text and generated tokens).
   Image tokens are never blocked as queries. Primary query set = B, as pre-registered.
3. **E2 scope:** 100 pairs per model (seed 0). InternVL3-2B is the confirmatory model (decision rules of Section 6); Qwen-7B and InternVL3-8B are run with the same procedure and reported as DESCRIPTIVE (no pre-registered rule);
   Qwen-3B is EXPLORATORY (text-light floor 0.093 < 0.15, pre-declared). E2 layers: embedding output plus the layers at 1/8..7/8 of depth, fixed before any run.
4. **E1 Qwen-3B machinery check** is re-stated on the headline (rp 1.0) text-only run: text-only 64.4%, tolerance 5 points (the Section 5 numbers 67.6%/61.8%/62.6% were rp 1.05). Qwen-3B E1 stays exploratory by pre-declaration.
5. **Decoding** for E1 and E2 generations: cached greedy generation, `repetition_penalty 1.0` (Qwen), max_new_tokens 32, eager fp32 attention inside the knockout context.

**2026-11-07, fourth entry (PRE-RUN for E1 and E2 implementation choices; written before any E1/E2 result exists; no hypothesis or threshold changed).**
1. **H-E1-2 needs one extra step.** The pre-registered cell list runs blank/mismatched_clock only for {all, early, late, final4}, but H-E1-2 tests "the sliding window with the largest matching_clock drop (selected on matching_clock only)" on
   mismatched_clock. That window is data-dependent, so it is run in a follow-up step (`mech_e1.py --parts followup`) after the matching_clock cells are complete, for each query set; selection uses matching_clock only.
2. **E1 trials/images** are exactly the conversion-test ones (432 groups; `conversion_test.ensure_image`; prompt `conversion_test.prompt_for`). Cached greedy generation, max_new_tokens 32, `--rep_penalty 1.0`. An extra `intact_eager`
   cell (matching_clock, eager fp32 attention, no hook) is run to quantify eager-vs-default differences; "intact" in all pre-registered comparisons is the default-attention run.
3. **E2 clocks and pairs.** Clocks are rendered with the conversion-test renderer (`clocks.generate_clock_image` defaults). 100 pairs (seed 0): A = 100 distinct (N, H) cells drawn from the 144; B = same hour H, minute number >= 3 away (circular); c1 partner = same N,
   different hour. Text = conversion-test prompt of A (lead-in 0) for variant `standard`; for variant `text_light` the hour-only prompt of the text-light pilot. Variants run: `standard` for all four models; `text_light` for InternVL3-2B (positive control) and Qwen-3B (exploratory; floor fails).
4. **E2 patch positions and sets.** Hidden-state index k: 0 = embedding output (vision features merged), k >= 1 = output of decoder layer k-1; k in {0, floor(L*i/8), i = 1..7}. Token sets: all image tokens; minute-hand tokens and hour-hand tokens (pixels of the drawn strokes of A union B, from re-rendering
   with and without hands, mapped to the token grid; a token is in the set if any of its pixels is); other = all minus the two. Patch acts on the prefill pass; decoding uses the patched KV cache. c2 (matched noise) is run for the sets {all, minute_hand} only (cost); c1 and real-B for all four sets.
   Generation for E2 uses max_new_tokens 8 (answers are "HH:MM"; the first parsed time is used).
5. **E2 gate interpretation.** "The embedding-level all-token patch reproduces the existing mismatched_clock follow-image rates within 3 points" is implemented as: patching the embedding output of clock A's image tokens with those of clock B gives the same follow-B rate (within 3 points, and identical parsed answers in >= 95% of
   pairs) as directly showing image B with A's text - the property the existing rates rest on. The existing conversion-test mismatched rate is not a like-for-like target (different hour and minute for the image) and is reported only for information. Identity patch (source = A itself): 0 of 100 generations may change.
6. **Machinery check for E1** is computed in `mech_e1.py` (`e1_machinery.json`) from the blocked-all matching_clock cell and the model's own text-only conversion run (same 432 trials); outcome reported for each query set whether or not it passes.

**2026-11-08 (POST-RUN for D3-Q and notebook E; facts only, no rule or threshold changed). Tables: `results/mech_e3/D3Q_report.txt`, `results/mech_e1/E1_report.txt`; scripts `analysis/analyze_D3Q_E1.py`, `analysis/analyze_E1.py`, `analysis/e3d_control_adjusted.py`.**
1. **D3-Q (Qwen, rp 1.0): all steps OK; gates pass for both models** (3B: 18 checks, only the informational non-gating G2-gold fails; 7B: 18/18). E3d Qwen-3B (confirmatory): leaks 40 -> 13 (-67.5%), accuracy +9.3 pts
   (CI [+5.8, +13.0]), random-span control -0.9 pts (leaks 44), destruction control accuracy 0.109 => pre-registered `M_bind_supported = true`. Qwen-7B (specificity control): leaks 6 -> 2, accuracy +1.4 pts
   (CI [+0.5, +2.5]), random-span 0.0 => `M_bind_supported = false` (accuracy gain below the 4-pt bar). Holm over {3B, 2B, 7B}: adjusted p (accuracy) 0.0003 / 0.0003 / 0.031; (leaks) 0.0003 / 0.0003 / 0.12.
2. **E3b Qwen-3B:** N-span rule met at 11 of 12 tested layers (shift up to +12.3 nats at layer 21; best control 0.66 nats); H-span, between-span, last-token spans: 0 layers. **E3c Qwen-3B:** one of 48 cells meets
   the rule (layer 21, hour-token span: 52.8% of 85 wrong runs restored, best control 16.7%). E3a is descriptive.
3. **Notebook E steps:** 01 (InternVL3-2B) OK, 230 min; 03 (Qwen-7B) FAILED rc=3 after 4.1 min and 05 (InternVL3-8B) FAILED rc=3 after 4.2 min, both at the verify gate, so no E1 cell of either model exists;
   07 (Qwen-3B) OK, 441 min; 08 skipped (time budget; it was only the cache purge). The failing check in both is G5 (an all-layer block must change >= 30% of 12 generations): Qwen-7B 0/12 for query sets B and A;
   InternVL3-8B 7/12 for B (pass) and 3/12 for A (fail). G1-G4 pass for both (hook reaches 28/28 layers in prefill and decode; empty block reproduces intact 10/10; eager vs default max |delta log-prob| 0.084 and 0.095,
   argmax equal 10/10). Both models are near ceiling without the image (text-only minute accuracy 97.9% for both; matching clock 99.5% and 100.0%).
4. **Machinery check (blocked-all matching_clock vs the model's own text-only run), as pre-registered:** InternVL3-2B query set B 43.1% vs 49.8% = 6.7 pts => INVALID; query set A 62.3% vs 49.8% = 12.5 pts => INVALID;
   so all InternVL3-2B E1 results are EXPLORATORY ONLY under the binding rule. Qwen-3B: B 67.8% vs 64.4% = 3.5 pts, A 63.7% vs 64.4% = 0.7 pts => both VALID (the 3B windows stay exploratory by pre-declaration).
   Additional fact for InternVL3-2B: with query set B the all-layer block gives per-trial identical correctness for matching_clock, blank and mismatched_clock images (432/432; accuracy 43.1% in all three), while with
   query set A it does not (62.3% / 45.1% / 59.0%), because the question-text positions before the last token still read the image.
5. **E1 InternVL3-2B (exploratory because of item 4), query set B:** intact 71.8%, text-only 49.8%, T = +22.0 pts. Largest sliding window w16-20: drop 10.6 pts (CI [+5.1, +16.2]; 48% of T; Holm p 0.006); w12-16 8.1,
   w0-4 6.9, w4-8 4.2, w8-12 3.9, w20-24 4.2 (CI [-1.6, +10.4]), w24-28 -1.2. No window reaches the 11-pt / 50%-of-T line (H-E1-1: not met, evidence distributed). H-E1-2: query set B, window w16-20 on mismatched_clock:
   follows_image 54 -> 36 (-33%, CI of paired difference [+0.9, +7.4] pts): not met; query set A (reported, not a decision set), window w20-24: 54 -> 5 (-91%): met. H-E1-3 final-4 window: -1.2 pts (within 3 pts).
   H-E1-3 for Qwen-7B and InternVL3-8B: not evaluable (no runs, item 3).
6. **E1 Qwen-3B (exploratory), query set B:** intact 54.2%, text-only 64.4%, T = -10.2 pts (the image lowers accuracy). All-layer block 67.8%; windows w16-20 +6.7 pts and w24-28 +5.8 pts (Holm p 0.002, 0.006) and
   w0-4 -5.3 pts (Holm p 0.002); cumulative-from-L accuracy rises as L moves earlier (L32 56.2%, L24 63.2%, L16 72.0%, L8 72.2%; L12 69.0%). H-E1-2 (best matching window): not met for either query set.
7. **Housekeeping:** files merged into `results/mech_e3/qwen2.5-vl-{3b,7b}/` and `results/mech_e1/` by `analysis/analyze_D3Q_E1.py --stage merge` (name clashes get a `_D3Q`/`_E` suffix; log `results/_D3Q_E_merge_log.json`).
   Unzipped originals are in `match_D3Q_results/` and `match_E_results/`.

**2026-11-08, second entry (PRE-RUN for E2 and for E1 follow-up scope; decisions of the project owner; written before any E2 result exists; no hypothesis or threshold changed).**
1. **E1 for Qwen-7B and InternVL3-8B is NOT re-run.** Their verify gate failed at G5 (Qwen-7B: all-layer block changed 0/12 generations for both query sets; InternVL3-8B: 7/12 for set B, 3/12 for set A), so no E1 cell exists for either.
   The H-E1-3 sub-clause "the 7B/8B all-windows cells change accuracy by < 3 points" is therefore **not evaluated**. Reason: InternVL3-2B's E1 is exploratory (machinery check invalid for both query sets, entry of 2026-11-08), so the specificity statement it would
   qualify has no confirmatory standing, and the gate failure is probably a property of near-ceiling models (text-only minute accuracy 97.9% for both), not yet confirmed. The H-E1-3 final-4-layers part for the 2B was evaluated (-1.2 pts).
2. **E2 verify check V3 ("patching all image tokens at the middle layer moves log p(B minute) - log p(A minute) by > 0.1 nat in >= 50% of pairs") becomes informational (non-gating) for Qwen-7B and InternVL3-8B only**, via a new flag `--v3_informational` passed only for those two models.
   Its value is still computed and logged. Reason: both are near ceiling without the image, so a patch may legitimately move the teacher-forced difference little; the patch mechanics are established by V1 (identity patch changes 0 generations) and V2
   (embedding-level all-token patch equals showing image B), which stay gating for every model. InternVL3-2B's gate is unchanged (V3 gating). Motivation observed in notebook E: the analogous generation-change check (G5) failed for exactly these two models.
3. **Qwen-2.5-VL-3B is DROPPED from E2** (standard and text-light variants). Reason: it fails the pre-declared 0.15 text-light floor (pilot 0.093 at rp 1.0), so its E2 results could only be exploratory, at a high runtime cost (notebook E took 441 min for the 3B against a 180-min
   guess). The pre-registered Qwen-3B text-light rule is therefore not tested; no claim about image-driven minute shifts in the 3B will be made from E2.
4. **Notebook split:** F1a = InternVL3-2B standard + text_light (confirmatory, run first); F1b = Qwen-7B standard (descriptive); F2 = InternVL3-8B standard (descriptive). Each prints a time extrapolation after the first pair. Decoding as before (greedy; Qwen `--rep_penalty 1.0`).

**2026-11-08, third entry (PRE-RUN for notebook G0b; written before any bf16/fp32 Gemma result exists; Gemma is outside the pre-registered E0-E3 scope, so this is a record of a decision, not a change of any rule).**
1. **G0 result (facts):** gemma-3-4b-it and gemma-3-12b-it in float16 on 2 x T4 (transformers 5.20.2) give non-finite logits on every prompt (3 image prompts, 3 text-only prompts), all 11 generations per model are empty strings, 0/5 describe replies parse; image-token count (256), pixel_values shape (1,3,896,896), single BOS and token_type_ids checks raised no problem. Peak memory 5.9/2.7 GiB (4B) and 13.6/9.7 GiB (12B).
2. **Decision of the project owner:** Gemma runs in its native bfloat16 because fp16 overflows (measured in G0); the other families (Qwen2.5-VL, InternVL3) stay fp16, so the dtype is a documented per-family difference. The final choice of dtype for Gemma depends on notebook G0b.
3. **G0b contents:** (a) a diagnostic of gemma-3-4b-it in fp16 that records, per module in execution order, non-finite counts and max |value|, for `sdpa` and `eager` attention, text-only and with an image (read-only; nothing is changed in the model); (b) bf16 smoke for 4B and 12B; (c) fp32 smoke for 4B only (backup; the 12B does not fit in fp32 on 2 x T4). Each smoke reports finite logits, raw answers, parseability, peak memory, load time and seconds per generation (text-only and with image).
4. **Code:** `adapters.py` gets the existing environment option `CLOCK_DTYPE` (float16 default, bfloat16, float32) for the Gemma adapters only (InternVL already read the same variable; its code and the Qwen code are unchanged, checked by an AST comparison against the committed file). A cosmetic print bug (the 12B load message named the 4B repo) is fixed.
4b. **Rationale for the paper (not a rule):** InternVL3 uses a Qwen2.5 language backbone, so Gemma (independent LLM lineage) is the real test of cross-family generality; if Gemma proves impractical, fallback third families with non-Qwen LLMs are to be chosen from a list prepared by the project owner's request.

**2026-11-08 (POST-RUN for F1a (E2, InternVL3-2B) and G0b (Gemma precision); facts only, no rule or threshold changed). Tables: `results/mech_e2/E2_report.txt`, `results/mech_e2/F1a_extra_report.txt`, `results/gemma/G0b_report.txt`; scripts `analysis/analyze_E2.py`, `analysis/analyze_F1a_extra.py`, `analysis/analyze_G0b.py`.**
1. **F1a:** both steps OK (standard 85.5 min, text_light 81.4 min, total 2.79 h; per-pair about 48-51 s; the notebook's own extrapolation was 1.3 h per variant, i.e. 78 min). Verify gate passed: V1 0/100 generations changed; V2 follow-B 0.18 direct vs 0.18 patched, identical parsed answers 100/100; V3 30/30 pairs moved > 0.1 nat; V4 masks non-empty (example sizes: minute-hand 12-16 tokens, hour-hand 8-9, other 235-238 of 256). Overlays (20 PNGs) are in `results/mech_e2/internvl3-2b/overlays/`; they were inspected by the analyst for two clocks only (hand tiles cover the drawn strokes), the project owner is to inspect the rest.
2. **E2 standard variant (confirmatory rules, 100 pairs):** unpatched A: stays-A 0.660, follow-B 0.020. Rule 1 (minute-hand patching >= 5 pts over the same-minute control, CI excluding 0, Holm, AND above the other-region patch at that layer): NOT supported. Minute-hand diff alone is +10.0 pts at layer 0 (CI [5, 16], Holm p 0.042) and +10.0 at layer 14 (Holm p 0.045), but minute-hand minus other-region is +1 to +4 pts with CIs including 0 at every layer (the other region, 238 tokens, gives +6 to +9). Rule 2 (cue/consistency account: only all-token or non-hand patching moves the answer, minute-hand upper bound < 3 pts): not supported (minute-hand CI upper 16 pts; all-token max +17 pts). All-token patch: +16 pts at layer 0 (CI [9, 24]); effect +12 at layer 17, +8 at layer 21 (Holm p 0.365), +2 at layer 24.
3. **E2 text_light variant (positive control; hour in text, minute only in the image):** unpatched A stays-A 0.500, follow-B 0.030. All-token patch +34 to +36 pts (layers 0-14), +29 (17), +20 (21), -1 (24). Minute-hand patch +19 to +21 pts at layers 0-17 (all Holm p 0.006), +10 at layer 21 (Holm p 0.034), 0 at layer 24. Pre-registered Qwen-3B text-light rule applied to the 2B (minute-hand >= +10 pts, CI excluding 0, Holm): met (best layer 10, +21.0 pts, CI [13, 29]); positive-control condition (all-token >= 20 pts) met. The Qwen-3B part of that rule is not tested (3B dropped from E2, entry above). Not in the pre-registered rules (descriptive, no multiplicity correction): minute-hand minus other-region +10 to +13 pts at layers 0-17 (CIs exclude 0), about 0.55-0.60 of the all-token effect carried by 13 of 256 tokens, 1.4-1.6 pts per token versus 0.03-0.04 for the other tokens.
4. **Where answers go:** standard, all-token patch at layer 0: stays-A 0.66 -> 0.39, follow-B 0.02 -> 0.18, other answers 0.32 -> 0.42. text_light, all-token layer 0: stays-A 0.50 -> 0.03, follow-B 0.03 -> 0.43.
5. **Layer comparison (descriptive):** a patch at hidden-state index k only affects layers >= k. In text_light the all-token effect is +35 (k=14), +29 (17), +20 (21), -1 (24), so image-driven minute evidence is read up to about layer 23 and not at all by layers >= 24; E1 (InternVL3-2B, exploratory because the machinery check failed) had its largest window at layers 16-20 and nothing at 24-28; E3d (2B) strongest window 16-24.
6. **G0b (gemma-3-4b-it, fp16 diagnostic):** logits non-finite for text-only and with-image prompts, with `sdpa` and with `eager` attention (identical). The first non-finite output is decoder layer 5 (1 inf element, no nan); the output of layer 5's sub-modules (e.g. post-feedforward norm, max 21568) is finite. Finite maxima of the residual stream at decoder-layer outputs: layer 1 14632, layer 2 20080, layer 3 25200, layer 4 51392 (fp16 max 65504); same values for text-only and image prompts. No vision-tower or projector module is non-finite before the language model.
7. **G0b smoke:** 4B bf16 PASS (finite logits; load 6 s; peak 7.8 / 2.7 GiB; text-only 0.77 s per generation, with image 4.69 s, of which prefill 4.60 s); 4B fp32 PASS (load 10 s; peak 11.5 / 5.4 GiB; text-only 0.72 s, with image 3.53 s). Raw answers to the three image prompts are identical in bf16 and fp32 ('10:27', '09:45', '4:00' for true 10:10, 3:45, 7:20); the describe replies differ in places. describe_check.py regexes fully parsed 1/5 (bf16) and 2/5 (fp32) replies because Gemma wraps numbers in markdown bold. 12B bf16 FAILED (rc=1): CUDA out of memory (1 GiB allocation) inside the SigLIP vision tower's SDPA at the first with-image forward on GPU 0 (14.38 GiB in use of 14.56); device map {0: 29, 1: 25} module groups; the 12B has no timing or memory result.

**2026-11-08, fourth entry (PRE-RUN for G0c and G1; decisions of the project owner; written before any G0c/G1 result exists; no hypothesis or threshold changed).**
1. **E2 verdicts as computed (F1a):** standard variant: rule 1 (minute-hand region) and rule 2 (cue/consistency) not supported; text-light variant: the minute-hand rule (>= +10 pts) met. The minute-hand-minus-other-region comparison in the text-light variant is exploratory (not pre-registered).
2. **Gemma dtype:** bfloat16 for both sizes (gemma-3-4b-it and gemma-3-12b-it); the other families stay fp16. In addition, a **4B fp32 cross-check**: the same `eval_behavior.py` run on the 500 behaviour images with `CLOCK_DTYPE=float32`, written to `results/gemma-3-4b-it_fp32_crosscheck/results.csv`, to measure the dtype effect (no rule; the paired difference in exact-match accuracy and the per-image agreement are reported as descriptive).
3. **Parsing change for ALL models:** a markdown-stripping step (`textnorm.normalize_reply`: removes runs of `*`, backticks, runs of `__`, and `~~`) is applied inside every answer parser before matching (`eval_behavior.parse_time_answer`, `describe_check.parse_reply` / `parse_hour_tolerant`, the integer parse of `text_only_v2.score`, `mech_lib.parse_hhmm`, `eval_prompt2`, `text_only_check`). The raw reply is stored unchanged; `format_ok` (bare-answer check) stays on the raw reply. Reason: Gemma writes `**3**`, which the describe regexes did not match (G0b: 1/5 and 2/5 replies fully parsed). Check (`analysis/check_normalization_noop.py`, `results/normalization_noop_check.txt`): over 54,810 saved replies of the four existing models (all result CSVs, incl. the rp 1.05 sensitivity folders), the old and the new parsers give identical outputs for every reply, and the new code reproduces every stored parsed column. Fact found in that check: 118 of those replies (all InternVL3-8B text-only / conversion replies, truncated long explanations) DO contain `*` characters; none of them changes any parsed value. The earlier plan sentence that the existing replies contain no markdown is therefore not exactly true; the claim that results are unchanged is true.
4. **G0c (12B memory test):** `max_memory` caps for the Gemma adapters only (env `CLOCK_MAX_MEMORY`, default unset = unchanged load call); `gemma_diag.py --mode ladder` tries (GPU 0 / GPU 1) 9.5 / 13.5, 10.5 / 13, 11 / 12.5 GiB in separate processes and stops at the first smoke PASS. Reason: G0b's 12B bf16 run ran out of memory in the vision tower on GPU 0.
5. **G1 (Gemma-3-4B, bf16):** gate (smoke), then `text_only_v2`, `conversion_test`, aggregates, `cap_check` (32 tokens), `describe_check`, `eval_behavior`, the fp32 cross-check (item 2), and the 4B probe (`probe.py --stage all` on `data_balanced/`). Same prompts, trial sets and greedy decoding as the other models. If the gate fails, nothing else runs. Priority: G0c and G1 before F1b and F2.

**2026-11-09 (POST-RUN for G0c; facts only). Source: `match_G0c_results/` (zip `mech_G0c_results.zip`).**
1. gemma-3-12b-it bf16 passed the smoke on the first cap, `max_memory = 0:9.5GiB,1:13.5GiB` (tries 2 and 3 were not run). Device map: GPU 0 19 module groups, GPU 1 32, **disk 3** (accelerate offloaded three module groups to disk; the log records counts only). Peak memory GPU 0 11.3 GiB, GPU 1 13.5 GiB. Finite logits on all 6 prompts. Seconds per generation (5 tokens, 3 repeats): text-only 2.06 s, with image 7.87 s. Load 198 s.
2. The describe_check strict regexes parsed 1/5 of the 12B describe smoke replies (layout "Minute hand: The long hand is pointing at the number N"); with the markdown stripping on, the time parsed 5/5 and the tolerant hour parse 5/5.

**2026-11-09, fifth entry (PRE-RUN for G2a/G2b and the describe parser; decisions of the project owner; written BEFORE any G1 describe result has been seen and before any G2 result exists; no hypothesis or threshold changed).**
1. **G2 (Gemma-3-12B, bf16)** is split into two notebooks so each stays well under Kaggle's 12 h limit: G2a = gate + `text_only_v2` + `conversion_test` + aggregates + `cap_check` (guess ~4.6 h); G2b = gate + `describe_check` (data_positions, 480) + `eval_behavior` (data, 500) (guess ~4.0 h). Both use `CLOCK_MAX_MEMORY=0:9.5GiB,1:13.5GiB`, `CLOCK_DTYPE=bfloat16`. No 12B probe, no fp32 cross-check (does not fit). They are launched only after G1's gate and first steps have succeeded. Disk offload of 3 module groups is accepted (same arithmetic, slower); the full device map, counts and cap are recorded in `run_config.json` (text_only_v2 / conversion_test via `Gemma3Adapter.run_config_notes`; the gate step writes `results/gemma/<adapter>_<dtype>/run_config.json`).
2. **Tolerant describe parse.** Added `describe_check.MIN_RE_TOL` / `parse_minute_tolerant` (same construction as the existing `HOUR_RE_TOL`; the original regex is tried first) and two additive columns `said_minute_number_tolerant`, `minute_desc_ok_tolerant` (next to the existing tolerant hour columns). The strict columns and `parse_reply` are unchanged.
3. **No-op check for making the tolerant parser primary for all models** (`analysis/check_describe_parser_noop.py`, `results/describe_parser_noop_check.txt`): over all 12 saved describe CSVs (the 4 per-model headline files, 5 rp 1.05 sensitivity files, 3 earlier describe runs), recomputing every describe metric from the saved replies: the strict recomputation reproduces every stored value (0 mismatches). Tolerant vs strict: **88 cell differences, so it is NOT a no-op.** All are hour-hand parses recovered by the tolerant hour pattern (e.g. "pointing just past 12"); minute parses differ only in qwen2.5-vl-7b rp 1.05 (2 cells). Headline file affected: internvl3-8b `describe.csv` (20 hour parses recovered; `hour_desc_ok` 66.9% -> 70.6%; the same 20 in its rp 1.05 sensitivity file); all other headline files (internvl3-2b, qwen2.5-vl-3b, qwen2.5-vl-7b) are identical. Because the condition set by the project owner ("if it is a no-op") is not met, the tolerant parser is NOT made primary: the strict columns stay primary for the four existing models, and the tolerant columns (hour and minute) are reported alongside for every model including Gemma. Whether to switch all six models to one tolerant primary parser (changing InternVL3-8B's headline hour_desc_ok) is left to the project owner.
4. After G1 finishes, 30 random Gemma-4B describe replies with strict and tolerant parsed values are printed for hand-checking (`analysis/print_describe_sample.py`).

**2026-11-09, sixth entry (PRE-RUN for the describe analysis of G1/G2; decision of the project owner; written BEFORE any Gemma describe result has been seen (G1 is running with the earlier `describe_check.py`; nothing from it has been looked at); no hypothesis or threshold changed). It supersedes item 3's last sentence of the fifth entry.**
1. **Decision:** the TOLERANT describe parse (`said_minute_number_tolerant`, `said_hour_number_tolerant`, `minute_desc_ok_tolerant`, `hour_desc_ok_tolerant`; original regex tried first, then the tolerant pattern) is the PRIMARY parse for ALL six models in the cross-model comparison. The original STRICT columns (`said_*_number`, `minute_desc_ok`, `hour_desc_ok`) are kept and reported beside it as a SENSITIVITY table for the original four models (and for Gemma).
2. **Reason:** the strict parser fails systematically on specific phrasings ("pointing just past 12"; Gemma's "**Minute hand:** The long hand is pointing at the number N"), which biases a cross-model comparison. The known effect on the existing models is small and conclusion-neutral: only InternVL3-8B changes (hour_desc_ok 66.9% -> 70.6%; both-hands-exact 216 -> 234 of 480; final-answer-wrong given both exact 7.4% -> 6.8%; with the +/-1 minute number 26.5% -> 24.8%); InternVL3-2B, Qwen-3B and Qwen-7B are identical. The decision is made before any Gemma describe result has been seen.
3. **What changes in code:** `compare_families.py` (describe rows: tolerant = primary, strict = rows marked "SENSITIVITY"; tolerant columns are recomputed from the saved raw answers when a CSV lacks them). The describe CSV columns written by `describe_check.py` are unchanged (strict stored, tolerant columns added). `describe_check.py --recompute_tolerant` now adds the MINUTE tolerant columns too, so G1's `describe.csv` (written by the earlier code, which has strict + hour-tolerant columns only) gets them offline from its raw replies, with no re-run; it verifies the original columns are unchanged.
4. **What does NOT change:** the E3e clock-set selection and every E3/E1/E2 result (`mech_e3.py` selected clocks with the strict columns; `analysis/analyze_notebook_D.py`, `analyze_Q_D3IVL.py`, `selection_bias_check.py` reproduce those selections and keep using the strict columns, because re-selecting clocks after the run would change what the pre-registered experiment used). Any later use of tolerant-based clock sets would be a new, labelled analysis.
5. **Hand-check material:** `results/describe_recovered_internvl3-8b.txt` (the 20 InternVL3-8B replies the tolerant parse recovers; 18 of the 20 recovered hour numbers equal the true hour, 2 are the model's own wrong hour, e.g. true 01:15 read as "just past 12"). After G1, `analysis/print_describe_sample.py` prints 30 random Gemma-4B describe replies with strict and tolerant parsed values.

**2026-11-09 (POST-RUN for G1, Gemma-3-4B; facts only, no rule or threshold changed). Tables: `results/gemma/G1_report.txt`; script `analysis/analyze_G1.py`, merge `analysis/merge_G1.py`.**
1. All G1 steps completed (gate OK; 6.81 h). Behaviour exact 0.4% (bf16) vs 0.6% (fp32); 68% of the 500 raw replies identical across dtypes. Text-only minute accuracy 32.4%, hour->minute leaks 132/432, 48 text-only replies truncated at 32 tokens.
2. `describe_check.py --recompute_tolerant` on G1's `describe.csv` recovered 0 replies (strict and tolerant agree for the 4B replies). The 30 random replies were printed for hand-checking; the 20 recovered InternVL3-8B cases were printed earlier.
3. `analysis/selection_bias_check.py` now has `--parse tolerant|strict` with tolerant as the default (per the sixth entry's primary-parser decision); strict output is kept beside it. This changes item 4 of the sixth entry for that one script only (it is a cross-model analysis, not the E3 clock-set reproduction); `mech_e3.py`, `analyze_notebook_D.py` and `analyze_Q_D3IVL.py` still use the strict columns.

**2026-11-09, seventh entry (PRE-RUN for the Gemma-4B 128-token re-run and for G2a; decisions of the project owner; written before any 128-token Gemma result exists; no hypothesis or threshold changed). Basis: `results/gemma/G1_truncation_report.txt`, script `analysis/analyze_G1_truncation.py`.**
1. **Facts from G1 (32-token cap):** Gemma-4B replies that reached the cap: `text_only_v2` 53 of 2016 (48 in task b / phrasing b2, 5 in a2), `conversion_test` 48 of 1728, ALL in the text_only condition and ALL with lead_in_id 1 (matching / blank / mismatched: 0). Every cap-hit conversion reply is unparsed (48 unparsed = 48 hits). (Per-condition counts use reply length >= 100 characters as the hit marker because the gated tokenizer is not cached locally; this reproduces the tokenizer-based Kaggle `cap_check` totals 53 and 48 exactly and no reply has a length between 60 and 109 characters.)
2. **Effect on the conversion comparison:** on the 384 of 432 keys parsed in all four conditions (no cap hit in any), minute accuracy is text-only 36.5%, matching 44.3%, blank 45.8%, mismatched 43.8% (all-trials: 32.4 / 46.1 / 47.9 / 44.9). Exact McNemar, Holm over 3: text vs matching p = 0.017, vs blank p = 0.0011, vs mismatched p = 0.017; matching vs blank p = 0.67 (only-matching 66, only-blank 72). So truncation explains part of the apparent image gain (the text-only vs blank gap shrinks from 15.5 to 9.3 points), not all of it, and blank is still as good as the matching clock.
3. **Re-run (proposed; to be run as a separate notebook, not yet built):** Gemma-4B `text_only_v2` + `conversion_test` at `max_new_tokens=128`, bf16, same prompts and trial sets; same rule as the InternVL3-8B re-run (raise the cap for a model that hits it; greedy replies that finished before 32 tokens are unchanged). Success = `cap_check` at 128 reports 0 hits in conversion_test and <= 5 in text_only_v2; otherwise the 32-token numbers stay primary with the 128-token run as sensitivity, as for the 8B. If successful, the 128-token results become the Gemma-4B headline text-only/conversion numbers; the 32-token files are kept (renamed, not overwritten) as the sensitivity. describe_check (cap 120) and eval_behavior (cap 16, replies of 5 tokens) are not affected and are not re-run.
4. **G2 (Gemma-12B):** `text_only_v2` and `conversion_test` run at `max_new_tokens=128` from the start (notebook G2a updated: both steps pass `--max_new_tokens 128`, `cap_check` at 128; step guesses 90 and 220 min; notebook guess ~5.0 h, range 3.5-8 h, a guess). Same rule and same success criterion as item 3. G2b is unchanged.

**2026-11-10 (POST-RUN for G1b, Gemma-3-4B at max_new_tokens=128; facts only, no rule or threshold changed). Source: `match_G1b_results/` (zip `mech_G1b_results.zip`); table `results/gemma/G1b_report.txt`; script `analysis/analyze_G1b.py`.**
1. All 7 steps OK (gate 2.8 min, text_only_v2 27.2 min, conversion_test 118.8 min; total 2.49 h; guess was 2.5-3 h). `cap_check` at 128: text_only_v2 6 of 2016 reached the cap (rule: <= 5), conversion_test 6 of 1728 (rule: 0). The success rule written in the seventh entry is NOT MET on both counts; per that entry the 32-token files stay primary and the 128-token run is a sensitivity. No Gemma-4B files were renamed, swapped or merged into `results/`.
2. Replies that had finished before 32 tokens are identical at 128 (text_only_v2 1963/1963, conversion_test 1680/1680). All 53 and 48 truncated replies changed; 0 replies are unparsed at 128. The 6 remaining cap hits (all in the text_only condition, lead-in 1) are parsed (3 of 6 correct; hit identified by reply length >= 430 characters, because the gated tokenizer is not cached locally).
3. Conversion text-only at 128: minute accuracy 170/432 = 39.4% (32 tokens: 140/432 = 32.4%); hour->minute leaks 133 (32 tokens: 132). Text-only vs matching / blank / mismatched (minute accuracy 46.1 / 47.9 / 44.9%): exact McNemar Holm p = 0.035 / 0.0030 / 0.052; matching vs blank p = 0.58.

**2026-11-10, eighth entry (PRE-RUN for G2a, Gemma-12B; decision of the project owner; written before any 12B result exists; no hypothesis or threshold changed). It supersedes item 4 of the seventh entry (128 tokens) for G2a only.**
1. **Gemma-4B outcome stays as logged:** the 128-token re-run (G1b) did not meet the success rule (6 cap hits in text_only_v2, rule <= 5; 6 in conversion_test, rule 0); the 32-token files are the headline and the 128-token run is a reported sensitivity; nothing was swapped.
2. **G2a (Gemma-12B):** `text_only_v2` and `conversion_test` run at `max_new_tokens=256` (was 128 in the seventh entry), `cap_check` at 256. Reason: the 4B still reached the 128 cap in 6 of 2016 and 6 of 1728 replies (long chain-of-thought answers); a higher cap does not change greedy replies that finish early (verified at 4B: 1963/1963 and 1680/1680 identical between 32 and 128 tokens).
3. **Success rule unchanged:** `cap_check` at 256 reports 0 hits in conversion_test and <= 5 in text_only_v2. If it does not, the 256-token numbers are still the 12B text-side results (there is no 32-token 12B run to fall back on) and the residual hits are reported; whether to re-run higher is then a decision for the project owner.
4. **Runtime (a guess):** G2a about 6 h, range 4-9 h (was 5.0 h, range 3.5-8 h at 128); step guesses 100 min (text_only_v2) and 240 min (conversion_test). The extra time depends on how many 12B replies run long (4B: 2.6% of text-only trials; each long reply costs up to ~80 s at the measured ~0.3 s per token). G2b is unchanged (describe_check cap 120, eval_behavior cap 16).

**2026-11-10, ninth entry (PRE-RUN, design and predictions for the clock-robustness experiment R: styled synthetic clocks and real photos; decisions of the project owner; written BEFORE any styled-set, real-photo or R-experiment result exists and before any R code is built; no existing hypothesis or threshold changed; R is outside the pre-registered E0-E3 scope).**
1. **Question.** Is the failure to read analog clocks specific to the default synthetic style (white face, numerals, 60 ticks, thick hands), and does it persist on real photographs?
2. **Styled sets.** Rendered from the EXISTING `data.csv` (hour, minute) lists of `data/` (500 clocks, seed 42) and `data_positions/` (480 clocks) with identical filenames, so each styled image is paired with its standard-style original; `conversion_test` styled images use the same (N, H, image_seed) keys as the standard run. Only the style parameters of `clocks.generate_clock_image` change: **S1 thin hands** = `hour_hand_thickness=3`, `minute_hand_thickness=1.5` (defaults 9 and 4), everything else default; **S2 ticks only** = `show_numbers=False`, everything else default (ticks, thicknesses, colours, 512 px). S3 (colour) is not run now. Check before use: re-rendering the default style from `data.csv` must reproduce the existing images pixel-for-pixel.
3. **Run matrix.** S1 and S2: Qwen2.5-VL-3B, InternVL3-2B, Gemma-3-4B-it (bf16, same code and caps as the headline runs): `eval_behavior` (500 images), `describe_check` (480 images, prompt unchanged, tolerant parse primary per the sixth entry, strict reported beside it), and `conversion_test` conditions `matching_clock` and `mismatched_clock` with styled images (432 trials each); the text_only and blank conditions are style-independent and are taken from the existing standard runs (Gemma-4B: the 32-token G1 files, the headline). Qwen2.5-VL-7B: S2 only, `eval_behavior` + `describe_check` (for prediction 4). InternVL3-8B and Gemma-12B are not run on styled sets. Decoding as in the headline runs (greedy; rp 1.0 for Qwen; max_new_tokens 16 / 120 / 32).
4. **Real photos.** Dataset: TickTockVQA, Hugging Face mirror `monica-sekoyan/TickTockVQA-segmented` (paper: "It's Time to Get It Right", arXiv 2603.08011); evaluation only, no redistribution (images are downloaded inside the Kaggle session and are not placed in the repository or in any shared dataset; only a provenance manifest (sample_id, hour, minute, source, per-row licence) is kept); the paper and the dataset are cited. Rows: split `test`, per-row `license == "CC BY 4.0"` (read from the datasets-server statistics on 2026-11-10: test licence values "CC BY 4.0" 3735, "Unknown/Custom" 1117; sources of the CC BY 4.0 rows are coco2017 1929 and OpenImages 1806); image = the `image` column (segmented clock crop); sample = 500 rows drawn without replacement with `numpy.random.RandomState(0)` from the CC BY 4.0 rows sorted by `sample_id`; `test_small` is not used (the card asks not to tune on it). Labels: `hour` compared modulo 12 and `minute` exactly, as for the synthetic sets. Caveat recorded now: the licence is the mirror's per-row label; I could not verify that it describes the image licence rather than the annotation licence for the COCO rows. Models: all five (Qwen-3B, Qwen-7B, InternVL3-2B, InternVL3-8B, Gemma-4B bf16), `eval_behavior` + `describe_check`, same prompts and caps; Gemma-12B not run.
5. **Reference values (standard style, existing runs; fixed here before the R data exist).** Behaviour exact HH:MM over 500 images: Qwen-3B 1.8%, InternVL3-2B 8.8%, Gemma-4B 0.4%, Qwen-7B 12.6%, InternVL3-8B 9.8%; hour (mod 12): 26.2 / 57.8 / 22.4 / 69.4 / 63.2%. Conversion, minute-correct only-image minus only-text trials (matching clock vs text-only, 432 keys): Qwen-3B -44 (79 only-text, 35 only-image), InternVL3-2B +95 (29 / 124), Gemma-4B +59 (46 / 105). Hour->minute leaks in the matching_clock condition (label_answer == leak): Qwen-3B 15, InternVL3-2B 34, Gemma-4B 77. Blank condition minute accuracy Gemma-4B 207/432 = 47.9%. Tolerant describe minute-exact Qwen-7B 84.4%.
6. **Predictions (P1, P2, P3, P5, P6 confirmatory; P4 exploratory).**
   - **P1** Style does not rescue reading: behaviour exact HH:MM (over all 500 images; parse failures count as wrong) is <= 20% in every run cell (Qwen-3B S1 and S2, InternVL3-2B S1 and S2, Gemma-4B S1 and S2, Qwen-7B S2 = 7 cells). Supported if all 7 cells hold.
   - **P2** The image effect keeps its direction: D = (trials only the styled matching clock gets right) - (trials only text-only gets right) on the 432 conversion keys has the standard sign in all 6 cells: Qwen-3B D < 0, InternVL3-2B D > 0, Gemma-4B D > 0 (S1 and S2 each). Qwen-7B and InternVL3-8B are excluded (ceiling). Supported if all 6 cells match; exact McNemar p reported per cell, Holm over the 6.
   - **P3** Gemma-4B's image effect is not reading: |minute accuracy(styled matching clock) - minute accuracy(blank)| < 5 percentage points in both S1 and S2 (paired on the same 432 keys). Supported if both.
   - **P4 (exploratory)** Ticks-only hurts number-based description: Qwen-7B S2 tolerant describe minute-exact <= 74.4% (standard 84.4% minus 10 points). InternVL3-8B is not run on styles, so it is not evaluated. The same contrast is reported descriptively for the other models.
   - **P5** Hour->minute leaks persist with a styled matching clock: leak count >= 70% of the standard-style matching-clock count: Qwen-3B >= 11, InternVL3-2B >= 24, Gemma-4B >= 54 (S1 and S2 each; 6 cells). Supported if all 6 (note: Qwen-3B's reference is small, 15).
   - **P6** Real photos: exact HH:MM <= 10% for all five models, and hour accuracy (mod 12) >= 25% for Qwen-7B and InternVL3-8B. Supported if all hold.
7. **Statistics.** Rates with 95% Wilson intervals; styled vs standard (same (hour, minute) times) compared by exact McNemar (descriptive); Holm over the confirmatory tests of the same prediction. No other comparison is confirmatory; photo subgroup analyses (by source, by numeral style, by crop size) are exploratory.
8. **Code and checks.** Additive options `--style` and `--conditions` on `conversion_test.py`; the defaults must leave every existing output byte-identical, shown by a regression check (old vs new code, same inputs, identical CSV and image bytes) before use. Notebook order after G2a + G2b: Gemma E3d minimal run, then styled sets, then real photos.

**2026-11-10, tenth entry (PRE-RUN, design and decision rule for the Gemma-3-4B E3d minimal run; decisions of the project owner; written BEFORE any Gemma E3d result exists and before the Gemma E3d notebook is built; exploratory (Gemma is outside the pre-registered E0-E3 scope); the E3d rule and thresholds are unchanged).**
1. **Run.** `gemma-3-4b-it`, bf16, `mech_e3.py` verify gates + part d in the minimal mode: three cells only: `intact`, `H_all` (block the minute-slot queries from the first ':' of the answer onward from the hour-token keys, at all 34 decoder layers) and `random_all` (random key span of the same size drawn from non-hour, non-number prompt positions, all layers), all as implemented for the other models. Trials: the 432 text-only prompts (3 lead-ins x 12 x 12), greedy decoding without a KV cache, `max_new=24`.
2. **Decision rule (unchanged).** Verdict "supported" only if all hold: leak count falls >= 30% relative (intact -> `H_all`) AND accuracy rises >= 4 points, both paired CIs excluding 0, AND the random-span control changes accuracy by < 2 points. The verdict is reported either way, labelled exploratory. Prior expectation (a prediction, not a rule): the leak and accuracy clauses are met (intact leaks were 132 of 432 in G1); the control clause is uncertain (it failed for InternVL3-2B, +3.0 points, and held for Qwen-3B).
3. **Gemma-specific handling (decided now).** In G1, 48 lead-in-1 trials produce long chain-of-thought replies that do not reach a time within `max_new=24`. Primary analysis = the keys whose INTACT reply is parsed; sensitivity = all 432 keys with unparsed replies counted as neither correct nor leak. This differs from the all-key analysis used for the other models and is recorded as a deviation for Gemma.
4. **Gates (gating unless marked; thresholds as implemented for the other models).** The existing verify gates G1-G9 (token spans, teacher-forced vs free generation >= 0.95, empty block equals no hook exactly, guard, hooks fire on every layer, block changes log-probs, eager vs default agreement, eager forced on every attention layer, finite eager logits close to default) with G10 and G11 informational as before. New Gemma gates: **G12 sliding-window**: (a) the sliding-window size and the layer-type pattern are read from the model config and logged; (b) every one of the 432 prompts is shorter than the sliding window, so local layers see the same keys as global layers; (c) in a probe forward, the additive mask received by each of the 34 attention modules has identical visible-key sets in local and global layers; any prompt >= the window makes the run invalid. **G13** the blocking hook fires on all 34 layers. Eager-vs-default agreement is judged in the same dtype (bf16). If any gating check fails, no cell is interpreted and the failure is reported.
5. **Before the real run.** An offline test on a tiny random Gemma3 whose sliding window is SMALLER than the prompt (so the local-layer mask path is exercised) must show that the hook edits the mask in local and global layers, that the unblocked path is unchanged, and that G12 flags a window shorter than the prompt. It tests code, not science.
6. **Out of scope:** the full window sweep (21 cells), Gemma-12B, and any claim beyond Gemma-4B.

**2026-11-10, eleventh entry (BUILD facts for experiment R and the Gemma-4B E3d minimal run; facts only, no prediction, rule or threshold changed; written after the ninth and tenth entries and before any R or Gemma-E3d result exists).**
1. **Default-style pixel check (ninth entry, item 2).** Re-rendering the default style from `data_positions/data.csv` locally (matplotlib 3.11.2) does NOT reproduce the existing images pixel-for-pixel: about 1.9k of 262k pixels per image differ (max difference 255), constant across the 60 images compared, all inside the numeral ring (pixel radius 95-131 of the 512 px image); hands, ticks, rim and face outside that ring are identical. Cause: numeral glyph rasterisation differs between matplotlib versions. The gate in `make_styled_set.py --check` is therefore "identical everywhere outside the numeral ring (radius 85-140 px)", with the ring differences reported; the notebooks R1 and R2 run it on `data/` and `data_positions/` first and skip everything if it fails. S2 (no numerals) is unaffected; S1 comparisons against the standard images carry this glyph-rasterisation difference (numerals only). The check on `data/` could not be run locally (the folder is not in the repository) and runs on Kaggle.
2. **Regression check for the additive `conversion_test.py` options** (`tests/test_conversion_style_regression.py`, old file frozen in `tests/fixtures/conversion_test_before_style.py`, tiny random Gemma3, CPU): with default arguments the old and the new code give a byte-identical results CSV, image manifest and all 11 rendered PNGs, and an identical run_config (minus the date) with no `style` / `conditions_run` keys; `--style s1_thin|s2_ticks` and `--conditions` write to their own file and image-folder names, keep the prompts and image keys of the default run, and unknown styles/conditions are rejected (20 checks, all pass). `mech_e3.py` / `mech_lib.py` changes are additive (`has_sliding_layers` on `Gemma3Adapter` only; the Gemma gates run only for adapters with that flag).
3. **Gemma E3d additions (tenth entry).** Implemented as specified: G12a/b/c and G13 in `mech_e3.py` `part_verify`, `--d_intact_parsed` (writes `e3d_summary_intactparsed.csv` and `e3d_prereg_verdict_intactparsed.json` beside the all-key files; only the intact-parsed verdict file carries an extra `subset` key). The standard outputs are unchanged: on the tiny-InternVL smoke run (verify + part d, minimal cells) the old and the new `mech_e3.py` / `mech_lib.py` give byte-identical `e3d_knockout.csv`, `e3d_summary.csv`, `e3d_prereg_verdict.json` and `verify.json`. One detail: G12b requires `max prompt tokens + max_new (24)` < sliding window (stricter than "every prompt shorter than the window", because the no-cache loop re-feeds the generated tokens). Offline test `tests/test_gemma_e3d_offline.py` (tiny random Gemma3, character-level tokenizer, 3 layers LLG): with window 32 < prompt (178-221 tokens) the local layers see 5200 vs 15931 key positions for the global layer, the blocking hook removes the hour keys for the late queries in EVERY layer (local and global) and leaves earlier rows and all other keys untouched, an empty block equals no hook, G12b and G12c flag the short window and G13 passes; with window 1024 G12a/b/c and G13 all pass; the intact-parsed summary uses exactly the keys whose intact reply parsed (synthetic file, 18 of 24). On the random tiny model G7b (blocking changes log-probs) did not pass, which only says the toy model is insensitive; it is a real-weight gate.
4. **Real photos.** `prep_ticktock_photos.py` implements the pre-registered selection (split `test`, `license == "CC BY 4.0"`, sorted by `sample_id`, `RandomState(0)`, 500 rows, `image` column, hour stored mod 12 as 1-12 with the original in the manifest); tested offline on synthetic rows (determinism, licence filter, error instead of loosening when fewer than 500 rows). Not run on the real dataset yet (needs internet). `.gitignore` now excludes `data_photos*/` and the regenerable styled folders so photos cannot be committed.
5. **Notebooks and runtime guesses (all guesses; scaled from measured step times of the standard runs).** R1 `kaggle/notebook_R1_styles_small_models.ipynb` (Qwen-3B and InternVL3-2B S1+S2, Qwen-7B S2): ~3.5 h; R2 `notebook_R2_styles_gemma4b.ipynb` (Gemma-4B S1+S2): ~7 h; R3 `notebook_R3_photos_five_models.ipynb` (five models, eval + describe on 500 photos): ~5 h; GE3d `notebook_GE3d_gemma4b_e3d_min.ipynb` (verify gates + minimal E3d): ~1.9 h. Planned order: G2a, G2b, then GE3d, then R1 and R2, then R3. The GE3d notebook needs the `clock-prior` dataset extended with `prior/gemma-3-4b-it/{conversion_test.csv,describe.csv}` (copy in `kaggle/upload_prior_gemma/`, the 32-token G1 files).

**2026-11-11, twelfth entry (POST-RUN, G2a: Gemma-3-12B-it text_only_v2 + conversion_test at max_new_tokens 256; facts only, no rule, prediction or threshold changed).**
1. Run (zip `mech_G2a_results.zip`, unzipped to `match_G2a_results/`): gate OK, all 7 steps OK, 4.39 h; bf16, `CLOCK_MAX_MEMORY` 0:9.5GiB,1:13.5GiB; recorded device map 19 groups GPU0, 32 GPU1, 3 on disk (disk offload as approved in the fifth entry); peak 11.3 / 13.5 GiB.
2. **Cap-check outcome (eighth entry rule):** at 256 tokens 0 hits in `text_only_v2` (rule <= 5) and 0 hits in `conversion_test` (rule 0); longest reply 5 characters; 0 unparsed in 2016 and 1728 trials. The rule is MET, no residual hits. 256 tokens is therefore the primary setting for the 12B.
3. Merged into `results/` with no-clobber (`analysis/merge_G2a.py`; 5 smoke images from G0c kept, not overwritten); `text_only_v2.py --aggregate` and `conversion_test.py --aggregate` re-run with the six models (rows of the other five models identical to before); `compare_families.py` re-run with six models (12B behaviour/describe not found: G2b pending).
4. Results (n = 432 per cell): text-only minute accuracy 92.4%, matching 88.7%, blank 81.9%, mismatched 83.6%; McNemar vs text-only p = 0.048 / 2.1e-10 / 2.8e-6 (text-only-better / image-better 37/21, 50/5, 52/14); hour->minute leaks 16; mismatched answers follow the image minute in 10 (2.3%). Details in FINDINGS section 8 and `results/gemma/G2a_report.txt`. No planned analysis was changed or added after seeing the data other than the descriptive tables in `analysis/analyze_G2a.py`.

**2026-11-12, thirteenth entry (POST-RUN, G2b and GE3d; facts only, no rule, prediction or threshold changed).**
1. **G2b (Gemma-3-12B describe + behaviour):** gate OK, all 4 steps OK, 4.29 h; merged no-clobber (`analysis/merge_G2b_GE3d.py`); `compare_families.py` (six models) and `analysis/selection_bias_check.py` (six models, tolerant primary and strict) re-run. Tolerant parse primary as pre-registered (sixth entry): both hands exactly right 111/480 = 23.1%, final answer wrong given both 10/111 = 9.0%; strict parse (sensitivity) reads only 7.7% of replies and gives 0/23. Behaviour: exact 14/500 = 2.8%, hour 35.6%, minute 5.4%, parse 100%.
2. **GE3d (Gemma-3-4B E3d minimal): the pre-registered verify gate FAILED, so the E3d knockout step was skipped by design; there is NO E3d result for Gemma-4B and none of the tenth-entry E3d rules was evaluated.** Failed gating checks: G8 (0.80 vs >= 0.95), G8b (0.78 vs >= 0.95), G9 (14/20 vs >= 0.9). The new Gemma gates G12a/b/c and G13 PASSED (29 local + 5 global layers, window 1024, longest prompt 62 tokens, identical visible keys, hook fires once per layer). From the saved trials the failures coincide with replies that contain no HH:MM within the token budget (11/100 free replies unparsed); the reply text is identical between the no-cache loop and the adapter generate in 98/100 trials. No gate threshold or definition has been changed; any amendment would need its own entry written before a re-run.

**2026-11-12, fourteenth entry (PRE-RUN AMENDMENT of the Gemma-4B verify gates G8, G8b, G9, written after the failed GE3d gate and BEFORE any Gemma-4B E3d result exists; the E3d decision rule, cells, prompts, thresholds of every other gate and of the E3d verdict are unchanged).**
1. **Why.** In the GE3d run (thirteenth entry) G8 (0.80), G8b (0.78) and G9 (14/20) failed. On the saved `verify_g2_trials.csv` the failures coincide with replies that are prose with no HH:MM inside the token budget (11 of 100 free replies unparsed; on the first 50 trials 40 parse = G8 0.80, 39 of them agree in parsed time between the no-cache loop and the adapter generate = G8b 0.78; reply text identical in 49/50). The gates as written count an unparsed reply as a disagreement (G8, G8b) or need a generated ':' (G9), so they measure reply format, not whether the attention paths or the block hook agree. No E3d output exists, so no outcome can have influenced this choice.
2. **Amendment (only with the new flag `--gate_parsed_only`, used only in notebook GE3d; without it `mech_e3.py` behaves as before, so Qwen/InternVL runs are unchanged).** The three gates use the same trial subset as before (G8, G8b: first 50 of the seeded 100-trial pick; G9: first 50 instead of first 20, to get enough usable keys), restricted to the USABLE keys, defined as keys whose INTACT no-cache greedy reply (eager attention, no block, max_new 24) contains a parsable HH:MM. G8: among usable keys, parsed time of the eager loop equals that of the default-attention loop. G8b: among keys whose default-attention loop reply parses, parsed time equals that of the adapter generate (max 32). G9: among usable keys, the H-span block fires at least once in free generation. **Thresholds unchanged (>= 0.95, >= 0.95, >= 0.9), plus a floor: at least 30 usable keys (of 50) for each gate, otherwise the gate fails.** The reported numbers include the usable count and the all-key agreement as information.
3. **Unchanged:** G1-G7, G8a, G8c, G12a/b/c, G13 and all non-gating checks; E3d cells `intact`, `H_all`, `random_all`, `max_new` 24, the E3d rule (leaks down >= 30% and accuracy up >= 4 points with CIs excluding 0, random-span control within 2 points), primary = keys whose intact reply parsed, sensitivity = all keys. The verdict is reported either way.
4. **Build facts for the amendment (2026-11-12, written before the re-run).** Implemented in `mech_e3.py` behind `--gate_parsed_only` (default off); notebook GE3d step 01 passes the flag; `kaggle/upload_D/clock-code/mech_e3.py` updated. Regression: on the tiny random InternVL, `verify --smoke` with the old and the new code WITHOUT the flag gives a byte-identical `verify.json` and identical output files. With the flag the tiny random model produces no parsable HH:MM, so usable keys = 0 and G8/G8b fail through the 30-key floor (G9 passes vacuously in smoke mode), which only shows the floor works; the pass path of the three amended gates has not been exercised on real weights.
