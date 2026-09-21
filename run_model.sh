#!/usr/bin/env bash
# run_model.sh -- run the full evaluation + interpretability pipeline for ONE
# model, in order: behavior check, describe check, text-only check, probe
# extraction + probing, --verify, and Experiment A activation patching (with
# the transfer-to-baseline analysis -- see intervene.py's module docstring).
#
# Every step writes to a <model_short_name>/ subfolder under that SCRIPT'S
# own established base directory (see adapters.output_dir_for) -- results/
# for the three eval scripts, probe_output/probe_results/ for probe.py,
# intervene_output/ for intervene.py -- so running this for several models
# never overwrites another model's results. compare_models.py knows all
# three roots.
#
# Assumes clocks.py / clocks.py --balanced / balanced_clocks.py have ALREADY
# been run once (data/, data_balanced/, data_positions/ exist) -- the clock
# images and their ground truth don't depend on which VLM is being tested,
# so this script does NOT regenerate them per model.
#
# Kaggle-friendly: paste this file's contents, or upload it, and run with
# `!bash run_model.sh <model_id> [adapter_name]` from a notebook cell (works
# the same as from a shell -- no bash-specific Kaggle setup needed beyond
# having the repo's .py files in the working directory).
#
# Usage:
#   ./run_model.sh <model_id> [adapter_name] [--smoke]
#
# Examples:
#   ./run_model.sh Qwen/Qwen2.5-VL-7B-Instruct
#   ./run_model.sh google/gemma-3-4b-it gemma-3-4b-it
#   ./run_model.sh OpenGVLab/InternVL3-2B internvl3-2b --smoke   # quick correctness check first
#
# --smoke caps every stage to a handful of images/pairs (a few minutes, not
# hours) -- ALWAYS run this first on a new adapter before committing to the
# full sweep; see adapters.py's module docstring on why Gemma3Adapter/
# InternVLAdapter are unverified against real weights until you do.

set -euo pipefail

MODEL_ID="${1:?Usage: $0 <model_id> [adapter_name] [--smoke]}"
shift

ADAPTER_NAME=""
SMOKE=0
for arg in "$@"; do
  if [ "$arg" = "--smoke" ]; then
    SMOKE=1
  else
    ADAPTER_NAME="$arg"
  fi
done

ADAPTER_ARGS=()
if [ -n "$ADAPTER_NAME" ]; then
  ADAPTER_ARGS=(--adapter "$ADAPTER_NAME")
fi

SMOKE_EVAL_ARGS=()
SMOKE_PROBE_ARGS=()
SMOKE_INTERVENE_ARGS=()
if [ "$SMOKE" = "1" ]; then
  echo ">>> --smoke: capping every stage to a handful of images/pairs <<<"
  SMOKE_EVAL_ARGS=(--max_images 5)
  SMOKE_PROBE_ARGS=(--max_images 8)
  SMOKE_INTERVENE_ARGS=(--max_pairs 1 --layers 0,1)
fi

echo "=================================================================="
echo "run_model.sh: $MODEL_ID ${ADAPTER_NAME:+(adapter: $ADAPTER_NAME)}"
echo "=================================================================="

echo ""
echo "--- 1/6: behavior check (500 clocks) ---"
python eval_behavior.py --model_id "$MODEL_ID" "${ADAPTER_ARGS[@]}" \
    --data_csv data/data.csv --images_dir data "${SMOKE_EVAL_ARGS[@]}"

echo ""
echo "--- 2/6: describe check (balanced minute-position set) ---"
python describe_check.py --model_id "$MODEL_ID" "${ADAPTER_ARGS[@]}" \
    --data_csv data_positions/data.csv --images_dir data_positions "${SMOKE_EVAL_ARGS[@]}"

echo ""
echo "--- 3/6: text-only check (no image -- perception vs. arithmetic) ---"
python text_only_check.py --model_id "$MODEL_ID" "${ADAPTER_ARGS[@]}"

echo ""
echo "--- 4/6: probe extraction + probing (needs data_balanced/) ---"
python probe.py --stage all --model_id "$MODEL_ID" "${ADAPTER_ARGS[@]}" \
    --data_csv data_balanced/data.csv --images_dir data_balanced "${SMOKE_PROBE_ARGS[@]}"

echo ""
echo "--- 5/6: --verify (MUST pass before trusting any intervention result below --"
echo "    see intervene.py's module docstring: this is not boilerplate, get_image_features"
echo "    interception looked fine by every superficial check for TWO rounds on Qwen before"
echo "    --verify caught it doing nothing) ---"
python intervene.py --verify --experiment none --model_id "$MODEL_ID" "${ADAPTER_ARGS[@]}" \
    "${SMOKE_INTERVENE_ARGS[@]}"

echo ""
echo "--- 6/6: Experiment A activation patching + transfer-to-baseline analysis ---"
python intervene.py --experiment a --model_id "$MODEL_ID" "${ADAPTER_ARGS[@]}" \
    "${SMOKE_INTERVENE_ARGS[@]}"

echo ""
echo "=================================================================="
echo "run_model.sh done for $MODEL_ID -- see results/, probe_output/probe_results/, and intervene_output/,"
echo "each under their own <model_short_name>/ subfolder"
echo "Read the --verify report BEFORE trusting Experiment A's results."
if [ "$SMOKE" = "1" ]; then
  echo "This was a --smoke run (a handful of images/pairs) -- re-run without --smoke for real results."
fi
echo "=================================================================="
