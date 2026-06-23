#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-git-llava-13b}"
SPLIT_LIST="${SPLIT_LIST:-E_metaphors C_metaphors}"
MODE_LIST="${MODE_LIST:-none}"
INFER_LIMIT="${INFER_LIMIT-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-32}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results}"
MULTI_VIEW_DIR="${MULTI_VIEW_DIR:-${RESULT_ROOT}/multi_view}"
CVEC_DIR="${CVEC_DIR:-${RESULT_ROOT}/cvec}"
PESC_DIR="${PESC_DIR:-${RESULT_ROOT}/pesc}"
EVAL_DIR="${EVAL_DIR:-${RESULT_ROOT}/evaluation}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/src/config/config_mercury.yaml}"

read -r -a SPLITS <<< "$SPLIT_LIST"
read -r -a MODES <<< "$MODE_LIST"
mkdir -p "$MULTI_VIEW_DIR" "$CVEC_DIR" "$PESC_DIR" "$EVAL_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src/infer:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

LIMIT_ARGS=()
if [[ -n "$INFER_LIMIT" ]]; then
  LIMIT_ARGS=(--infer_limit "$INFER_LIMIT")
fi

"$PYTHON_BIN" src/infer/infer_mercury.py \
  --config "$CONFIG_PATH" \
  --split "${SPLITS[@]}" \
  --mode "${MODES[@]}" \
  --model_name "$MODEL_NAME" \
  --output_dir "$MULTI_VIEW_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  "${LIMIT_ARGS[@]}"

"$PYTHON_BIN" src/infer/evidence_calibration.py \
  --stage cvec \
  --config "$CONFIG_PATH" \
  --split "${SPLITS[@]}" \
  --mode "${MODES[@]}" \
  --model_name "$MODEL_NAME" \
  --output_dir "$CVEC_DIR" \
  --baseline_result_dir "$MULTI_VIEW_DIR" \
  --baseline_model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  "${LIMIT_ARGS[@]}"

"$PYTHON_BIN" src/infer/evidence_calibration.py \
  --stage pesc \
  --config "$CONFIG_PATH" \
  --split "${SPLITS[@]}" \
  --mode "${MODES[@]}" \
  --model_name "$MODEL_NAME" \
  --output_dir "$PESC_DIR" \
  --baseline_result_dir "$MULTI_VIEW_DIR" \
  --baseline_model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --answer_max_new_tokens "$ANSWER_MAX_NEW_TOKENS" \
  "${LIMIT_ARGS[@]}"

for stage in multi_view cvec pesc; do
  input_dir="${RESULT_ROOT}/${stage}"
  status_dir="${EVAL_DIR}/${stage}"
  "$PYTHON_BIN" src/eval_mercury.py \
    --evaluate_all \
    --output_dir "$input_dir" \
    --save_dir "$status_dir"
  "$PYTHON_BIN" src/eval_mercury_sub.py \
    --input_dir "$status_dir" \
    --save_to "${status_dir}/mercury_results_table.txt"
done

if [[ -z "$INFER_LIMIT" ]]; then
  for split in "${SPLITS[@]}"; do
    for mode in "${MODES[@]}"; do
      "$PYTHON_BIN" src/validate_result_coverage.py \
        --data_dir data \
        --split "$split" \
        --result "${MULTI_VIEW_DIR}/${MODEL_NAME}_${split}_${mode}.jsonl"
      "$PYTHON_BIN" src/validate_result_coverage.py \
        --data_dir data \
        --split "$split" \
        --result "${CVEC_DIR}/${MODEL_NAME}-cvec_${split}_${mode}.jsonl"
      "$PYTHON_BIN" src/validate_result_coverage.py \
        --data_dir data \
        --split "$split" \
        --result "${PESC_DIR}/${MODEL_NAME}-pesc_${split}_${mode}.jsonl"
    done
  done
fi
