#!/bin/bash

set -euo pipefail

: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${DATA_ROOT:=$PROJECT_ROOT}"
: "${OUTPUT_ROOT:=$PROJECT_ROOT/results}"
: "${MODEL_PATH:=$PROJECT_ROOT/checkpoints/model}"
: "${QUESTION_FILE:=$DATA_ROOT/playground/data/image_data/test_questions.jsonl}"
: "${ANSWERS_FILE:=$OUTPUT_ROOT/answers.jsonl}"
: "${VISION_TOWER:=microsoft/git-large-vatex}"
: "${CUDA_VISIBLE_DEVICES:=0}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -m llava.eval.model_vqa_video \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --answers-file "$ANSWERS_FILE" \
  --vision-tower "$VISION_TOWER" \
  --temperature 0.2 \
  --interval-type "multi_interval" \
  --no_of_intervals 4 \
  --conv-mode vicuna_v1
