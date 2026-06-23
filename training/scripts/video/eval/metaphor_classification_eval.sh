#!/bin/bash

set -euo pipefail

: "${PROJECT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${DATA_ROOT:=$PROJECT_ROOT}"
: "${OUTPUT_ROOT:=$PROJECT_ROOT/results}"
: "${VISION_TOWER:=microsoft/git-large-vatex}"
: "${CUDA_VISIBLE_DEVICES:=0}"

: "${MODEL_PATH:=$PROJECT_ROOT/checkpoints/classification/model}"
: "${QUESTION_FILE:=$DATA_ROOT/playground/data/image_data/classification/test_questions.jsonl}"
: "${ANSWERS_FILE:=$OUTPUT_ROOT/classification_answers.jsonl}"
: "${TEMPERATURE:=0.5}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -m llava.eval.model_vqa_video \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --answers-file "$ANSWERS_FILE" \
  --vision-tower "$VISION_TOWER" \
  --temperature "$TEMPERATURE" \
  --conv-mode vicuna_v1
