# MERCURY

MERCURY is the final multimodal metaphor reasoning pipeline. It runs three stages:

1. role-aware multi-view reasoning;
2. cross-view evidence calibration;
3. bidirectional option-permutation verification.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Set the task-adapted model checkpoint without editing repository files:

```bash
export MERCURY_MODEL_PATH=/path/to/model/checkpoint
```

Optional variables are `MERCURY_MODEL_BASE` and `MERCURY_CONV_MODE`.

## Data layout

Dataset files belong in `data/`. Image paths in each record are resolved relative to
`images/`. Both JSON and JSONL splits are supported, including split files ending in
`_cot`.

## Run the complete pipeline

```bash
bash run_mercury.sh
```

The default splits are `E_metaphors` and `C_metaphors`. Settings can be overridden:

```bash
SPLIT_LIST="E_metaphors" MODE_LIST="none cot" INFER_LIMIT=10 bash run_mercury.sh
```

Outputs are written beneath `results/` unless `RESULT_ROOT` is provided.

## Run individual stages

```bash
python src/infer/infer_mercury.py \
  --config src/config/config_mercury.yaml \
  --split E_metaphors \
  --mode none \
  --model_name git-llava-13b \
  --output_dir results/multi_view
```

The calibration script accepts `--stage cvec` or `--stage pesc` and reads the
multi-view result directory through `--baseline_result_dir`.
