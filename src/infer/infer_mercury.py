import argparse
import copy
import json
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config_wrapper import ConfigWrapper
from data_loader import load_data
from models import infer, load_model
from multi_view_reasoning import build_multi_view_prompts, resolve_sample_question


def parse_final_answer(response_text):
    if not response_text:
        return "A"
    text = str(response_text).strip()
    patterns = (
        r"(?:answer|option)\s+is\s*\(?([A-H])\)?",
        r"^\s*\(?([A-H])\)?\s*[\.\):：]?",
        r"答案\s*[:：]\s*\(?([A-H])\)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    matches = re.findall(r"\b([A-H])\b", text.upper())
    return matches[-1] if matches else "A"


def infer_batch(model_components, model_name, batch, mode):
    inference_inputs = []
    metadata = []
    for sample_index, sample in enumerate(batch):
        declared_language = (sample.get("meta_data") or {}).get("language")
        question = resolve_sample_question(sample, declared_language)
        language = declared_language or (
            "zh" if any("\u4e00" <= character <= "\u9fff" for character in question) else "en"
        )
        image_path = sample.get("_image_path")
        for prompt in build_multi_view_prompts(sample, language, mode):
            inference_inputs.append(
                {
                    "prompt": prompt["text"],
                    "images": [image_path] if image_path else [],
                }
            )
            metadata.append(
                {
                    "sample_index": sample_index,
                    "weight": prompt["weight"],
                    "type": prompt["type"],
                }
            )

    responses = infer(model_name)(
        inference_inputs,
        **model_components,
        temperature=0.3,
        max_new_tokens=256,
    )
    results = copy.deepcopy(batch)
    votes = [Counter() for _ in batch]
    logs = [[] for _ in batch]
    for metadata_item, response in zip(metadata, responses):
        sample_index = metadata_item["sample_index"]
        prediction = parse_final_answer(response)
        votes[sample_index][prediction] += metadata_item["weight"]
        logs[sample_index].append(
            {
                "type": metadata_item["type"],
                "pred": prediction,
                "weight": metadata_item["weight"],
                "text": response,
            }
        )

    for index, result in enumerate(results):
        prediction = votes[index].most_common(1)[0][0] if votes[index] else "A"
        result.pop("_image_path", None)
        result["final_prediction"] = prediction
        result["view_logs"] = logs[index]
        result["inference_mode"] = mode
        preferred = next(
            (entry["text"] for entry in logs[index] if entry["type"] == "direct_anchor"),
            logs[index][0]["text"] if logs[index] else "",
        )
        result["response"] = preferred
    return results


def _load_completed(paths, config_wrapper):
    completed = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = config_wrapper.get_id(record)
                if record_id is not None:
                    completed[str(record_id)] = record
    return completed


def run(args):
    config_wrapper = ConfigWrapper(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_components = None

    for split in args.split:
        for mode in args.mode:
            output_path = output_dir / f"{args.model_name}_{split}_{mode}.jsonl"
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            completed = _load_completed((output_path, temporary_path), config_wrapper)
            processed = 0
            batches = []
            current_batch = []

            for payload, sample in load_data(split=split, mode=mode):
                sample_id = str(config_wrapper.get_id(sample))
                if sample_id in completed:
                    continue
                if args.infer_limit is not None and processed >= args.infer_limit:
                    break
                sample["_image_path"] = payload["images"][0]
                current_batch.append(sample)
                processed += 1
                if len(current_batch) >= args.batch_size:
                    batches.append(current_batch)
                    current_batch = []
            if current_batch:
                batches.append(current_batch)

            if batches and model_components is None:
                model_components = load_model(
                    args.model_name,
                    args.use_accel,
                    model_path=args.model_path,
                )

            with temporary_path.open("w", encoding="utf-8") as handle:
                for record in completed.values():
                    json.dump(record, handle, ensure_ascii=False)
                    handle.write("\n")
                with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                    futures = [
                        executor.submit(
                            infer_batch,
                            model_components,
                            args.model_name,
                            batch,
                            mode,
                        )
                        for batch in batches
                    ]
                    for future in as_completed(futures):
                        for record in future.result():
                            json.dump(record, handle, ensure_ascii=False)
                            handle.write("\n")
                            handle.flush()
            shutil.move(str(temporary_path), str(output_path))
            print(f"saved={output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run MERCURY multi-view inference.")
    parser.add_argument("--model_name", default="git-llava-13b")
    parser.add_argument("--model_path")
    parser.add_argument("--config", default="src/config/config_mercury.yaml")
    parser.add_argument("--split", nargs="+", default=["E_metaphors", "C_metaphors"])
    parser.add_argument("--mode", nargs="+", default=["none"])
    parser.add_argument("--output_dir", default="results/multi_view")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--infer_limit", type=int)
    parser.add_argument("--use_accel", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
