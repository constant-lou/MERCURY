import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(input_dir):
    groups = defaultdict(lambda: [0, 0])
    for path in sorted(Path(input_dir).glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                metadata = record.get("meta_data") or {}
                dimensions = {
                    "language": metadata.get("language", "unknown"),
                    "domain": metadata.get("domain", "unknown"),
                    "difficulty": metadata.get("difficulty", "unknown"),
                    "emotion": metadata.get("emotion", "unknown"),
                }
                for dimension, value in dimensions.items():
                    key = (dimension, str(value))
                    groups[key][0] += 1
                    groups[key][1] += int(bool(record.get("is_correct")))
    return groups


def run(input_dir, save_to):
    groups = summarize(input_dir)
    output_path = Path(save_to)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("dimension\tvalue\ttotal\tcorrect\taccuracy\n")
        for (dimension, value), (total, correct) in sorted(groups.items()):
            accuracy = correct / total if total else 0.0
            handle.write(f"{dimension}\t{value}\t{total}\t{correct}\t{accuracy:.4f}\n")
    print(f"saved={output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize MERCURY subgroup accuracy.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--save_to", default="results/mercury_results_table.txt")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.input_dir, arguments.save_to)
