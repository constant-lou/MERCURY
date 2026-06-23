import argparse
import json
from pathlib import Path


def normalize_letter(value):
    text = str(value or "").strip().upper()
    return text if len(text) == 1 and "A" <= text <= "H" else None


def gold_answer(record):
    answer = normalize_letter(record.get("answer"))
    if answer:
        return answer
    options = record.get("options") or []
    correct_option = record.get("correct_option")
    if correct_option in options:
        return chr(ord("A") + options.index(correct_option))
    return None


def evaluate_file(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    correct = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            gold = gold_answer(record)
            prediction = normalize_letter(record.get("final_prediction"))
            is_correct = gold is not None and prediction == gold
            record["gold_answer"] = gold
            record["is_correct"] = is_correct
            json.dump(record, target, ensure_ascii=False)
            target.write("\n")
            total += 1
            correct += int(is_correct)
    accuracy = correct / total if total else 0.0
    return {"file": input_path.name, "total": total, "correct": correct, "accuracy": accuracy}


def run(input_dir, save_dir):
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)
    summaries = []
    for input_path in sorted(input_dir.glob("*.jsonl")):
        summaries.append(evaluate_file(input_path, save_dir / input_path.name))
    if not summaries:
        raise RuntimeError(f"No JSONL result files found in {input_dir}")
    summary_path = save_dir / "evaluation_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        for summary in summaries:
            line = (
                f"{summary['file']}\ttotal={summary['total']}\tcorrect={summary['correct']}"
                f"\taccuracy={summary['accuracy']:.4f}"
            )
            print(line)
            handle.write(line + "\n")
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate MERCURY result files.")
    parser.add_argument("--evaluate_all", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    run(arguments.output_dir, arguments.save_dir)
