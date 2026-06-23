import argparse
import json
from pathlib import Path


def resolve_dataset_path(data_dir, split):
    candidates = (
        data_dir / f"{split}.json",
        data_dir / f"{split}.jsonl",
        data_dir / f"{split}_cot.json",
        data_dir / f"{split}_cot.jsonl",
        data_dir / "processed_data" / f"{split}.json",
        data_dir / "processed_data" / f"{split}.jsonl",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Dataset split not found: {split}")


def load_records(path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_ids(path):
    identifiers = set()
    for record in load_records(path):
        questions = record.get("questions")
        if questions:
            identifiers.update(str(item["id"]) for item in questions if item.get("id"))
        elif record.get("id"):
            identifiers.add(str(record["id"]))
    return identifiers


def result_ids(path):
    identifiers = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("id"):
                    identifiers.add(str(record["id"]))
    return identifiers


def main():
    parser = argparse.ArgumentParser(description="Validate result ID coverage.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--result", type=Path, required=True)
    arguments = parser.parse_args()
    expected = dataset_ids(resolve_dataset_path(arguments.data_dir, arguments.split))
    actual = result_ids(arguments.result) if arguments.result.exists() else set()
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    print(
        f"split={arguments.split} expected={len(expected)} actual={len(actual)} "
        f"missing={len(missing)} extra={len(extra)}"
    )
    raise SystemExit(1 if missing or extra else 0)


if __name__ == "__main__":
    main()
