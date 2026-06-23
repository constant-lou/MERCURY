import json
from pathlib import Path

from multi_view_reasoning import resolve_sample_question


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = PROJECT_ROOT / "images"


def resolve_dataset_path(split, data_dir=DATA_DIR):
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


def _load_records(path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_split(split):
    language = "zh" if split.startswith("C_") else "en"
    samples = []
    for image_record in _load_records(resolve_dataset_path(split)):
        questions = image_record.get("questions") or [image_record]
        for question in questions:
            sample = dict(question)
            metadata = dict(image_record.get("meta_data") or {})
            metadata.update(sample.get("meta_data") or {})
            metadata["language"] = language
            sample["meta_data"] = metadata
            sample["local_path"] = image_record.get("local_path", sample.get("local_path", ""))
            sample["question"] = resolve_sample_question(sample, language)
            samples.append(sample)
    return samples


def resolve_image_path(sample):
    local_path = Path(str(sample.get("local_path") or sample.get("image") or ""))
    if local_path.is_absolute():
        return local_path
    parts = local_path.parts
    if parts and parts[0].lower() == "images":
        local_path = Path(*parts[1:])
    return (IMAGE_DIR / local_path).resolve()


def load_data(split="E_metaphors", mode="none"):
    for sample in read_split(split):
        image_path = resolve_image_path(sample)
        if not image_path.exists():
            continue
        payload = {
            "prompt": resolve_sample_question(sample, sample["meta_data"]["language"]),
            "images": [str(image_path)],
        }
        yield payload, sample
