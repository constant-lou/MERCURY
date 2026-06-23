import argparse
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

from config_wrapper import ConfigWrapper
from data_loader import load_data
from multi_view_reasoning import build_mode_context, resolve_sample_question


ROLE_MAP = {
    "direct_anchor": "anchor",
    "direct_fallback": "anchor",
    "text_interaction": "textual",
    "literal_challenge": "challenge",
    "emotional_challenge": "challenge",
}

DOMAIN_ANCHORS = {
    "environment": (
        "ecological",
        "environmental",
        "environment",
        "industrial",
        "destruction",
        "pollution",
        "生态",
        "环境",
        "工业",
        "污染",
        "破坏",
    ),
    "rights_irony": (
        "rights",
        "literal",
        "absurd",
        "extreme",
        "权利",
        "字面",
        "荒谬",
        "极端",
    ),
    "fairness": (
        "privilege",
        "unfair",
        "fairness",
        "ethics",
        "opportunities",
        "advantages",
        "特权",
        "不公平",
        "公平",
        "伦理",
        "机会",
    ),
    "scale_skill_contrast": (
        "smaller",
        "limited",
        "expertise",
        "skills",
        "skill",
        "尺度",
        "有限",
        "能力",
        "技能",
        "专业",
    ),
}


def parse_final_answer(response_text):
    if not response_text:
        return None
    text = str(response_text).strip()
    patterns = (
        r"^\s*\(?([A-H])\)?\s*[\.\):：]?",
        r"Final\s*Answer\s*[:：]\s*\(?([A-H])\)?",
        r"Answer\s*[:：]\s*\(?([A-H])\)?",
        r"The\s+(?:final\s+)?(?:answer|option)\s+is\s*\(?([A-H])\)?",
        r"答案\s*[:：]\s*\(?([A-H])\)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    standalone = re.findall(r"\b([A-H])\b", text.upper())
    return standalone[-1] if standalone else None


def normalize_answer_letter(value):
    if not value:
        return None
    text = str(value).strip()
    if re.fullmatch(r"[A-H]", text, flags=re.IGNORECASE):
        return text.upper()
    return parse_final_answer(text)


def get_question_data(sample):
    if isinstance(sample, dict) and sample.get("questions"):
        return sample["questions"][0]
    return sample


def get_option_text(sample, prediction):
    prediction = normalize_answer_letter(prediction)
    if not prediction:
        return ""
    options = get_question_data(sample).get("options", []) or []
    index = ord(prediction) - ord("A")
    return str(options[index]) if 0 <= index < len(options) else ""


def get_baseline_prediction(record):
    if not record:
        return None
    for key in ("final_prediction", "extracted_answer", "prediction"):
        prediction = normalize_answer_letter(record.get(key))
        if prediction:
            return prediction
    return parse_final_answer(record.get("response"))


def collect_cross_view_evidence(record):
    votes = {}
    roles = {}
    sources = {}
    source_text = {}
    for log in (record or {}).get("view_logs", []) or []:
        prediction = normalize_answer_letter(log.get("pred")) or parse_final_answer(log.get("text"))
        if not prediction:
            continue
        try:
            weight = float(log.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        source_type = log.get("type")
        votes[prediction] = votes.get(prediction, 0.0) + weight
        sources.setdefault(prediction, []).append(source_type)
        role = ROLE_MAP.get(source_type)
        if role:
            roles.setdefault(prediction, set()).add(role)
        text = str(log.get("text") or "")
        if text:
            source_text[prediction] = (source_text.get(prediction, "") + "\n" + text).strip()
    return votes, roles, sources, source_text


def has_weak_support(text):
    weak_en = (
        r"\b(not\s+fully|does\s+not\s+fully|misses|unclear|ambiguous|"
        r"overly|too\s+broad|less\s+specific)\b"
    )
    weak_zh = (
        "该选项不完全",
        "该选项不够",
        "该选项模糊",
        "该选项过于",
        "该选项缺乏",
        "候选不完全",
        "候选不够",
        "候选模糊",
        "候选过于",
        "候选缺乏",
    )
    text = str(text or "")
    return bool(re.search(weak_en, text, flags=re.IGNORECASE)) or any(term in text for term in weak_zh)


def normalize_reference(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower().replace(",", "")


def extract_reference_anchors(text):
    text = str(text or "")
    anchors = set()
    for quoted in re.findall(r'["“”‘’]([^"“”‘’]{2,80})["“”‘’]', text):
        anchors.add(normalize_reference(quoted))
    for acronym in re.findall(r"\b[A-Z][A-Z0-9&.-]{1,12}\b", text):
        anchors.add(normalize_reference(acronym))
    for number in re.findall(r"(?:[$¥€£]\s*)?\b\d[\d,.]*\b", text):
        anchors.add(normalize_reference(number))
    return sorted(anchor for anchor in anchors if anchor)


def missing_reference_anchors(baseline_option, candidate_option):
    anchors = extract_reference_anchors(baseline_option)
    candidate = normalize_reference(candidate_option)
    return anchors, [anchor for anchor in anchors if anchor not in candidate]


def expansion_ratio(baseline_option, candidate_option):
    return len(str(candidate_option or "").strip()) / max(1, len(str(baseline_option or "").strip()))


def count_domain_hits(text, terms):
    text = str(text or "")
    lowered = text.lower()
    hits = 0
    for term in terms:
        if any("\u4e00" <= char <= "\u9fff" for char in term):
            hits += int(term in text)
        else:
            hits += int(bool(re.search(r"\b" + re.escape(term.lower()) + r"\b", lowered)))
    return hits


def missing_domain_anchor(baseline_option, candidate_option, min_hits=2):
    for domain, terms in DOMAIN_ANCHORS.items():
        baseline_hits = count_domain_hits(baseline_option, terms)
        if baseline_hits < min_hits:
            continue
        candidate_hits = count_domain_hits(candidate_option, terms)
        if candidate_hits == 0:
            return {
                "domain": domain,
                "baseline_domain_hits": baseline_hits,
                "candidate_domain_hits": candidate_hits,
            }
    return None


def calibrate_cross_view_prediction(
    sample,
    baseline_record,
    anchor_ratio=0.9,
    max_anchor_expansion_ratio=1.5,
):
    baseline_prediction = get_baseline_prediction(baseline_record)
    votes, roles, sources, source_text = collect_cross_view_evidence(baseline_record)
    baseline_weight = votes.get(baseline_prediction, 0.0)
    baseline_roles = roles.get(baseline_prediction, set())
    info = {
        "version": "cross_view_evidence_coherence",
        "baseline_prediction": baseline_prediction,
        "baseline_weight": round(baseline_weight, 4),
        "baseline_roles": sorted(baseline_roles),
        "vote_summary": {key: round(value, 4) for key, value in sorted(votes.items())},
        "role_summary": {key: sorted(value) for key, value in sorted(roles.items())},
        "changed": False,
        "rule": "keep",
    }
    if not baseline_prediction or not votes:
        info["rule"] = "keep_no_scores"
        return baseline_prediction, info

    candidates = []
    blocked = []
    for prediction, weight in votes.items():
        if prediction == baseline_prediction or has_weak_support(source_text.get(prediction, "")):
            continue
        candidate_roles = roles.get(prediction, set())
        anchor_coherence = (
            "anchor" in candidate_roles
            and weight >= baseline_weight * anchor_ratio
            and bool({"textual", "challenge"} & baseline_roles)
        )
        complementary_consensus = (
            {"textual", "challenge"} <= candidate_roles and weight >= baseline_weight
        )
        if not (anchor_coherence or complementary_consensus):
            continue
        gate = "complementary_consensus" if complementary_consensus else "anchor_coherence"
        if gate == "anchor_coherence":
            baseline_option = get_option_text(sample, baseline_prediction)
            candidate_option = get_option_text(sample, prediction)
            anchors, missing = missing_reference_anchors(baseline_option, candidate_option)
            if missing:
                blocked.append(
                    {
                        "candidate_prediction": prediction,
                        "reference_anchors": anchors,
                        "missing_reference_anchors": missing,
                    }
                )
                continue
            domain_block = missing_domain_anchor(baseline_option, candidate_option)
            if domain_block:
                blocked.append({"candidate_prediction": prediction, **domain_block})
                continue
            ratio = expansion_ratio(baseline_option, candidate_option)
            if ratio > max_anchor_expansion_ratio:
                blocked.append(
                    {
                        "candidate_prediction": prediction,
                        "candidate_expansion_ratio": round(ratio, 4),
                    }
                )
                continue
        priority = 2 if complementary_consensus else 1
        candidates.append((priority, weight, len(candidate_roles), prediction, gate))

    if not candidates:
        if blocked:
            info["blocked_candidates"] = blocked
            info["rule"] = "keep_semantic_preservation"
        return baseline_prediction, info

    priority, candidate_weight, _, candidate_prediction, gate = max(candidates)
    info.update(
        {
            "candidate_prediction": candidate_prediction,
            "candidate_weight": round(candidate_weight, 4),
            "candidate_roles": sorted(roles.get(candidate_prediction, set())),
            "candidate_sources": sources.get(candidate_prediction, []),
            "candidate_gate": gate,
            "candidate_priority": priority,
            "changed": candidate_prediction != baseline_prediction,
            "rule": "cross_view_evidence_coherence_switch",
        }
    )
    return candidate_prediction, info


def build_cyclic_permutations(option_count):
    if option_count < 2:
        return []
    return [
        [((index + shift) % option_count) for index in range(option_count)]
        for shift in range(option_count)
    ]


def build_dihedral_permutations(option_count):
    if option_count < 3:
        return [
            {"orbit": "forward", "permutation": permutation}
            for permutation in build_cyclic_permutations(option_count)
        ]
    probes = []
    for shift in range(option_count):
        probes.append(
            {
                "orbit": "forward",
                "permutation": [((index + shift) % option_count) for index in range(option_count)],
            }
        )
    for shift in range(option_count):
        probes.append(
            {
                "orbit": "reverse",
                "permutation": [((shift - index) % option_count) for index in range(option_count)],
            }
        )
    return probes


def map_prediction_to_original(prediction, permutation):
    prediction = normalize_answer_letter(prediction)
    if not prediction:
        return None
    displayed_index = ord(prediction) - ord("A")
    if not 0 <= displayed_index < len(permutation):
        return None
    original_index = permutation[displayed_index]
    return chr(ord("A") + original_index) if 0 <= original_index < 26 else None


def _format_options(options):
    return "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(options))


def build_permutation_prompt(sample, permutation):
    question_data = get_question_data(sample)
    language = (sample.get("meta_data") or {}).get("language", "en")
    question = resolve_sample_question(sample, language)
    mode = sample.get("inference_mode", "none")
    context = build_mode_context(sample, mode, language)
    if context:
        question = f"{question}\n\n{context}"
    options = question_data.get("options", []) or []
    permuted = [options[index] for index in permutation]
    return (
        "Answer the multiple-choice question from the image. The option order has changed; "
        "judge the content rather than the position.\n"
        "请根据图像独立作答。选项顺序已经改变，请依据内容而不是位置判断。\n\n"
        f"Question / 问题:\n{question}\n\nOptions / 选项:\n{_format_options(permuted)}\n\n"
        "Return only one option letter in parentheses, for example (A)."
    )


def normalized_vote_entropy(predictions):
    normalized = [normalize_answer_letter(value) for value in predictions]
    normalized = [value for value in normalized if value]
    if len(normalized) < 2:
        return 0.0
    counts = Counter(normalized)
    total = float(len(normalized))
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    maximum = math.log(len(normalized))
    return entropy / maximum if maximum else 0.0


def resolve_permutation_prediction(
    sample,
    baseline_record,
    mapped_predictions,
    displayed_predictions,
    probe_orbits,
    min_candidate_votes=8,
    min_direction_votes=4,
    max_anchor_expansion_ratio=1.5,
):
    baseline_prediction = get_baseline_prediction(baseline_record)
    votes, roles, sources, source_text = collect_cross_view_evidence(baseline_record)
    mapped = [normalize_answer_letter(value) for value in mapped_predictions]
    mapped = [value for value in mapped if value]
    displayed = [normalize_answer_letter(value) for value in displayed_predictions]
    displayed = [value for value in displayed if value]
    probe_votes = Counter(mapped)
    orbit_votes = {"forward": Counter(), "reverse": Counter()}
    for prediction, orbit in zip(mapped_predictions, probe_orbits):
        prediction = normalize_answer_letter(prediction)
        if prediction and orbit in orbit_votes:
            orbit_votes[orbit][prediction] += 1

    info = {
        "version": "dihedral_permutation_equivariance",
        "baseline_prediction": baseline_prediction,
        "mapped_probe_predictions": mapped,
        "displayed_probe_predictions": displayed,
        "probe_vote_summary": dict(sorted(probe_votes.items())),
        "orbit_vote_summary": {
            orbit: dict(sorted(values.items())) for orbit, values in orbit_votes.items()
        },
        "display_position_entropy": round(normalized_vote_entropy(displayed), 4),
        "changed": False,
        "rule": "keep_no_stable_consensus",
    }
    if len(mapped) < min_candidate_votes or not all(orbit_votes.values()):
        info["rule"] = "keep_incomplete_orbit"
        return baseline_prediction, info

    candidates = []
    for prediction, count in probe_votes.items():
        if prediction == baseline_prediction:
            continue
        forward = orbit_votes["forward"][prediction]
        reverse = orbit_votes["reverse"][prediction]
        if (
            forward >= min_direction_votes
            and reverse >= min_direction_votes
            and forward > orbit_votes["forward"][baseline_prediction]
            and reverse > orbit_votes["reverse"][baseline_prediction]
        ):
            candidates.append((count, prediction))
    if not candidates:
        return baseline_prediction, info
    candidate_votes, candidate_prediction = max(candidates)
    if sum(count == candidate_votes for count, _ in candidates) > 1:
        info["rule"] = "keep_tied_candidate_consensus"
        return baseline_prediction, info
    if candidate_prediction not in votes:
        info["candidate_prediction"] = candidate_prediction
        info["rule"] = "keep_candidate_without_cross_view_support"
        return baseline_prediction, info

    baseline_option = get_option_text(sample, baseline_prediction)
    candidate_option = get_option_text(sample, candidate_prediction)
    anchors, missing = missing_reference_anchors(baseline_option, candidate_option)
    if missing:
        info.update(
            {
                "candidate_prediction": candidate_prediction,
                "reference_anchors": anchors,
                "missing_reference_anchors": missing,
                "rule": "keep_reference_anchor_missing",
            }
        )
        return baseline_prediction, info
    domain_block = missing_domain_anchor(baseline_option, candidate_option)
    if domain_block:
        info.update(
            {
                "candidate_prediction": candidate_prediction,
                "rule": "keep_candidate_missing_domain_anchor",
                **domain_block,
            }
        )
        return baseline_prediction, info
    ratio = expansion_ratio(baseline_option, candidate_option)
    if ratio > max_anchor_expansion_ratio or has_weak_support(source_text.get(candidate_prediction, "")):
        info.update(
            {
                "candidate_prediction": candidate_prediction,
                "candidate_expansion_ratio": round(ratio, 4),
                "rule": "keep_weak_or_overexpanded_candidate",
            }
        )
        return baseline_prediction, info

    info.update(
        {
            "candidate_prediction": candidate_prediction,
            "candidate_probe_votes": candidate_votes,
            "candidate_sources": sources.get(candidate_prediction, []),
            "candidate_roles": sorted(roles.get(candidate_prediction, set())),
            "changed": True,
            "rule": "dihedral_permutation_equivariance_switch",
        }
    )
    return candidate_prediction, info


def _load_result_records(path, config_wrapper):
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = config_wrapper.get_id(record)
            if record_id is not None:
                records[str(record_id)] = record
    return records


def _baseline_path(directory, model_name, split, mode):
    return Path(directory) / f"{model_name}_{split}_{mode}.jsonl"


def _output_path(directory, model_name, split, mode, stage):
    return Path(directory) / f"{model_name}-{stage}_{split}_{mode}.jsonl"


def _model_inputs(payload, prompts):
    images = payload.get("images", []) if isinstance(payload, dict) else []
    return [{"prompt": prompt, "images": images} for prompt in prompts]


def run_stage(args):
    config_wrapper = ConfigWrapper(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_components = None

    for split in args.split:
        for mode in args.mode:
            baseline_path = _baseline_path(args.baseline_result_dir, args.baseline_model_name, split, mode)
            baseline_records = _load_result_records(baseline_path, config_wrapper)
            if not baseline_records:
                raise RuntimeError(f"Baseline results are missing or empty: {baseline_path}")

            output_path = _output_path(output_dir, args.model_name, split, mode, args.stage)
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            processed = 0
            with temporary_path.open("w", encoding="utf-8") as handle:
                for payload, sample in load_data(split=split, mode=mode):
                    if args.infer_limit is not None and processed >= args.infer_limit:
                        break
                    sample_id = str(config_wrapper.get_id(sample))
                    baseline = baseline_records.get(sample_id)
                    if baseline is None:
                        raise RuntimeError(f"Baseline result missing for sample: {sample_id}")
                    sample["inference_mode"] = mode
                    cvec_prediction, cvec_info = calibrate_cross_view_prediction(sample, baseline)

                    if args.stage == "cvec":
                        prediction = cvec_prediction
                        calibration_info = cvec_info
                        probe_records = []
                        decision_source = "cvec" if cvec_info.get("changed") else "baseline"
                    elif cvec_info.get("changed"):
                        prediction = cvec_prediction
                        calibration_info = {
                            "version": "dihedral_permutation_equivariance",
                            "baseline_prediction": get_baseline_prediction(baseline),
                            "changed": False,
                            "rule": "keep_cross_view_switch",
                        }
                        probe_records = []
                        decision_source = "cvec"
                    else:
                        votes, _, _, _ = collect_cross_view_evidence(baseline)
                        permutation_specs = (
                            build_dihedral_permutations(len(sample.get("options", [])))
                            if len(votes) >= 2
                            else []
                        )
                        prompts = [
                            build_permutation_prompt(sample, item["permutation"])
                            for item in permutation_specs
                        ]
                        if prompts:
                            if model_components is None:
                                from models import load_model

                                model_components = load_model(args.model_name, args.use_accel)
                            from models import infer

                            responses = infer(args.model_name)(
                                _model_inputs(payload, prompts),
                                **model_components,
                                temperature=0.0,
                                max_new_tokens=args.answer_max_new_tokens,
                            )
                        else:
                            responses = []
                        probe_records = []
                        for spec, prompt, response in zip(permutation_specs, prompts, responses):
                            displayed = parse_final_answer(response)
                            mapped = map_prediction_to_original(displayed, spec["permutation"])
                            probe_records.append(
                                {
                                    "orbit": spec["orbit"],
                                    "permutation": spec["permutation"],
                                    "prompt": prompt,
                                    "response": response,
                                    "displayed_prediction": displayed,
                                    "mapped_prediction": mapped,
                                }
                            )
                        prediction, calibration_info = resolve_permutation_prediction(
                            sample,
                            baseline,
                            [record["mapped_prediction"] for record in probe_records],
                            [record["displayed_prediction"] for record in probe_records],
                            [record["orbit"] for record in probe_records],
                        )
                        decision_source = "pesc" if calibration_info.get("changed") else "baseline"

                    result = dict(sample)
                    result["baseline_prediction"] = get_baseline_prediction(baseline)
                    result["baseline_response"] = baseline.get("response", "")
                    result["cvec_calibration_info"] = cvec_info
                    result["permutation_probes"] = probe_records
                    result["calibration_info"] = calibration_info
                    result["decision_source"] = decision_source
                    result["final_prediction"] = prediction
                    result[config_wrapper.get("response_key", "response")] = baseline.get("response", "")
                    json.dump(result, handle, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    processed += 1
            shutil.move(str(temporary_path), str(output_path))
            print(f"saved={output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run MERCURY evidence calibration.")
    parser.add_argument("--stage", choices=("cvec", "pesc"), required=True)
    parser.add_argument("--model_name", default="git-llava-13b")
    parser.add_argument("--config", default="src/config/config_mercury.yaml")
    parser.add_argument("--split", nargs="+", default=["E_metaphors", "C_metaphors"])
    parser.add_argument("--mode", nargs="+", default=["none"])
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--baseline_result_dir", required=True)
    parser.add_argument("--baseline_model_name", default="git-llava-13b")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--infer_limit", type=int)
    parser.add_argument("--answer_max_new_tokens", type=int, default=32)
    parser.add_argument("--use_accel", action="store_true")
    return parser


if __name__ == "__main__":
    run_stage(build_parser().parse_args())
