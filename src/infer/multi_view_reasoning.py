SUPPORTED_MODES = ("none", "cot", "domain", "emotion")


def resolve_sample_question(sample, lang=None):
    if isinstance(sample, dict) and sample.get("question"):
        return str(sample["question"])
    questions = sample.get("questions") if isinstance(sample, dict) else None
    if questions and questions[0].get("question"):
        return str(questions[0]["question"])
    if lang is None and isinstance(sample, dict):
        lang = (sample.get("meta_data") or {}).get("language")
    return "这张图片表达了什么隐喻？" if lang == "zh" else "What metaphor does this image convey?"


def build_mode_context(sample, mode, lang):
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported inference mode: {mode}")
    if mode == "none":
        return ""
    if mode == "cot":
        if lang == "zh":
            return "请逐步比较视觉证据、图中文字和每个选项，然后给出最终答案。"
        return "Compare the visual evidence, image text, and every option before answering."

    metadata = sample.get("meta_data", {}) if isinstance(sample, dict) else {}
    value = metadata.get(mode, "")
    if lang == "zh":
        label = "语义领域" if mode == "domain" else "情感基调"
        return f"已知该样本的{label}是“{value}”，请将其作为辅助线索。"
    label = "semantic domain" if mode == "domain" else "emotional tone"
    return f'The provided {label} is "{value}". Use it as an auxiliary clue.'


def _format_options(options):
    return "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(options))


def build_multi_view_prompts(sample, lang="en", mode="none"):
    question = resolve_sample_question(sample, lang)
    context = build_mode_context(sample, mode, lang)
    if context:
        question = f"{question}\n\n{context}"
    options = sample.get("options")
    if options is None and sample.get("questions"):
        options = sample["questions"][0].get("options", [])
    options_text = _format_options(options or [])

    if lang == "zh":
        return [
            {
                "type": "direct_anchor",
                "weight": 1.5,
                "text": (
                    f"<image>\n用户：{question}\n选项：\n{options_text}\n\n"
                    "仔细比较选项，选择最能解释图像为何如此设计的答案。只回答选项字母。"
                ),
            },
            {
                "type": "emotional_challenge",
                "weight": 1.3,
                "text": (
                    f"<image>\n用户：{question}\n选项：\n{options_text}\n\n"
                    "判断图像的情感、讽刺或批判基调，选择与该基调最一致的答案。"
                ),
            },
            {
                "type": "text_interaction",
                "weight": 1.2,
                "text": (
                    f"<image>\n用户：{question}\n选项：\n{options_text}\n\n"
                    "阅读图中文字并观察物体，判断图文互动如何形成隐喻，然后选择答案。"
                ),
            },
            {
                "type": "direct_fallback",
                "weight": 1.0,
                "text": f"<image>\n用户：{question}\n选项：\n{options_text}\n\n直接选择正确选项。",
            },
        ]

    return [
        {
            "type": "direct_anchor",
            "weight": 2.0,
            "text": (
                f"<image>\nUser: {question}\nOptions:\n{options_text}\n\n"
                "Based on the image, choose the correct option directly."
            ),
        },
        {
            "type": "literal_challenge",
            "weight": 1.2,
            "text": (
                f"<image>\nUser: {question}\nOptions:\n{options_text}\n\n"
                "Reject any literal surface trap and choose the option expressing the deeper meaning."
            ),
        },
        {
            "type": "text_interaction",
            "weight": 1.0,
            "text": (
                f"<image>\nUser: {question}\nOptions:\n{options_text}\n\n"
                "Read any image text and explain how it interacts with the visual metaphor, then choose."
            ),
        },
    ]
