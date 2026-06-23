def _runtime():
    import torch
    from PIL import Image
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    return {
        "torch": torch,
        "Image": Image,
        "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
        "DEFAULT_IM_END_TOKEN": DEFAULT_IM_END_TOKEN,
        "DEFAULT_IM_START_TOKEN": DEFAULT_IM_START_TOKEN,
        "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
        "SeparatorStyle": SeparatorStyle,
        "conv_templates": conv_templates,
        "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
        "tokenizer_image_token": tokenizer_image_token,
        "load_pretrained_model": load_pretrained_model,
        "disable_torch_init": disable_torch_init,
    }


def load_model(model_name, model_args, use_accel=False):
    if use_accel:
        raise NotImplementedError("Accelerated loading is not supported by this adapter.")
    runtime = _runtime()
    runtime["disable_torch_init"]()
    tokenizer, model, image_processor, _ = runtime["load_pretrained_model"](
        model_args["model_path_or_name"],
        model_args.get("model_base"),
        model_name,
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "conv_mode": model_args.get("conv_mode", "llava_v1"),
    }


def _build_conversation(prompt_item, model, conv_mode, runtime):
    if not isinstance(prompt_item, dict) or "prompt" not in prompt_item:
        raise ValueError("Each inference item must contain a prompt.")
    images = prompt_item.get("images") or []
    if not images:
        raise ValueError("Each inference item must contain one image.")
    user_prompt = str(prompt_item["prompt"])
    if model.config.mm_use_im_start_end:
        image_token = (
            runtime["DEFAULT_IM_START_TOKEN"]
            + runtime["DEFAULT_IMAGE_TOKEN"]
            + runtime["DEFAULT_IM_END_TOKEN"]
        )
    else:
        image_token = runtime["DEFAULT_IMAGE_TOKEN"]
    conversation = runtime["conv_templates"][conv_mode].copy()
    conversation.append_message(conversation.roles[0], image_token + "\n" + user_prompt)
    conversation.append_message(conversation.roles[1], None)
    return conversation, images[0]


def infer(prompts, **kwargs):
    runtime = _runtime()
    torch = runtime["torch"]
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    image_processor = kwargs.get("image_processor")
    conv_mode = kwargs.get("conv_mode")
    if not all((model, tokenizer, image_processor, conv_mode)):
        raise ValueError("Missing model, tokenizer, image_processor, or conv_mode.")

    temperature = float(kwargs.get("temperature", 0.0))
    max_new_tokens = int(kwargs.get("max_new_tokens", 256))
    responses = []
    for prompt_item in prompts:
        conversation, image_file = _build_conversation(prompt_item, model, conv_mode, runtime)
        image = runtime["Image"].open(image_file).convert("RGB")
        try:
            image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        except Exception:
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).to(device=model.device, dtype=torch.float16)

        prompt = conversation.get_prompt()
        input_ids = runtime["tokenizer_image_token"](
            prompt,
            tokenizer,
            runtime["IMAGE_TOKEN_INDEX"],
            return_tensors="pt",
        ).unsqueeze(0).to(model.device)
        stop_text = (
            conversation.sep
            if conversation.sep_style != runtime["SeparatorStyle"].TWO
            else conversation.sep2
        )
        stopping = runtime["KeywordsStoppingCriteria"]([stop_text], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping],
            )
        input_length = input_ids.shape[1]
        output = tokenizer.batch_decode(
            output_ids[:, input_length:],
            skip_special_tokens=True,
        )[0].strip()
        if stop_text and output.endswith(stop_text):
            output = output[: -len(stop_text)].strip()
        responses.append(output)
    return responses
