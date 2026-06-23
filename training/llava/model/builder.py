#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# import os
# import warnings
# import shutil

# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
# import torch
# from llava.model import *
# from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
#     kwargs = {"device_map": device_map}

#     if load_8bit:
#         kwargs['load_in_8bit'] = True
#     elif load_4bit:
#         kwargs['load_in_4bit'] = True
#         kwargs['quantization_config'] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'
#         )
#     else:
#         kwargs['torch_dtype'] = torch.float16

#     if 'llava' in model_name.lower():
#         # Load LLaVA model
#         if 'lora' in model_name.lower() and model_base is None:
#             warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/anonymous/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
#         if 'lora' in model_name.lower() and model_base is not None:
#             lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             print('Loading LLaVA from base model...')
#             model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
#             token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
#             if model.lm_head.weight.shape[0] != token_num:
#                 model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
#                 model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

#             print('Loading additional LLaVA weights...')
#             if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
#                 non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
#             else:
#                 # this is probably from HF Hub
#                 from huggingface_hub import hf_hub_download
#                 def load_from_hf(repo_id, filename, subfolder=None):
#                     cache_file = hf_hub_download(
#                         repo_id=repo_id,
#                         filename=filename,
#                         subfolder=subfolder)
#                     return torch.load(cache_file, map_location='cpu')
#                 non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
#             non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
#             if any(k.startswith('model.model.') for k in non_lora_trainables):
#                 non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
#             model.load_state_dict(non_lora_trainables, strict=False)

#             from peft import PeftModel
#             print('Loading LoRA weights...')
#             model = PeftModel.from_pretrained(model, model_path)
#             print('Merging LoRA weights...')
#             model = model.merge_and_unload()
#             print('Model is loaded...')
#         elif model_base is not None:
#             # this may be mm projector only
#             print('Loading LLaVA from base model...')
#             if 'mpt' in model_name.lower():
#                 if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
#                     shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#                 model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path)
#                 model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

#             mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
#             mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
#             model.load_state_dict(mm_projector_weights, strict=False)
#         else:
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#             else:
#                 print("inside else of yes llava of else")
#                 print("\n\n\n")
#                 print("model path: ", model_path)
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#     else:
#         # Load language model
#         if model_base is not None:
#             # PEFT model
#             from peft import PeftModel
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
#             print(f"Loading LoRA weights from {model_path}")
#             model = PeftModel.from_pretrained(model, model_path)
#             print(f"Merging weights")
#             model = model.merge_and_unload()
#             print('Convert to FP16...')
#             model.to(torch.float16)
#         else:
#             use_fast = False
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
#             else:
#                 print("inside else of not llava of else")
#                 print("model path: ", model_path)
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

#     image_processor = None

#     if 'llava' in model_name.lower():
#         mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
#         mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
#         if mm_use_im_patch_token:
#             tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#         if mm_use_im_start_end:
#             tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
#         model.resize_token_embeddings(len(tokenizer))

#         vision_tower = model.get_vision_tower()
#         print("*"*12)
#         print(vision_tower)
#         print(vision_tower.is_loaded)
#         if not vision_tower.is_loaded:
#             vision_tower.load_model()
#         vision_tower.to(device=device, dtype=torch.float16)
#         image_processor = vision_tower.image_processor

#     if hasattr(model.config, "max_sequence_length"):
#         context_len = model.config.max_sequence_length
#     else:
#         context_len = 2048

#     return tokenizer, model, image_processor, context_len


####################修改点
# filepath: llava/model/multimodal_encoder/builder.py

import torch
import torch.nn as nn
import os
# [核心修改] 我们只导入我们确定要用的这一个类
from llava.model.multimodal_encoder.git_image_encoder import GITImageTower

def build_vision_tower(model_args, delay_load=False):
    vision_tower_path = model_args.vision_tower

    # ------------------- 强制执行区 -------------------
    # 我们暂时注释掉所有的判断逻辑，只留下一条路
    # 这样可以100%确定加载的是我们修改过的 GITImageTower
    
    print("=======================================================================")
    print(f"DEBUG: Bypassing all checks. FORCE-LOADING GITImageTower.")
    print(f"       Vision Tower Path: {vision_tower_path}")
    print("=======================================================================")

    return GITImageTower(vision_tower_path, model_args, delay_load)
    
    # ------------------- 以下是原始逻辑（已注释） -------------------
    # print(f"DEBUG: Building vision tower for path: {vision_tower}")
    #
    # if 'clip' in vision_tower_path.lower():
    #     print("DEBUG: Matched 'clip', building CLIPVisionTower.")
    #     from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    #     return CLIPVisionTower(vision_tower_path, model_args, delay_load)
    # 
    # elif 'git' in vision_tower_path.lower():
    #     print("DEBUG: Matched 'git', building GITImageTower from git_image_encoder.py.")
    #     from llava.model.multimodal_encoder.git_image_encoder import GITImageTower
    #     return GITImageTower(vision_tower_path, model_args, delay_load)
    # 
    # else:
    #     print(f"WARNING: No specific match for '{vision_tower_path}'. Falling back to default VideoTower.")
    #     from llava.model.multimodal_encoder.video_encoder import GITVideoTower
    #     return GITVideoTower(vision_tower_path, model_args, delay_load)
    # ==============================================================================
# [补丁] 为了兼容 CII-Bench 推理脚本，手动添加 load_pretrained_model 函数
# ==============================================================================
from transformers import AutoTokenizer, AutoConfig

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
    else:
        kwargs['torch_dtype'] = torch.float16

    # 尝试导入模型类，适配不同的文件结构
    try:
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
    except ImportError:
        try:
            from llava.model.llava_llama import LlavaLlamaForCausalLM
        except ImportError:
            # 最后的尝试
            from llava.model import LlavaLlamaForCausalLM

    print(f"[Builder] Loading Tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print(f"[Builder] Loading Model from {model_path}...")
    # 加载模型
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # 初始化 Vision Tower (这会调用上面原本就有的 build_vision_tower)
    print(f"[Builder] Initializing Vision Tower...")
    model.get_model().get_vision_tower().load_model()
    
    # 移动 Vision Tower 到正确的设备和精度
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=device, dtype=torch.float16)
    
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    print(f"[Builder] Load Success. Context Len: {context_len}")
    return tokenizer, model, image_processor, context_len