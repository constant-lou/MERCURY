# import torch
# import torch.nn as nn
# import re


# class IdentityMap(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, *args, **kwargs):
#         return x

#     @property
#     def config(self):
#         return {"mm_projector_type": 'identity'}


# class SimpleResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.pre_norm = nn.LayerNorm(channels)

#         self.proj = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels)
#         )
#     def forward(self, x):
#         x = self.pre_norm(x)
#         return x + self.proj(x)


# def build_vision_projector(config, delay_load=False, **kwargs):
#     projector_type = getattr(config, 'mm_projector_type', 'linear')

#     if projector_type == 'linear':
#         return nn.Linear(config.mm_hidden_size, config.hidden_size)

#     mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
#     if mlp_gelu_match:
#         mlp_depth = int(mlp_gelu_match.group(1))
#         modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
#         for _ in range(1, mlp_depth):
#             modules.append(nn.GELU())
#             modules.append(nn.Linear(config.hidden_size, config.hidden_size))
#         return nn.Sequential(*modules)

#     if projector_type == 'identity':
#         return IdentityMap()

#     raise ValueError(f'Unknown projector type: {projector_type}')


###############################
# filepath: llava/model/multimodal_projector/builder.py

import torch
import torch.nn as nn
import re

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}

def build_vision_projector(config, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    # [核心修改] 从 config 中正确获取两个模型的 hidden_size
    # config.hidden_size 是 LLM (Vicuna) 的大小, e.g., 5120
    # config.mm_hidden_size 是 Vision Tower (GIT) 的大小, e.g., 1024
    llm_hidden_size = config.hidden_size
    vision_hidden_size = config.mm_hidden_size

    # 添加调试打印，这至关重要！
    print("=======================================================================")
    print(f"DEBUG: Building MM Projector ({projector_type})")
    print(f"       Vision Tower (Input) Size:  {vision_hidden_size}")
    print(f"       LLM (Output) Size:        {llm_hidden_size}")
    print("=======================================================================")

    if projector_type == 'linear':
        # 应该构建 nn.Linear(1024, 5120)
        return nn.Linear(vision_hidden_size, llm_hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        # 第一层: 1024 -> 5120
        modules = [nn.Linear(vision_hidden_size, llm_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            # 后续层: 5120 -> 5120
            modules.append(nn.Linear(llm_hidden_size, llm_hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
