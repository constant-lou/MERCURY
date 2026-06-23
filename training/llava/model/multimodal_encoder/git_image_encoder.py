import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel, AutoConfig
import logging

logger = logging.getLogger(__name__)

class GITImageTower(nn.Module):
    def __init__(self, vision_tower_path: str, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_path
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        
        # 先用 AutoConfig 获取配置，避免加载整个大模型
        self.config = AutoConfig.from_pretrained(self.vision_tower_name)
        
        if not delay_load:
            self.load_model()

    def load_model(self):
        """加载 GIT 模型并只提取视觉部分"""
        logger.info(f"Loading GIT vision tower from {self.vision_tower_name}")
        
        # 使用 AutoModel 加载，因为它更通用，并且我们只需要 vision 部分
        full_model = AutoModel.from_pretrained(self.vision_tower_name, output_hidden_states=True)
        # 提取视觉编码器
        self.vision_tower = full_model.vision_model
        
        # 加载配套的图像处理器
        self.image_processor = AutoProcessor.from_pretrained(self.vision_tower_name)
        
        # 冻结参数
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        logger.info("GIT vision tower loaded successfully.")

    @torch.no_grad()
    def forward(self, images):
        """
        接收一个批次的图像张量 [B, C, H, W]，返回特征。
        这个 forward 方法应该非常简单。
        """
        if not self.is_loaded:
            self.load_model()
            
        # 将图像张量移动到与模型相同的设备
        pixel_values = images.to(device=self.device, dtype=self.dtype)
        
        # 直接通过 vision tower 获取特征
        image_outputs = self.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 根据 select_layer 选择特征层
        # [B, N, D] -> N 是 patch 数量, D 是 hidden_size
        image_features = image_outputs.hidden_states[self.select_layer]
        
        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config.vision_config.hidden_size

    @property
    def num_patches(self):
        """
        [核心修改] 从模型配置中动态、准确地计算 patch 数量
        """
        # 从 vision_tower 的配置中获取图像和 patch 大小
        img_size = self.config.vision_config.image_size
        patch_size = self.config.vision_config.patch_size
        
        # 计算网格大小并返回 patch 总数
        patches_per_side = img_size // patch_size
        return patches_per_side * patches_per_side