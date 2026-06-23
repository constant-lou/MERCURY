import torch
import torch.nn as nn

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

def get_frames_in_interval(start, end, number_of_frames):
        indices = np.random.randint(low = start, high = end, size = number_of_frames)
        indices = np.sort(indices)
        return indices


def sample_frames_on_interval(file_path, num_frames):
    
    #####修改点
    # 检查文件扩展名确定是图像还是视频
    import os
    ext = os.path.splitext(file_path)[1].lower()
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    # 如果是图像文件，使用已有的make_video_from_image函数处理
    if ext in image_exts:
        return make_video_from_image(file_path, num_frames)
    
    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        # Making three intervals: Start, middle and the end
        interval_gap = seg_len // 3
        frames_per_gap = clip_len // 3
        
        begin_interval = get_frames_in_interval(0, interval_gap, frames_per_gap)
        middle_interval = get_frames_in_interval(interval_gap + 1, interval_gap * 2, frames_per_gap)
        end_interval = get_frames_in_interval((interval_gap*2) + 1, seg_len, frames_per_gap)
        frame_indices = np.concatenate((begin_interval, middle_interval, end_interval), axis = 0)
        # print(frame_indices)
        return frame_indices
    
    # video clip consists of 300 frames (10 seconds at 30 FPS)
    try:
        videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
    except:
        print("******"*10)
        print("Failed for : ", file_path)
    # sample 6 frames
    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=4, seg_len=len(videoreader))
    frames = videoreader.get_batch(indices).asnumpy()

    return list(frames)

    

def sample_all_frames_on_multiple_interval(file_path, num_frames, interval_count=4):
    

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        # Making 4 interval of videos
        no_of_intervals = interval_count
        interval_gap = seg_len // no_of_intervals
        frames_per_gap = 6

        frames = []

        for i in range(0, no_of_intervals):
            curr_interval = get_frames_in_interval(interval_gap * i, interval_gap * (i+1), frames_per_gap)
            frames.append(curr_interval)
        return frames
    


    # video clip consists of 300 frames (10 seconds at 30 FPS)
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=4, seg_len=len(videoreader))
    frame_set = []
    for index_set in indices:
        video_frames = videoreader.get_batch(index_set).asnumpy()
        frame_set.append(list(video_frames))

    return frame_set


def make_video_from_image(file_path, num_frames):
    image = Image.open(file_path).convert('RGB')
    image.load()
    data = np.asarray(image)
    # print(data.shape)
    frames = [data] * num_frames
    return frames




class GITVideoTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        print("\n\n\n>>>>>> 正在执行 video_encoder.py 中最新版本的 GITVideoTower！<<<<<<\n\n\n")

        print("")

        self.is_loaded = False

        # vision_tower_name = "microsoft/git-large-vatex"
        self.vision_tower_name = vision_tower
        # self.select_layer = args.mm_vision_select_layer
        self.select_layer = -2
        try:
            self.interval_type = args.interval_type
            self.no_of_intervals = args.no_of_intervals
        except:
            self.interval_type = "multi_interval"
            self.no_of_intervals = 4
        


        self.load_model()
        
    def load_model(self):
        # [核心修改] 我们不加载 AutoModelForCausalLM，而是加载 AutoModel
        from transformers import AutoModel
        
        self.processor = AutoProcessor.from_pretrained(self.vision_tower_name, local_files_only=True)
        self.image_processor = self.processor # 它们是同一个
        
        # 加载基础 GIT 模型
        print(f"DEBUG: Loading base AutoModel from {self.vision_tower_name}")
        full_model = AutoModel.from_pretrained(self.vision_tower_name, output_hidden_states=True,  local_files_only=True)
        
        # [核心修改] 只提取纯粹的视觉编码器部分
        self.vision_tower = full_model.image_encoder 
        
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        print("DEBUG: GIT vision_model extracted and loaded.")


    @torch.no_grad()
    def forward(self, images):
        """
        [核心修改] 大幅简化 forward 方法。
        纯粹的 vision_model 只需要 pixel_values。
        """
        if not self.is_loaded:
            self.load_model()
        
        # 检查 'images' 是 list 还是 tensor
        if isinstance(images, list):
            # 如果是 list, 我们在这里堆叠它
            try:
                pixel_values = torch.stack(images, dim=0).to(device=self.device, dtype=self.dtype)
            except RuntimeError as e:
                print(f"CRITICAL ERROR: Failed to stack image tensors in video_encoder.py. {e}")
                for i, img in enumerate(images):
                    print(f"  Image {i} shape: {img.shape}")
                raise e
        else:
            # 如果它已经是 tensor，我们就像以前一样处理
            pixel_values = images.to(device=self.device, dtype=self.dtype)
        
        # [核心修改] 纯粹的 vision_model 不需要 dummy_input_ids
        outputs = self.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )
        
        # GITVisionModel 的 hidden_states 包含 [CLS] token
        # 第-2层特征, 形状为 [B, 901, 768]
        image_features = outputs.hidden_states[self.select_layer]
        
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype
        # return torch.bfloat16
        # return torch.float32

    @property
    def return_dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        # print("***"*20)
        # print("Config: ", self.vision_tower.config)
        # print("***"*20)

        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only



    @property
    def hidden_size(self):
        # [核心修改] config 对象就是 vision_config，直接访问
        # GitVisionConfig 的 hidden_size 就是 1024
        print(f"DEBUG: Accessing hidden_size directly from config: {self.config.hidden_size}")
        return self.config.hidden_size

    @property
    def num_patches(self):
        try:
            # [核心修改] config 对象就是 vision_config，直接访问
            img_size = self.config.image_size
            patch_size = self.config.patch_size
            patches_per_side = img_size // patch_size
            # 视觉编码器会额外输出一个 [CLS] token
            num_patches_val = (patches_per_side * patches_per_side) + 1 
            
            print(f"DEBUG: Successfully calculated num_patches (with CLS): {num_patches_val}")
            return num_patches_val

        except AttributeError as e:
            print(f"ERROR inside num_patches property: {e}")
            print("       Could not find .image_size or .patch_size in config.")
            print("       Falling back to hardcoded 901 patches for GIT model (900 + 1 CLS).")
            return 901 # 900 + 1
