# ComfyUI-FramePack-HY/nodes/sampler.py

import torch
import math
import numpy as np
from tqdm import tqdm
import random
import traceback # Import traceback for better error logging

# 导入 ComfyUI 相关
import comfy.model_management as model_management
import comfy.utils
import comfy.model_base
import comfy.latent_formats
import comfy.model_patcher

# 导入自定义采样函数 (使用相对路径)
from ..diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
# 导入辅助函数
from ..diffusers_helper.utils import repeat_to_batch_size, crop_or_pad_yield_mask
from ..diffusers_helper.memory import move_model_to_device_with_memory_preservation, get_cuda_free_memory_gb

# VAE缩放因子 (Hunyuan Video使用0.476986)
vae_scaling_factor = 0.476986

class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.load_device = model_management.get_torch_device()

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

class FramePackDiffusersSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fp_pipeline": ("FP_DIFFUSERS_PIPELINE",), # 接收来自加载节点的 Pipeline
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}), # real guidance scale
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1}), # distilled guidance scale
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, 
                                                     "tooltip": "GPU内存保留量(GB)，越大越稳定但速度越慢"}),
                "sampler": (["unipc"],
                            {"default": "unipc", 
                             "tooltip": "采样器类型，目前仅支持unipc"}),
                "start_latent_out": ("LATENT", {"tooltip": "来自Keyframe节点的起始潜变量"}),
                "target_latent_out": ("LATENT", {"tooltip": "(可选) 来自Keyframe节点的目标潜变量"}),
                "target_index_out": ("INT", {"tooltip": "(可选) 目标潜变量生效的分段索引"}),
            },
            "optional": {
                # 从CreateKeyframes节点直接连接的视频参数
                "video_length_seconds": ("video_length_seconds", {"default": None, 
                                                  "tooltip": "从CreateKeyframes节点获取的视频时长(秒)，优先级最高"}),
                "video_fps": ("video_fps", {"default": None,  
                                     "tooltip": "从CreateKeyframes节点获取的帧率(fps)，优先级最高"}),
                "window_size": ("window_size", {"default": None, 
                                       "tooltip": "从CreateKeyframes节点获取的窗口大小，优先级最高"}),
                
                "clip_vision": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP Vision的输出，用于图像引导"}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, 
                                   "tooltip": "影响运动幅度，数值越高运动越强"}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "使用teacache加速采样"}),
                "teacache_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, 
                                             "tooltip": "teacache相对L1损失阈值"}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, 
                                               "tooltip": "I2V模式的去噪强度(当前未使用，未来可能用于start_latent)"}),
                # 关键帧相关参数
                "keyframe_guidance_strength": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1, 
                                                         "tooltip": "关键帧引导强度。控制关键帧对视频的影响程度。值越高，视频在关键帧位置越接近目标图像，过渡效果越明显"})
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "FramePack"

    def sample(self, fp_pipeline, positive, negative, steps, cfg, guidance_scale, seed,
               width, height, gpu_memory_preservation, sampler, 
               total_second_length=5, fps=24, latent_window_size=9,
               video_length_seconds=None, video_fps=None, window_size=None,
               clip_vision=None, shift=0.0, use_teacache=True, 
               teacache_thresh=0.15, denoise_strength=1.0, 
               start_latent_out=None, target_latent_out=None, target_index_out=-1,
               keyframe_guidance_strength=1.5):

        # 优先使用从CreateKeyframes节点连接的视频参数
        if video_length_seconds is not None:
            total_second_length = video_length_seconds
            print(f"[FramePack Sampler] 使用从CreateKeyframes节点获取的视频时长: {total_second_length}秒")
            
        if video_fps is not None:
            fps = video_fps
            print(f"[FramePack Sampler] 使用从CreateKeyframes节点获取的帧率: {fps}fps")
            
        if window_size is not None:
            latent_window_size = window_size
            print(f"[FramePack Sampler] 使用从CreateKeyframes节点获取的窗口大小: {latent_window_size}")
        
        print(f"[FramePack Sampler] 最终视频参数: 总时长={total_second_length}秒, 帧率={fps}fps, 窗口大小={latent_window_size}")
        
        # 确保尺寸足够大
        if height < 256 or width < 256:
            raise ValueError(f"输入尺寸太小: {width}x{height}，请确保宽度和高度至少为256像素")
        
        # 确保我们有一个加载好的transformer
        if "transformer" not in fp_pipeline or "dtype" not in fp_pipeline:
            raise ValueError("无效的Pipeline对象。请使用Load FramePack Pipeline节点加载有效的模型。")
        
        transformer = fp_pipeline["transformer"]
        dtype = fp_pipeline["dtype"]
        
        # 设备和数据类型准备
        device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        print(f"[FramePack Sampler] 使用设备: {device}, 精度: {dtype}")
        
        # 计算潜变量尺寸
        latent_height = height // 8
        latent_width = width // 8
        
        # 计算视频帧数和分段
        num_frames_per_window = latent_window_size * 4 - 3
        total_latent_sections = (total_second_length * fps) / num_frames_per_window
        total_latent_sections = math.ceil(total_latent_sections)
        total_latent_sections = max(1, int(total_latent_sections))
        print(f"[FramePack Sampler] 总分段数: {total_latent_sections}, 每段帧数: {num_frames_per_window}")
        
        # 内存管理
        model_management.unload_all_models()
        model_management.cleanup_models()
        model_management.soft_empty_cache()
        
        # 处理条件输入
        print("[FramePack Sampler] 处理条件输入...")
        
        # 处理正向条件
        llama_vec = positive[0][0].to(dtype=dtype, device=device)
        clip_l_pooler = positive[0][1]["pooled_output"].to(dtype=dtype, device=device)
        
        # 处理负向条件
        if not math.isclose(cfg, 1.0):  # 如果需要真正的CFG
            llama_vec_n = negative[0][0].to(dtype=dtype, device=device)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(dtype=dtype, device=device)
        else:
            # 如果CFG为1.0，创建全零条件
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)
        
        # 裁剪或填充LLAMA嵌入和创建注意力掩码
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        
        # 准备CLIP视觉特征
        image_embeddings = None
        if clip_vision is not None:
            image_embeddings = clip_vision["last_hidden_state"].to(dtype=dtype, device=device)
            print(f"[FramePack Sampler] CLIP视觉特征形状: {image_embeddings.shape}")
        
        # 移除了 start_latent 的处理
        batch_size = 1
        initial_latent = None # 当前逻辑不使用I2V的 initial_latent
        
        # 初始化历史潜变量 - 修改：初始化为空，将在循环中构建
        history_latents = torch.zeros(
            (batch_size, 16, 0, latent_height, latent_width), # 时间维度从0开始
            dtype=torch.float32,
            device="cpu" # 存储在CPU以节省显存
        )
        total_generated_latent_frames = 0 # 追踪已生成的帧数
        
        # 准备随机生成器
        generator = torch.Generator("cpu").manual_seed(seed)
        
        # 创建ComfyUI模型封装
        comfy_model = HyVideoModel(
            HyVideoModelConfig(dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
        
        # 创建模型patcher
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        
        # 创建进度条回调函数 - 修改为更准确地反映多分段进度
        # 计算总步数为 steps * total_latent_sections
        total_steps = steps * total_latent_sections
        progress_bar = comfy.utils.ProgressBar(total_steps)
        
        # 记录当前分段索引和已完成的分段数
        current_section_index = 0
        
        # 定义一个适配k_diffusion库调用格式的回调函数
        def callback_adapter(d):
            # k_diffusion的callback传入参数是一个字典: {'x': x, 'i': i, 'denoised': model_prev_list[-1]}
            if 'i' in d:
                step = d['i']
                # 只更新一步，因为总步数已经是考虑了所有分段的
                progress_bar.update(1)
                
                # 打印更详细的进度信息
                if step % 5 == 0 or step == steps - 1:  # 每5步或最后一步打印一次
                    section_progress = f"{current_section_index + 1}/{total_latent_sections}"
                    overall_progress = f"{(current_section_index * steps + step + 1)}/{total_steps}"
                    print(f"[FramePack Sampler] 进度: 分段 {section_progress}, 步骤 {step + 1}/{steps}, 总进度 {overall_progress}")
            return None
            
        # ---------- 改进的模型加载与内存管理 ----------
        print(f"[FramePack Sampler] 开始加载模型到GPU设备，内存保留量设置为 {gpu_memory_preservation} GB")
        
        # 检查可用内存并估算模型大小
        try:
            current_free_memory = get_cuda_free_memory_gb(device)
            print(f"[FramePack Sampler] 当前GPU可用内存: {current_free_memory:.2f} GB")
            
            # 如果内存保留值大于当前可用内存的80%，发出警告并调整
            if gpu_memory_preservation > current_free_memory * 0.8:
                adjusted_preservation = current_free_memory * 0.5  # 调整为可用内存的50%
                print(f"[FramePack Sampler] 警告: 内存保留值({gpu_memory_preservation}GB)过大, 自动调整为 {adjusted_preservation:.2f}GB")
                gpu_memory_preservation = adjusted_preservation
            
            # 在加载前先检查是否有足够内存
            if current_free_memory <= gpu_memory_preservation + 1.0:  # 需要至少保留值+1GB
                print(f"[FramePack Sampler] 警告: GPU内存不足! 可用: {current_free_memory:.2f}GB, 需要: >{gpu_memory_preservation+1.0}GB")
                print(f"[FramePack Sampler] 尝试降低采样分辨率或降低保留内存值")
        except Exception as e:
            print(f"[FramePack Sampler] 内存检查过程出错: {e}")
            print("[FramePack Sampler] 继续执行，但可能不稳定")
        
        # 改进的模型加载方法
        try:
            # 分阶段加载模型，每阶段检查内存
            print(f"[FramePack Sampler] 阶段1: 将模型移动至 {device}...")
            move_model_to_device_with_memory_preservation(
                transformer, 
                target_device=device, 
                preserved_memory_gb=gpu_memory_preservation
            )
            
            # 检查加载后的内存状态
            try:
                post_load_memory = get_cuda_free_memory_gb(device)
                print(f"[FramePack Sampler] 模型加载后GPU可用内存: {post_load_memory:.2f} GB")
                
                if post_load_memory < gpu_memory_preservation:
                    print(f"[FramePack Sampler] 注意: 加载后可用内存({post_load_memory:.2f}GB)低于保留目标({gpu_memory_preservation}GB)")
                    print(f"[FramePack Sampler] 将尝试继续运行，但可能会出现内存不足错误")
            except Exception as e:
                print(f"[FramePack Sampler] 加载后内存检查出错: {e}")
        
        except Exception as e:
            print(f"[FramePack Sampler] 模型加载失败: {e}")
            print(f"[FramePack Sampler] 尝试使用备用加载方法...")
            
            # 备用加载方法 - 直接加载但不管理内存
            transformer.to(device)
            print("[FramePack Sampler] 使用备用方法加载模型完成")
        
        # 运行采样
        print("[FramePack Sampler] 开始采样...")
        print(f"  - 尺寸: {width}x{height}, 总帧数: {total_second_length * fps}")
        print(f"  - 分段数: {total_latent_sections}, 每段窗口大小: {latent_window_size}")
        print(f"  - 步数: {steps}, CFG: {cfg}, Guidance Scale: {guidance_scale}")
        print(f"  - 种子: {seed}, 移位: {shift}")
        
        try:
            # --- 修改: 处理新的 Start-Target 输入 --- 
            visual_start_latent = None
            visual_target_latent = None
            target_start_index = target_index_out # 直接使用传入的索引
            # === 添加调试打印 ===
            print(f"[FramePack Sampler DEBUG] Received target_index_out: {target_index_out}, Initial target_start_index: {target_start_index}")
            # ==================
            
            if start_latent_out is not None and "samples" in start_latent_out:
                # 起始潜变量是必需的，应用VAE缩放因子
                vs_latent = start_latent_out["samples"]
                if vs_latent is not None and vs_latent.shape[2] > 0: # 检查时间维度是否有效
                     visual_start_latent = vs_latent * vae_scaling_factor
                     visual_start_latent = visual_start_latent.to(dtype=dtype, device="cpu") # 准备在CPU上
                     print(f"[关键帧逻辑] 已准备起始潜变量 (来自start_latent_out)，形状: {visual_start_latent.shape}")
                else:
                    raise ValueError("输入的起始潜变量无效或为空！")
            else:
                 raise ValueError("未提供有效的起始潜变量 (start_latent_out)！")

            # 处理可选的目标潜变量
            if target_latent_out is not None and "samples" in target_latent_out and target_start_index >= 0:
                vt_latent = target_latent_out["samples"]
                if vt_latent is not None and vt_latent.shape[2] > 0: # 检查时间维度是否有效
                     visual_target_latent = vt_latent * vae_scaling_factor
                     visual_target_latent = visual_target_latent.to(dtype=dtype, device="cpu") # 准备在CPU上
                     print(f"[关键帧逻辑] 已准备目标潜变量 (来自target_latent_out)，目标索引: {target_start_index}，形状: {visual_target_latent.shape}")
                     # 确保目标索引在范围内
                     if target_start_index >= total_latent_sections:
                         print(f"[关键帧逻辑] 警告: 目标索引 {target_start_index} 超出总分段数 {total_latent_sections}，将调整为最后一个分段 {total_latent_sections - 1}")
                         target_start_index = total_latent_sections - 1
                     # 确保目标索引不为0
                     if target_start_index == 0:
                         print(f"[关键帧逻辑] 警告: 目标索引不能为0，已禁用目标引导。")
                         target_start_index = -1 # 禁用目标
                         visual_target_latent = None
                else:
                    print(f"[关键帧逻辑] 提供的目标潜变量为空或无效，已禁用目标引导。")
                    target_start_index = -1 # 禁用目标
                    visual_target_latent = None
            else:
                print(f"[关键帧逻辑] 未提供有效的目标潜变量或目标索引，禁用目标引导。")
                target_start_index = -1 # 禁用目标
                visual_target_latent = None
            # ------------------------------------------

            # 重置旧的变量，以防意外使用
            visual_end_latent = None 
            idx_visual_end = -1
            idx_visual_start = 0 # 起点固定为0
            
            # 在循环开始前准备好关键帧潜变量 - 修改：逻辑已移到上面处理输入的部分
            # if keyframes is not None and len(keyframe_idx_list) >= 1:
            #     ...
            # else:
            #     ...

            # --- 修改: 从前向后逐段生成 ---
            for current_section_index in range(total_latent_sections):
                print(f"[FramePack Sampler] 生成分段 {current_section_index + 1}/{total_latent_sections}")
                is_first_section = current_section_index == 0
                is_last_section = current_section_index == total_latent_sections - 1

                # --- 修改: 参照参考代码定义索引分割 ---
                indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                # 分割方式: [起始目标帧(1), 4x历史(16), 2x历史(2), 1x历史(1), 当前窗口(window_size)]
                clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split(
                    [1, 16, 2, 1, latent_window_size], dim=1
                )
                # 用于 sample_hunyuan 的 clean_latents 参数的索引
                clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

                # --- 修改: 关键帧处理逻辑 (计算 target_latent 和 weight) ---
                # 使用 forward_section_no 表达更清晰，即 current_section_index
                forward_section_no = current_section_index

                # 初始化默认目标 (零潜变量) 和权重
                # 需要创建一个形状正确的零张量作为默认目标
                # 修改：使用 visual_start_latent (它现在是必需的) 的形状
                # if visual_start_latent is not None: 
                #     default_target_shape = visual_start_latent.shape
                # else:
                #     # 创建一个基于配置的形状 (B=1, C=16, T=1, H, W)
                #     default_target_shape = (batch_size, 16, 1, latent_height, latent_width)
                default_target_shape = visual_start_latent.shape # 直接使用start_latent的形状

                # 确保在正确的设备和类型上创建
                default_target_latent = torch.zeros(default_target_shape, dtype=dtype, device=device)

                target_latent = default_target_latent # 默认为零
                calculated_weight = 1.0 # 默认权重

                # --- 关键帧引导逻辑 (与之前类似，基于 forward_section_no) ---
                # --- 修改：实现新的 Start -> Target 引导逻辑 ---
                print(f"[关键帧逻辑] 处理分段 {forward_section_no}/{total_latent_sections - 1} (Start-Target模式)")

                # --- 修改：持续引导，中途切换 --- 
                if visual_target_latent is not None and target_start_index > 0 and forward_section_no >= target_start_index:
                    # 达到或超过目标索引，强引导至目标
                    target_latent = visual_target_latent.to(device=device, dtype=dtype)
                    calculated_weight = keyframe_guidance_strength
                    print(f"  -> 分段达到/超过目标索引({target_start_index})，强引导至目标，权重: {calculated_weight:.2f}")
                else:
                    # 在目标索引之前 (包括索引0) 或无目标时，强引导至起点
                    target_latent = visual_start_latent.to(device=device, dtype=dtype)
                    calculated_weight = keyframe_guidance_strength
                    print(f"  -> 引导至起点，权重: {calculated_weight:.2f}")
                
                # --- 修改: 准备 clean_latents (参照参考代码) --- 
                # 需要从 history_latents 末尾提取历史信息
                # 处理 history_latents 不足的情况 (特别是第一帧)

                # 需要的历史长度: 1 (1x) + 2 (2x) + 16 (4x) = 19
                required_history_len = 1 + 2 + 16
                current_history_len = history_latents.shape[2]

                # 创建零值占位符，用于填充不足的历史
                zero_latent_1x = torch.zeros((batch_size, 16, 1, latent_height, latent_width), dtype=dtype, device=device)
                zero_latent_2x = torch.zeros((batch_size, 16, 2, latent_height, latent_width), dtype=dtype, device=device)
                zero_latent_4x = torch.zeros((batch_size, 16, 16, latent_height, latent_width), dtype=dtype, device=device)

                if current_history_len == 0:
                    # 第一帧，完全没有历史
                    clean_latents_1x = zero_latent_1x
                    clean_latents_2x = zero_latent_2x
                    clean_latents_4x = zero_latent_4x
                    print(f"[FramePack Sampler] 第一个分段，使用零历史")
                else:
                    # 从历史末尾提取，不足部分用零填充
                    print(f"[FramePack Sampler] 从历史潜变量(长度 {current_history_len})末尾提取上下文")
                    # 提取可用历史
                    available_history = history_latents[:, :, -min(current_history_len, required_history_len):, :, :].to(device=device, dtype=dtype)
                    available_len = available_history.shape[2]

                    # 分配给 1x, 2x, 4x (从后往前)
                    len_1x = min(available_len, 1)
                    start_idx_1x = available_len - len_1x
                    actual_1x = available_history[:, :, start_idx_1x : start_idx_1x + len_1x, :, :]
                    clean_latents_1x = torch.cat([zero_latent_1x[:, :, :(1 - len_1x), :, :], actual_1x], dim=2) if len_1x < 1 else actual_1x

                    available_len -= len_1x
                    len_2x = min(available_len, 2)
                    start_idx_2x = available_len - len_2x
                    actual_2x = available_history[:, :, start_idx_2x : start_idx_2x + len_2x, :, :]
                    clean_latents_2x = torch.cat([zero_latent_2x[:, :, :(2 - len_2x), :, :], actual_2x], dim=2) if len_2x < 2 else actual_2x

                    available_len -= len_2x
                    len_4x = min(available_len, 16)
                    start_idx_4x = available_len - len_4x
                    actual_4x = available_history[:, :, start_idx_4x : start_idx_4x + len_4x, :, :]
                    clean_latents_4x = torch.cat([zero_latent_4x[:, :, :(16 - len_4x), :, :], actual_4x], dim=2) if len_4x < 16 else actual_4x

                    print(f"  - 提取长度: 1x={clean_latents_1x.shape[2]}, 2x={clean_latents_2x.shape[2]}, 4x={clean_latents_4x.shape[2]}")


                # 组合最终的 clean_latents
                # --- 修改：实现三阶段引导逻辑 ---
                is_guiding_target = visual_target_latent is not None and target_start_index > 0
                
                if is_guiding_target and forward_section_no == target_start_index:
                    # 第二阶段：过渡到目标 (仅在此分段使用混合)
                    alpha = 0.5 # 混合比例
                    weighted_target = target_latent * calculated_weight
                    # 形状检查
                    if clean_latents_1x.shape != weighted_target.shape:
                         print(f"[FramePack Sampler] 警告：混合引导时形状不匹配！ History: {clean_latents_1x.shape}, Target: {weighted_target.shape}. 跳过混合，使用 [target, target]")
                         mixed_latent = weighted_target # 回退到之前的强引导
                    else:
                         mixed_latent = (1 - alpha) * clean_latents_1x + alpha * weighted_target
                         print(f"[FramePack Sampler] 过渡至目标 (混合 alpha={alpha})")
                    
                    clean_latents = torch.cat([weighted_target, mixed_latent], dim=2)
                    print(f"[FramePack Sampler] 组合 clean_latents: [target, mixed(history, target)]")

                elif is_guiding_target and forward_section_no > target_start_index:
                     # 第三阶段：从目标状态开始演变 (使用偏向历史的混合上下文)
                     alpha_evo = 0.1 # 演变阶段混合比例，更偏向历史
                     weighted_target = target_latent * calculated_weight # 确保 target_latent 是目标潜变量
                     # 形状检查
                     if clean_latents_1x.shape != weighted_target.shape:
                         print(f"[FramePack Sampler] 警告：演变阶段混合时形状不匹配！ History: {clean_latents_1x.shape}, Target: {weighted_target.shape}. 使用 [target, history_1x]")
                         mixed_latent_evo = clean_latents_1x # 回退到只使用历史
                     else:
                         mixed_latent_evo = (1 - alpha_evo) * clean_latents_1x + alpha_evo * weighted_target
                         print(f"[FramePack Sampler] 从目标演变 (混合上下文 alpha={alpha_evo})")
                    
                     clean_latents = torch.cat([weighted_target, mixed_latent_evo], dim=2)
                     print(f"[FramePack Sampler] 组合 clean_latents: [target, mixed_evo(history, target)]")

                else: # 对应 forward_section_no < target_start_index 或无目标的情况
                    # 第一阶段：引导至起点 (使用真实历史)
                    # 注意：此时 target_latent 实际上是 visual_start_latent
                    weighted_start = target_latent * calculated_weight
                    clean_latents = torch.cat([weighted_start, clean_latents_1x], dim=2)
                    print(f"[FramePack Sampler] 引导至起点，组合 clean_latents: [start, history_1x]")
                # -----------------------------------
                
                # 结构: [目标关键帧(加权), 1x历史]
                # clean_latents = torch.cat([target_latent * calculated_weight, clean_latents_1x], dim=2)
                print(f"[FramePack Sampler] 准备好的 clean_latents 形状: {clean_latents.shape}")
                print(f"  - clean_latents_1x 形状: {clean_latents_1x.shape}")
                print(f"  - clean_latents_2x 形状: {clean_latents_2x.shape}")
                print(f"  - clean_latents_4x 形状: {clean_latents_4x.shape}")

                # 设置teacache
                if hasattr(transformer, 'initialize_teacache'):
                    if use_teacache:
                        transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_thresh)
                    else:
                        transformer.initialize_teacache(enable_teacache=False)
                
                # 检查当前内存状态
                try:
                    current_mem = get_cuda_free_memory_gb(device)
                    print(f"[FramePack Sampler] 采样前GPU可用内存: {current_mem:.2f} GB")
                    if current_mem < 1.0:  # 如果内存严重不足
                        print("[FramePack Sampler] 警告: GPU内存严重不足，尝试释放部分内存...")
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[FramePack Sampler] 内存检查出错: {e}")
                
                # 执行采样
                with torch.autocast(device_type=model_management.get_autocast_device(device), dtype=dtype, enabled=True):
                    # 更新当前分段索引 (虽然循环变量就是它，但为了回调函数清晰)
                    # print(f"[FramePack Sampler] 开始处理分段 {current_section_index + 1}/{total_latent_sections}") # 日志位置调整

                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler=sampler,  # 使用用户选择的采样器
                        initial_latent=initial_latent,  # 设为 None，由 clean_latents 控制
                        concat_latent=None, # -> 参考代码未使用，保持None
                        strength=denoise_strength,  # 应用去噪强度 (I2V时可能有用)
                        width=width,
                        height=height,
                        frames=num_frames_per_window,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=guidance_scale,
                        guidance_rescale=0.0,
                        shift=shift if shift > 0 else None,
                        num_inference_steps=steps,
                        batch_size=batch_size,
                        generator=generator,
                        prompt_embeds=llama_vec,
                        prompt_embeds_mask=llama_attention_mask,
                        prompt_poolers=clip_l_pooler,
                        negative_prompt_embeds=llama_vec_n,
                        negative_prompt_embeds_mask=llama_attention_mask_n,
                        negative_prompt_poolers=clip_l_pooler_n,
                        dtype=dtype,
                        device=device,
                        negative_kwargs=None, # -> 参考代码未使用，保持None
                        callback=callback_adapter,  # 使用我们的适配器回调函数
                        # 添加额外参数 - 使用新的索引和clean_latents
                        image_embeddings=image_embeddings, # CLIP Vision 特征
                        latent_indices=latent_indices, # 当前窗口索引 [T]
                        clean_latents=clean_latents, # [目标帧(加权), 1x历史] [B,C,1+1,H,W]
                        clean_latent_indices=clean_latent_indices, # 对应 clean_latents 的索引 [1+1]
                        clean_latents_2x=clean_latents_2x, # [2x历史] [B,C,2,H,W]
                        clean_latent_2x_indices=clean_latent_2x_indices, # [2]
                        clean_latents_4x=clean_latents_4x, # [4x历史] [B,C,16,H,W]
                        clean_latent_4x_indices=clean_latent_4x_indices, # [16]
                    )
                
                # 检查采样后的内存状态
                try:
                    post_sample_mem = get_cuda_free_memory_gb(device)
                    print(f"[FramePack Sampler] 分段采样后GPU可用内存: {post_sample_mem:.2f} GB")
                    
                    # 如果内存低于保留值的50%，清理缓存
                    if post_sample_mem < gpu_memory_preservation * 0.5:
                        print("[FramePack Sampler] 内存不足，清理缓存...")
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[FramePack Sampler] 采样后内存检查出错: {e}")
                
                # --- 修改: 更新历史潜变量和总帧数 ---
                # 将生成的潜变量移到CPU并拼接到 history_latents
                current_generated_frames = generated_latents.shape[2]
                history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
                total_generated_latent_frames += current_generated_frames
                print(f"[FramePack Sampler] 分段 {current_section_index + 1} 生成了 {current_generated_frames} 帧，总帧数: {total_generated_latent_frames}")
                print(f"[FramePack Sampler] 更新后 history_latents 形状: {history_latents.shape}")

                # 移除旧的反向逻辑和最后分段的特殊处理
                # if is_last_section:
                #     # 移除静态帧添加逻辑，防止视频开头出现静态画面
                #     print("[FramePack Sampler] 处理最后分段 (时间上的第一段)")
                #     # 不再额外添加静态帧，保持动态效果
                #     print(f"[FramePack Sampler] 保持动态开始，形状: {generated_latents.shape}")

                # 更新总帧数和历史潜变量 (已移到上面)
                # total_generated_latent_frames += int(generated_latents.shape[2])
                # history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                # 获取实际生成的潜变量 (已移到循环外)
                # real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                # 如果是最后一段，停止生成 (现在由 for 循环控制)
                # if is_last_section:
                #     break

            # --- 修改: 循环结束后处理最终结果 ---
            print("[FramePack Sampler] 所有分段采样完成.")

            # 检查最终生成的帧数是否合理
            expected_total_frames = total_second_length * fps
            # 注意: Hunyuan Video 的 latent_window_size 和 num_frames_per_window 计算可能导致实际帧数略有差异
            print(f"[FramePack Sampler] 预期总帧数: {expected_total_frames:.1f}, 实际生成潜变量帧数: {total_generated_latent_frames}")
            print(f"[FramePack Sampler] 最终输出 history_latents 形状: {history_latents.shape}")

            # 记录实际调用的采样信息
            print(f"[FramePack Sampler] 📊 采样信息: 窗口大小={latent_window_size}, 帧数/窗口={num_frames_per_window}")

            # 确保进度条显示为完成状态
            # 计算已完成的总步数
            completed_steps = total_latent_sections * steps
            progress_remaining = total_steps - completed_steps # total_steps = steps * total_latent_sections
            if progress_remaining != 0: # 理论上应该为0，除非计算有误或提前中断
                 print(f"[FramePack Sampler] 警告: 进度计算可能存在偏差，剩余 {progress_remaining} 步")
                 # 强制更新到100%
                 progress_bar.update(progress_remaining)

            print("[FramePack Sampler] ✅ 进度: 100% 完成!")

            # 返回结果，应用VAE缩放因子
            # history_latents 已经在CPU上
            final_latents = history_latents.to(model_management.intermediate_device()) / vae_scaling_factor
            print(f"[FramePack Sampler] 返回最终潜变量，形状: {final_latents.shape}")
            return ({"samples": final_latents},)

        except Exception as e:
            # 详细记录错误信息
            error_message = f"[FramePack Sampler] ❌ 错误: {str(e)}"
            print(error_message)
            print("[FramePack Sampler] 📋 错误详情:")
            traceback.print_exc()
            
            # 更新进度条到错误状态
            try:
                # 计算剩余步数并更新进度条
                if 'progress_bar' in locals() and 'current_section_index' in locals() and 'total_steps' in locals():
                    progress_so_far = min(current_section_index * steps, total_steps)
                    progress_remaining = total_steps - progress_so_far
                    if progress_remaining > 0:
                        print(f"[FramePack Sampler] 由于错误更新进度条 (剩余 {progress_remaining} 步)")
                        progress_bar.update(progress_remaining)
                print("[FramePack Sampler] ⚠️ 进度: 由于错误而中断!")
            except Exception as progress_error:
                print(f"[FramePack Sampler] 更新进度条失败: {progress_error}")
            
            # 提供通用的建议解决方案
            print("[FramePack Sampler] 🔧 建议解决方案:")
            print("1. 检查GPU内存是否足够，可能需要降低分辨率或减少生成的帧数")
            print("2. 确保模型正确加载")
            print("3. 检查条件输入是否有效")
            print("4. 如果问题持续，可以尝试重启ComfyUI或清理缓存")
            
            # 创建一个空的有效潜变量作为返回值
            try:
                print("[FramePack Sampler] 创建空潜变量作为错误恢复...")
                # 创建一个小的空潜变量
                empty_latent = torch.zeros(
                    (1, 16, 1, height // 8, width // 8), # 保持通道数16
                    dtype=torch.float32,
                    device=model_management.intermediate_device()
                )
                return ({"samples": empty_latent},)
            except Exception as fallback_error:
                print(f"[FramePack Sampler] 创建空潜变量失败: {fallback_error}")
                # 最小的可能潜变量
                minimal_latent = torch.zeros(
                    (1, 16, 1, 8, 8), # 保持通道数16
                    dtype=torch.float32,
                    device=model_management.intermediate_device()
                )
                return ({"samples": minimal_latent},)
        finally:
            # 主动释放内存
            print("[FramePack Sampler] 主动清理GPU内存...")
            torch.cuda.empty_cache()
            
            # 释放transformer
            try:
                print(f"[FramePack Sampler] 将transformer卸载到 {offload_device}")
                transformer.to(offload_device)
            except Exception as e:
                print(f"[FramePack Sampler] 卸载transformer时出错: {e}")
            
            model_management.soft_empty_cache()


NODE_CLASS_MAPPINGS = {
    "FramePackDiffusersSampler_HY": FramePackDiffusersSampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FramePackDiffusersSampler_HY": "FramePack Sampler (HY)"
} 

# 导入辅助节点并添加到映射中
try:
    from .keyframe_helper import NODE_CLASS_MAPPINGS as KEYFRAME_NODE_CLASS_MAPPINGS
    from .keyframe_helper import NODE_DISPLAY_NAME_MAPPINGS as KEYFRAME_NODE_DISPLAY_NAME_MAPPINGS
    
    # 更新映射
    NODE_CLASS_MAPPINGS.update(KEYFRAME_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(KEYFRAME_NODE_DISPLAY_NAME_MAPPINGS)
except ImportError as e:
    print(f"[FramePack] 警告: 无法导入关键帧辅助节点: {e}") 