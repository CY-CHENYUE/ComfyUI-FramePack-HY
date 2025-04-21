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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_window_size": ("INT", {"default": 9, "min": 4, "max": 33, "step": 1, 
                                              "tooltip": "窗口大小参数，控制每个分段处理的帧数。较大的值可能生成更连贯的视频，但需要更多GPU内存"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 8}),
                "total_second_length": ("FLOAT", {"default": 5, "min": 1, "max": 60, "step": 0.1, 
                                                 "tooltip": "视频总时长(秒)"}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0, "step": 0.1, 
                                                     "tooltip": "GPU内存保留量(GB)，越大越稳定但速度越慢"}),
                "sampler": (["unipc"],
                            {"default": "unipc", 
                             "tooltip": "采样器类型，目前仅支持unipc"}),
            },
            "optional": {
                "start_latent": ("LATENT", {"tooltip": "I2V模式的输入潜变量，可从VAE Encode获取"}),
                "clip_vision": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP Vision的输出，用于图像引导"}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, 
                                   "tooltip": "影响运动幅度，数值越高运动越强"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60, 
                               "tooltip": "视频帧率(每秒帧数)"}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "使用teacache加速采样"}),
                "teacache_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, 
                                             "tooltip": "teacache相对L1损失阈值"}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, 
                                               "tooltip": "I2V模式的去噪强度，越低保留越多原始图像特征"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "FramePack"

    def sample(self, fp_pipeline, positive, negative, steps, cfg, guidance_scale, seed,
               latent_window_size, width, height, total_second_length, gpu_memory_preservation,
               sampler, start_latent=None, clip_vision=None, shift=0.0, fps=8, use_teacache=True, 
               teacache_thresh=0.15, denoise_strength=1.0):

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
        num_frames_per_window = latent_window_size * 4 - 3  # 每个窗口可生成的有效帧数
        total_latent_sections = (total_second_length * fps) / num_frames_per_window
        total_latent_sections = int(max(round(total_latent_sections), 1))
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
        
        # 准备初始潜变量
        batch_size = 1
        initial_latent = None
        
        # 如果提供了起始潜变量(I2V模式)
        if start_latent is not None:
            print("[FramePack Sampler] 准备I2V模式的起始潜变量...")
            # 获取潜变量并应用VAE缩放因子
            initial_latent = start_latent["samples"] * vae_scaling_factor
            
            # 详细打印潜变量的形状信息用于调试
            print(f"[FramePack Sampler] 原始潜变量形状: {initial_latent.shape}, 类型: {initial_latent.dtype}")
            
            # 检查潜变量维度是否正确
            if initial_latent.ndim == 4:  # 单帧潜变量 [B, C, H, W]
                # 添加时间维度 [B, C, 1, H, W]
                initial_latent = initial_latent.unsqueeze(2)
                print(f"[FramePack Sampler] 添加时间维度后形状: {initial_latent.shape}")
            
            # 确认形状是5维 [B, C, T, H, W]
            if initial_latent.ndim != 5:
                raise ValueError(f"输入潜变量形状错误: {initial_latent.shape}，应为 [B, C, T, H, W] 或 [B, C, H, W]")
            
            # 安全地获取调整目标尺寸
            current_height = initial_latent.shape[3]
            current_width = initial_latent.shape[4]
            
            # 调整潜变量尺寸
            if current_height != latent_height or current_width != latent_width:
                print(f"[FramePack Sampler] 调整潜变量尺寸从 {current_height}x{current_width} 到 {latent_height}x{latent_width}")
                
                # 使用更安全的方法重塑和调整尺寸
                batch, channels, frames = initial_latent.shape[0], initial_latent.shape[1], initial_latent.shape[2]
                
                # 先展平所有帧进行处理
                flattened = initial_latent.reshape(batch * channels * frames, 1, current_height, current_width)
                print(f"[FramePack Sampler] 展平后形状: {flattened.shape}")
                
                # 应用插值
                resized = torch.nn.functional.interpolate(
                    flattened, 
                    size=(latent_height, latent_width),
                    mode='bilinear',
                    align_corners=False
                )
                print(f"[FramePack Sampler] 调整尺寸后形状: {resized.shape}")
                
                # 重塑回原始维度结构
                initial_latent = resized.reshape(batch, channels, frames, latent_height, latent_width)
                print(f"[FramePack Sampler] 重塑后最终形状: {initial_latent.shape}")
            
            # 将潜变量移动到计算设备
            initial_latent = initial_latent.to(device=device, dtype=dtype)
            print(f"[FramePack Sampler] 最终起始潜变量形状: {initial_latent.shape}")
        
        # 初始化起始潜变量(如果是T2V模式或没有提供起始潜变量)
        if initial_latent is None:
            start_latent = torch.zeros(
                (batch_size, 16, 1, latent_height, latent_width),
                dtype=torch.float32,
                device="cpu"
            )
        else:
            # 如果有起始潜变量，用它初始化start_latent
            start_latent = initial_latent.detach().cpu().to(torch.float32)
            if start_latent.shape[2] > 1:
                # 只取第一帧
                start_latent = start_latent[:, :, :1, :, :]
        
        # 初始化历史潜变量
        history_latents = torch.zeros(
            (batch_size, 16, 1 + 2 + 16, latent_height, latent_width),
            dtype=torch.float32, 
            device="cpu"
        )
        
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
        if initial_latent is not None:
            print(f"  - I2V模式: 去噪强度 = {denoise_strength}")
        
        try:
            # 处理分段生成
            total_generated_latent_frames = 0
            latent_paddings_list = list(reversed(range(total_latent_sections)))
            latent_paddings = latent_paddings_list.copy()
            
            # 对于长视频，优化分段策略
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
                latent_paddings_list = latent_paddings.copy()
            
            # 逐段生成
            for latent_padding in latent_paddings:
                print(f"[FramePack Sampler] 生成分段 {latent_padding + 1}/{total_latent_sections}")
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size
                
                print(f'  - 填充大小 = {latent_padding_size}, 是否最后分段 = {is_last_section}')
                
                # 创建和分割索引
                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split(
                    [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1
                )
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                
                # 准备清理潜变量
                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                
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
                    # 更新当前分段索引
                    current_section_index = total_latent_sections - latent_padding - 1
                    print(f"[FramePack Sampler] 开始处理分段 {current_section_index + 1}/{total_latent_sections}")
                    
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler=sampler,  # 使用用户选择的采样器
                        initial_latent=initial_latent,  # 使用初始潜变量(I2V模式)
                        concat_latent=None,
                        strength=denoise_strength,  # 应用去噪强度
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
                        negative_kwargs=None,
                        callback=callback_adapter,  # 使用我们的适配器回调函数
                        # 添加额外参数
                        image_embeddings=image_embeddings,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
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
                
                # 如果是最后一段，连接起始潜变量
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                
                # 更新总帧数和历史潜变量
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
                
                # 获取实际生成的潜变量
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
                
                # 如果是最后一段，停止生成
                if is_last_section:
                    break
            
            print("[FramePack Sampler] 采样完成.")
            print(f"[FramePack Sampler] 输出形状: {real_history_latents.shape}")
            
            # 记录实际调用的采样信息
            print(f"[FramePack Sampler] 📊 采样信息: 窗口大小={latent_window_size}, 帧数/窗口={num_frames_per_window}")
            
            # 确保进度条显示为完成状态
            progress_remaining = total_steps - (current_section_index + 1) * steps
            if progress_remaining > 0:
                print(f"[FramePack Sampler] 更新进度条到完成状态 (剩余 {progress_remaining} 步)")
                progress_bar.update(progress_remaining)
            print("[FramePack Sampler] ✅ 进度: 100% 完成!")
            
            # 返回结果，应用VAE缩放因子
            return ({"samples": real_history_latents.to(model_management.intermediate_device()) / vae_scaling_factor},)
            
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
                    (1, 16, 1, height // 8, width // 8), 
                    dtype=torch.float32,
                    device=model_management.intermediate_device()
                )
                return ({"samples": empty_latent},)
            except Exception as fallback_error:
                print(f"[FramePack Sampler] 创建空潜变量失败: {fallback_error}")
                # 最小的可能潜变量
                minimal_latent = torch.zeros(
                    (1, 4, 1, 8, 8), 
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