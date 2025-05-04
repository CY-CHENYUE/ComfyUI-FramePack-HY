# ComfyUI-FramePack-HY/nodes/keyframe_helper.py

import torch
import math
from ..diffusers_helper.utils import repeat_to_batch_size

class CreateKeyframes_HY:
    """
    辅助节点：定义视频的起点和可选的目标关键帧及其索引。

    输入一个必需的起始潜变量 (keyframe_1) 和一个可选的目标潜变量 (target_latent_in)。
    为目标潜变量指定一个分段索引 (target_index)，视频将从该索引开始向目标演变。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframe_1": ("LATENT", {"tooltip": "定义视频起点的潜变量 (必需)"}),
                "video_length_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.1,
                                                  "tooltip": "视频总时长(秒)，用于计算总分段数和验证索引"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1,
                               "tooltip": "视频帧率(每秒帧数)，用于计算总分段数"}),
                "window_size": ("INT", {"default": 9, "min": 4, "max": 33, "step": 1,
                                       "tooltip": "采样窗口大小参数，必须与sampler中的latent_window_size保持一致"}),
            },
            "optional": {
                "target_latent_in": ("LATENT", {"tooltip": "(可选) 定义视频演变目标的潜变量"}),
                "target_index": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1, # -1 表示不使用目标
                                            "tooltip": "(可选) 目标潜变量开始强力影响视频的分段索引 (从0开始，-1表示不使用)"}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "video_length_seconds", "video_fps", "window_size",)
    RETURN_NAMES = ("start_latent_out", "target_latent_out", "target_index_out", "video_length_seconds", "fps", "window_size",)
    FUNCTION = "create_start_target_internal"
    CATEGORY = "FramePack"
    DESCRIPTION = """定义视频的起始潜变量和可选的目标潜变量及索引。

用于指定视频的开端 (keyframe_1)，并可选地设定一个后续目标状态 (target_latent_in) 及其作用时间点 (target_index)。
Sampler节点将使用这些信息来引导视频从起点开始，并在到达目标索引后强力引导向目标状态。

使用说明：
1. 输入必需的 keyframe_1 作为视频起点。
2. （可选）输入 target_latent_in 和 target_index。target_index 指定目标潜变量开始强力影响视频的分段。
3. 设置视频参数以确保索引有效。

注意：
- target_index 必须在有效的分段范围内 [0, N-1]。
- 如果提供了 target_latent_in 但 target_index 为 -1 或无效，目标潜变量将被忽略。
"""

    def create_start_target_internal(self, keyframe_1, video_length_seconds, fps, window_size,
                                     target_latent_in=None, target_index=-1):
        print(f"[CreateKeyframes_HY (Start-Target Mode)] 处理起点和可选目标关键帧")
        print(f"[CreateKeyframes_HY (Start-Target Mode)] 视频参数: 时长={video_length_seconds}秒, 帧率={fps}fps, 窗口大小={window_size}")

        # --- 计算总分段数 --- 
        frames_per_window = window_size * 4 - 3
        total_frames = video_length_seconds * fps
        if total_frames <= 0 or frames_per_window <= 0:
             total_sections = 1
        else:
             total_sections = math.ceil(total_frames / frames_per_window)
        total_sections = max(1, int(total_sections))
        print(f"[CreateKeyframes_HY (Start-Target Mode)] 计算得到总帧数: {total_frames}, 总分段数: {total_sections}")

        # --- 处理起始潜变量 (来自 keyframe_1) --- 
        start_latent_samples = keyframe_1["samples"]
        print(f"[CreateKeyframes_HY (Start-Target Mode)] 处理起始潜变量 (来自keyframe_1)，形状: {start_latent_samples.shape}")
        # 确保5D和单帧
        if start_latent_samples.ndim == 4:
            start_latent_samples = start_latent_samples.unsqueeze(2)
        elif start_latent_samples.ndim != 5:
            raise ValueError(f"起始潜变量 (keyframe_1) 形状错误: {start_latent_samples.shape}")
        if start_latent_samples.shape[2] > 1:
            print(f"[CreateKeyframes_HY (Start-Target Mode)] 起始潜变量包含多帧，仅使用第一帧")
            start_latent_samples = start_latent_samples[:, :, :1, :, :]
        
        start_latent_out_dict = {"samples": start_latent_samples}

        # --- 处理目标潜变量 (来自 target_latent_in) --- 
        target_latent_out_dict = None
        valid_target_index = -1
        
        use_target = target_latent_in is not None and target_index >= 0
        
        if use_target:
            # 验证 target_index 范围
            if target_index >= total_sections:
                print(f"[CreateKeyframes_HY (Start-Target Mode)] 警告: 目标索引 {target_index} 超出有效范围 [0, {total_sections - 1}]，将使用最后一个索引 {total_sections - 1} 代替。")
                target_index = total_sections - 1
            # 确保目标索引不为0 (起点由start_latent定义)
            if target_index == 0:
                 print(f"[CreateKeyframes_HY (Start-Target Mode)] 警告: 目标索引不能为0 (已被起始潜变量占用)，已禁用目标潜变量。")
                 use_target = False
            
            if use_target:
                target_latent_samples = target_latent_in["samples"]
                print(f"[CreateKeyframes_HY (Start-Target Mode)] 处理目标潜变量，指定索引: {target_index}，形状: {target_latent_samples.shape}")
                
                # 确保5D和单帧
                if target_latent_samples.ndim == 4:
                    target_latent_samples = target_latent_samples.unsqueeze(2)
                elif target_latent_samples.ndim != 5:
                     print(f"[CreateKeyframes_HY (Start-Target Mode)] 警告: 目标潜变量形状错误 {target_latent_samples.shape}, 已禁用目标。")
                     use_target = False
                
                if use_target and target_latent_samples.shape[2] > 1:
                    print(f"[CreateKeyframes_HY (Start-Target Mode)] 目标潜变量包含多帧，仅使用第一帧")
                    target_latent_samples = target_latent_samples[:, :, :1, :, :]

                # 检查尺寸是否与起始潜变量匹配
                if use_target and (target_latent_samples.shape[3:] != start_latent_samples.shape[3:]):
                     print(f"[CreateKeyframes_HY (Start-Target Mode)] 警告: 目标潜变量尺寸 {target_latent_samples.shape[3:]} 与起始潜变量尺寸 {start_latent_samples.shape[3:]} 不匹配。将尝试调整目标潜变量尺寸。")
                     try:
                          target_latent_samples = torch.nn.functional.interpolate(
                              target_latent_samples.squeeze(2), 
                              size=start_latent_samples.shape[3:], 
                              mode='bilinear', 
                              align_corners=False
                          ).unsqueeze(2)
                          print(f"[CreateKeyframes_HY (Start-Target Mode)] 目标潜变量尺寸已调整为 {target_latent_samples.shape[3:]}")
                     except Exception as e:
                          print(f"[CreateKeyframes_HY (Start-Target Mode)] 调整目标潜变量尺寸失败: {e}。已禁用目标。")
                          use_target = False
                
                if use_target:
                     target_latent_out_dict = {"samples": target_latent_samples}
                     valid_target_index = target_index
                     print(f"[CreateKeyframes_HY (Start-Target Mode)] 成功准备目标潜变量，索引: {valid_target_index}")
                else:
                     print(f"[CreateKeyframes_HY (Start-Target Mode)] 因验证或处理失败，目标潜变量被禁用")
        else:
            print(f"[CreateKeyframes_HY (Start-Target Mode)] 未提供或未启用目标潜变量 (索引: {target_index})")

        # 如果目标无效，确保返回None/空字典和-1索引
        if not use_target:
            valid_target_index = -1
            target_latent_out_dict = None

        # --- 返回结果 --- 
        if target_latent_out_dict is None:
            print("[CreateKeyframes_HY (Start-Target Mode)] 未使用目标潜变量，输出空目标潜变量")
            device = start_latent_samples.device
            dtype = start_latent_samples.dtype
            empty_latent = torch.zeros((1, start_latent_samples.shape[1], 0, start_latent_samples.shape[3], start_latent_samples.shape[4]), 
                                       dtype=dtype, device=device)
            target_latent_out_dict = {"samples": empty_latent}
            valid_target_index = -1

        return (start_latent_out_dict, target_latent_out_dict, valid_target_index, 
                video_length_seconds, fps, window_size)


NODE_CLASS_MAPPINGS = {
    "CreateKeyframes_HY": CreateKeyframes_HY,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateKeyframes_HY": "FramePack Create Keyframes (HY)",
} 