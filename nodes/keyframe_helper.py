# ComfyUI-FramePack-HY/nodes/keyframe_helper.py

import torch
import math

class CreateKeyframes_HY:
    """
    辅助节点：定义视频的起始关键帧。

    输入一个必需的起始潜变量 (keyframe_1) 来定义视频的开端。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframe_1": ("LATENT", {"tooltip": "定义视频起点的潜变量 (必需)"}),
                "video_length_seconds": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1,
                                                  "tooltip": "视频总时长(秒)"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1,
                               "tooltip": "视频帧率(每秒帧数)"}),
                "window_size": ("INT", {"default": 9, "min": 4, "max": 33, "step": 1,
                                       "tooltip": "采样上下文窗口大小。控制模型生成时考虑的历史信息长度，影响视频连贯性和计算开销。必须与Sampler节点的设置一致。"}),
            },
        }

    RETURN_TYPES = ("LATENT", "video_length_seconds", "video_fps", "window_size",)
    RETURN_NAMES = ("start_latent_out",  "video_length_seconds", "fps", "window_size",)
    FUNCTION = "create_start_keyframe"
    CATEGORY = "FramePack"
    DESCRIPTION = """定义视频的起始潜变量。

用于指定视频的开端 (keyframe_1)。Sampler节点将使用此潜变量作为视频生成的第一帧。

使用说明：
1. 输入必需的 keyframe_1 作为视频起点。
2. 设置视频参数 (时长、帧率、窗口大小)。
"""

    def create_start_keyframe(self, keyframe_1, video_length_seconds, fps, window_size):
        print(f"[CreateKeyframes_HY (Single Keyframe Mode)] 处理起始关键帧")
        print(f"[CreateKeyframes_HY (Single Keyframe Mode)] 视频参数: 时长={video_length_seconds}秒, 帧率={fps}fps, 窗口大小={window_size}")

        start_latent_samples = keyframe_1["samples"]
        print(f"[CreateKeyframes_HY (Single Keyframe Mode)] 处理起始潜变量 (来自keyframe_1)，形状: {start_latent_samples.shape}")
        if start_latent_samples.ndim == 4:
            start_latent_samples = start_latent_samples.unsqueeze(2)
        elif start_latent_samples.ndim != 5:
            raise ValueError(f"起始潜变量 (keyframe_1) 形状错误: {start_latent_samples.shape}")
        if start_latent_samples.shape[2] > 1:
            print(f"[CreateKeyframes_HY (Single Keyframe Mode)] 起始潜变量包含多帧，仅使用第一帧")
            start_latent_samples = start_latent_samples[:, :, :1, :, :]

        start_latent_out_dict = {"samples": start_latent_samples}

        print("[CreateKeyframes_HY (Single Keyframe Mode)] 成功准备起始潜变量")
        return (start_latent_out_dict, video_length_seconds, fps, window_size)


NODE_CLASS_MAPPINGS = {
    "CreateKeyframes_HY": CreateKeyframes_HY,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateKeyframes_HY": "FramePack Create Keyframes (HY)",
} 