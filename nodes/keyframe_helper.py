# ComfyUI-FramePack-HY/nodes/keyframe_helper.py

import torch
import math
from ..diffusers_helper.utils import repeat_to_batch_size

class CreateKeyframes:
    """
    辅助节点：创建和管理视频首尾关键帧输入 (无级联)

    输入首、尾两个潜变量，创建一个按顺序排列的关键帧集合，分别对应视频的第0段和最后一段。
    此节点独立工作，不接受来自上游节点的级联关键帧。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_keyframes": ("LATENT", {"tooltip": "第一个关键帧潜变量，对应视频开头"}),
                "video_length_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.1,
                                                  "tooltip": "视频总时长(秒)，用于计算总分段数"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1,
                               "tooltip": "视频帧率(每秒帧数)，用于计算总分段数"}),
                "window_size": ("INT", {"default": 9, "min": 4, "max": 33, "step": 1,
                                       "tooltip": "采样窗口大小参数，必须与sampler中的latent_window_size保持一致，用于计算总分段数"}),
                # 移除了 first_frame 输入
            },
            "optional": {
                # 第二个关键帧设置 (对应视频结尾)
                "second_keyframes": ("LATENT", {"tooltip": "第二个关键帧潜变量，对应视频结尾"}),
                # 移除了 second_frame 输入
                # 移除了第三、第四和之前的关键帧输入
            }
        }

    RETURN_TYPES = ("LATENT", "KEYFRAME_INDICES", "video_length_seconds", "video_fps", "window_size",)
    RETURN_NAMES = ("keyframes", "keyframe_indices", "video_length_seconds", "fps", "window_size",)
    FUNCTION = "create_keyframes"
    CATEGORY = "FramePack"
    DESCRIPTION = """创建首尾关键帧集合和对应的分段索引 (无级联)。

此节点接收代表视频开头和结尾的图像（潜变量）。
它会自动将它们分配给视频的第0个分段和最后一个分段。
此节点独立生成关键帧，不接受来自上游节点的级联输入。

使用说明：
1. 将1或2个图像通过VAE编码转换为潜变量，输入为first_keyframes（必需，开头）和second_keyframes（可选，结尾）
2. 设置视频参数：总时长(秒)、帧率(fps)和窗口大小（与sampler中的latent_window_size保持一致）

注意：
- 第一个关键帧总是对应分段索引 0。
- 如果提供了第二个关键帧，它总是对应最后一个分段索引。
- 此节点的输出可以直接连接到FramePack Sampler (HY)节点的keyframes和keyframe_indices输入
"""

    # 移除了 frame_to_section_index 函数

    def create_keyframes(self, first_keyframes, video_length_seconds, fps, window_size,
                         second_keyframes=None):
        # 移除了 first_frame, second_frame, prev_*, third_*, fourth_* 参数
        """使用固定的首尾索引创建关键帧 (最多2个, 无级联)"""
        # 初始化关键帧列表和索引列表
        keyframes_list = []
        indices_list = []

        print(f"[CreateKeyframes] 开始处理首尾关键帧输入 (无级联)")
        print(f"[CreateKeyframes] 视频参数: 时长={video_length_seconds}秒, 帧率={fps}fps, 窗口大小={window_size}")

        # --- 计算总分段数 (之前在 frame_to_section_index 中) ---
        frames_per_window = window_size * 4 - 3
        total_frames = video_length_seconds * fps
        if total_frames <= 0 or frames_per_window <= 0:
             total_sections = 1 # 至少有1段
        else:
             total_sections = math.ceil(total_frames / frames_per_window)
        total_sections = max(1, int(total_sections)) # 确保至少为1
        print(f"[CreateKeyframes] 计算得到总帧数: {total_frames}, 总分段数: {total_sections}")
        # -----------------------------------------------------

        # 固定索引
        first_index = 0
        print(f"[CreateKeyframes] 第一个关键帧固定到分段索引: {first_index}")

        second_index = None
        if second_keyframes is not None:
            second_index = max(0, total_sections - 1) # 对应最后一个分段索引
            print(f"[CreateKeyframes] 第二个关键帧固定到最后一个分段索引: {second_index}")
            # 如果总共只有1段，那么 second_index 也会是 0
            if first_index == second_index:
                 print("[CreateKeyframes] 注意: 只有一个分段，首尾关键帧索引相同 ({first_index})。将只使用第一个关键帧。")
                 # 设置 second_keyframes 为 None，避免后续处理和索引冲突
                 second_keyframes = None
                 second_index = None

        # 处理关键帧 (最多2个)

        # 添加第一个关键帧（必需）
        first_kf_samples = first_keyframes["samples"]
        print(f"[CreateKeyframes] 处理第一个关键帧 (首帧)，形状: {first_kf_samples.shape}")

        # 确保是5D张量
        if first_kf_samples.ndim == 4:
            first_kf_samples = first_kf_samples.unsqueeze(2)
            print(f"[CreateKeyframes] 为第一个关键帧添加时间维度，新形状: {first_kf_samples.shape}")
        elif first_kf_samples.ndim != 5:
            raise ValueError(f"第一个关键帧形状错误: {first_kf_samples.shape}, 应为[B, C, H, W]或[B, C, T, H, W]")

        # 如果是多帧潜变量，只取第一帧
        if first_kf_samples.shape[2] > 1:
            first_kf_samples = first_kf_samples[:, :, :1, :, :]
            print(f"[CreateKeyframes] 第一个关键帧包含多帧，仅使用第一帧, 新形状: {first_kf_samples.shape}")

        keyframes_list.append(first_kf_samples)
        indices_list.append(first_index)

        # 添加第二个关键帧 (如果存在且有效)
        if second_keyframes is not None and second_index is not None:
            kf_samples = second_keyframes["samples"]
            print(f"[CreateKeyframes] 处理第二个关键帧 (尾帧)，形状: {kf_samples.shape}, 索引: {second_index}")

            # 确保是5D张量
            if kf_samples.ndim == 4:
                kf_samples = kf_samples.unsqueeze(2)
                print(f"[CreateKeyframes] 为第二个关键帧添加时间维度，新形状: {kf_samples.shape}")
            elif kf_samples.ndim != 5:
                print(f"[CreateKeyframes] 警告: 第二个关键帧形状错误 {kf_samples.shape}, 跳过")
                # 跳过，不添加
            else:
                 # 如果是多帧潜变量，只取第一帧
                if kf_samples.shape[2] > 1:
                    kf_samples = kf_samples[:, :, :1, :, :]
                    print(f"[CreateKeyframes] 第二个关键帧包含多帧，仅使用第一帧, 新形状: {kf_samples.shape}")

                keyframes_list.append(kf_samples)
                indices_list.append(second_index)
                print(f"[CreateKeyframes] 成功添加第二个关键帧 (尾帧)，索引: {second_index}")

        elif second_keyframes is not None:
             # 这种情况理论上不应该发生，因为我们在上面处理了索引冲突
             print(f"[CreateKeyframes] 第二个关键帧存在，但索引无效，跳过")
        else:
             print(f"[CreateKeyframes] 未提供第二个关键帧 (尾帧)")

        # 检查是否至少有一个有效的关键帧 (理论上总有第一个)
        if not keyframes_list:
            print("[CreateKeyframes] 警告: 没有有效的关键帧输入")
            empty_latent = torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)
            result = {"samples": empty_latent}
            # 返回关键帧、索引和视频参数
            return result, "", video_length_seconds, fps, window_size

        print(f"[CreateKeyframes] 总共有{len(indices_list)}个关键帧待处理，索引列表: {indices_list}")

        # 按索引排序 (现在只有 0 或 [0, N-1])
        pairs = sorted(zip(indices_list, range(len(keyframes_list))), key=lambda x: x[0])
        sorted_indices = [p[0] for p in pairs]
        sorted_kf_indices = [p[1] for p in pairs]

        # 按排序后的顺序重新排列关键帧
        sorted_keyframes = []
        for idx in sorted_kf_indices:
            sorted_keyframes.append(keyframes_list[idx])

        # 打印排序信息
        print(f"[CreateKeyframes] 关键帧已排序，索引: {sorted_indices}")

        # 合并所有排序后的关键帧
        if not sorted_keyframes:
             print("[CreateKeyframes] 警告: 排序后关键帧列表为空")
             empty_latent = torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)
             return {"samples": empty_latent}, "", video_length_seconds, fps, window_size

        device = sorted_keyframes[0].device
        dtype = sorted_keyframes[0].dtype

        # 确保所有关键帧尺寸一致
        height = sorted_keyframes[0].shape[3]
        width = sorted_keyframes[0].shape[4]
        channels = sorted_keyframes[0].shape[1]
        batch_size = sorted_keyframes[0].shape[0]

        print(f"[CreateKeyframes] 基准尺寸: {batch_size}x{channels}x{height}x{width}, 设备: {device}, 类型: {dtype}")

        # 调整大小并组合
        combined_samples = []
        for i, kf in enumerate(sorted_keyframes):
            print(f"[CreateKeyframes] 处理第{i+1}个关键帧 (索引 {sorted_indices[i]}), 形状: {kf.shape}")
            current_height = kf.shape[3]
            current_width = kf.shape[4]

            # 如果尺寸不匹配，调整大小
            if current_height != height or current_width != width:
                print(f"[CreateKeyframes] 调整关键帧大小从 {current_height}x{current_width} 到 {height}x{width}")
                # 展平并调整尺寸
                b, c, t = kf.shape[0], kf.shape[1], kf.shape[2]
                flat = kf.reshape(b*c*t, 1, current_height, current_width)
                resized = torch.nn.functional.interpolate(
                    flat, size=(height, width), mode='bilinear', align_corners=False
                )
                kf = resized.reshape(b, c, t, height, width)

            # 确保批次大小一致
            if kf.shape[0] != batch_size:
                print(f"[CreateKeyframes] 调整批次大小从 {kf.shape[0]} 到 {batch_size}")
                kf = repeat_to_batch_size(kf, batch_size)

            combined_samples.append(kf)

        # 沿时间维度拼接
        if not combined_samples:
            print("[CreateKeyframes] 警告: 调整大小和组合后列表为空")
            empty_latent = torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)
            return {"samples": empty_latent}, "", video_length_seconds, fps, window_size

        combined_latent = torch.cat(combined_samples, dim=2).to(device=device, dtype=dtype)

        # 沿时间维度 (dim=2) 逆转关键帧顺序以匹配Sampler的反向累积
        combined_latent = torch.flip(combined_latent, dims=[2]) # 注释掉以保持原始顺序
        print(f"[CreateKeyframes] 已翻转关键帧顺序以适应Sampler: {sorted_indices}") # 不再翻转，注释掉此打印

        # 转换索引为字符串
        indices_str = ",".join(map(str, sorted_indices))

        print(f"[CreateKeyframes] 成功创建{len(sorted_indices)}个关键帧, 形状: {combined_latent.shape}, 索引字符串: {indices_str}")

        # 在最终返回前更新结果
        result = {"samples": combined_latent}
        # 返回关键帧、索引和视频参数
        return result, indices_str, video_length_seconds, fps, window_size


NODE_CLASS_MAPPINGS = {
    "CreateKeyframes_HY": CreateKeyframes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateKeyframes_HY": "FramePack CreateKeyframes (HY)",
} 