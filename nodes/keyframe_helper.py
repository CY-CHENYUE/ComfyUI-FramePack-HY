# ComfyUI-FramePack-HY/nodes/keyframe_helper.py

import torch
import math
from ..diffusers_helper.utils import repeat_to_batch_size

class CreateKeyframes:
    """
    辅助节点：创建和管理多个关键帧输入
    
    输入多个潜变量及其对应的帧位置，创建一个按顺序排列的关键帧集合
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_keyframes": ("LATENT", {"tooltip": "第一个关键帧潜变量，通常是起始帧"}),
                "video_length_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.1, 
                                                  "tooltip": "视频总时长(秒)"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60, "step": 1, 
                               "tooltip": "视频帧率(每秒帧数)"}),
                "window_size": ("INT", {"default": 9, "min": 4, "max": 33, "step": 1, 
                                       "tooltip": "采样窗口大小参数，必须与sampler中的latent_window_size保持一致"}),
                "first_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, 
                                       "tooltip": "第一个关键帧的帧位置，通常设为0表示视频开始"}),
            },
            "optional": {
                # 其他关键帧设置
                "second_keyframes": ("LATENT", {"tooltip": "第二个关键帧潜变量"}),
                "second_frame": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1, 
                                        "tooltip": "第二个关键帧的帧位置"}),
                
                "third_keyframes": ("LATENT", {"tooltip": "第三个关键帧潜变量"}),
                "third_frame": ("INT", {"default": 60, "min": 0, "max": 1000, "step": 1, 
                                       "tooltip": "第三个关键帧的帧位置"}),
                
                "fourth_keyframes": ("LATENT", {"tooltip": "第四个关键帧潜变量"}),
                "fourth_frame": ("INT", {"default": 90, "min": 0, "max": 1000, "step": 1, 
                                        "tooltip": "第四个关键帧的帧位置"}),
                
                # 之前的关键帧
                "prev_keyframes": ("LATENT", {"tooltip": "之前创建的关键帧集合，用于级联更多关键帧"}),
                "prev_keyframe_indices": ("KEYFRAME_INDICES", {"default": "", "tooltip": "之前关键帧的索引字符串，格式为逗号分隔的数字，例如'0,5,10'"}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "KEYFRAME_INDICES", "video_length_seconds", "video_fps", "window_size",)
    RETURN_NAMES = ("keyframes", "keyframe_indices", "video_length_seconds", "fps", "window_size",)
    FUNCTION = "create_keyframes"
    CATEGORY = "FramePack/辅助节点"
    DESCRIPTION = """创建关键帧集合和对应的分段索引。

此节点允许您指定多个图像（潜变量）作为视频生成过程中的关键帧，并使用帧位置设置它们在视频中的位置。
系统会自动将帧位置转换为分段索引用于视频生成。

使用说明：
1. 将多个图像通过VAE编码转换为潜变量，输入为first_keyframes, second_keyframes等
2. 设置视频参数：总时长(秒)、帧率(fps)和窗口大小（与sampler中的latent_window_size保持一致）
3. 为每个潜变量指定一个帧位置，表示它应该出现在视频中的位置
4. 可以通过prev_keyframes和prev_keyframe_indices级联添加更多关键帧

注意：
- 第一个关键帧通常设置为帧位置0，表示视频开始
- 帧位置的设置应考虑视频总时长和帧率，例如5秒24fps的视频总共有120帧
- 此节点的输出可以直接连接到FramePack Sampler (HY)节点的keyframes和keyframe_indices输入
"""

    def frame_to_section_index(self, frame_position, fps, total_seconds, window_size):
        """将帧位置转换为分段索引"""
        # 计算每个窗口的有效帧数
        frames_per_window = window_size * 4 - 3
        
        # 计算总帧数和总分段数
        total_frames = total_seconds * fps
        total_sections = math.ceil(total_frames / frames_per_window)
        
        # 显示计算信息，帮助用户理解
        print(f"[CreateKeyframes] 总帧数: {total_frames}, 总分段数: {total_sections}")
        print(f"[CreateKeyframes] 每个分段对应约 {frames_per_window} 帧")
        
        # 将帧位置映射到分段索引
        if total_frames <= 0:
            return 0
        
        # 计算归一化位置(0-1)，然后映射到分段索引
        normalized_position = min(1.0, max(0.0, frame_position / total_frames))
        section_index = int(normalized_position * (total_sections - 1) + 0.5)  # 四舍五入
        
        print(f"[CreateKeyframes] 帧位置 {frame_position} 映射到分段索引 {section_index}")
        return section_index
    
    def create_keyframes(self, first_keyframes, video_length_seconds, fps, window_size, first_frame,
                         second_keyframes=None, second_frame=30,
                         third_keyframes=None, third_frame=60,
                         fourth_keyframes=None, fourth_frame=90,
                         prev_keyframes=None, prev_keyframe_indices=""):
        """使用帧位置创建关键帧"""
        # 初始化关键帧列表和索引列表
        keyframes_list = []
        indices_list = []
        
        print(f"[CreateKeyframes] 开始处理关键帧输入，使用帧位置模式")
        print(f"[CreateKeyframes] 视频参数: 时长={video_length_seconds}秒, 帧率={fps}fps, 窗口大小={window_size}")
        
        # 转换帧位置为分段索引
        first_index = self.frame_to_section_index(first_frame, fps, video_length_seconds, window_size)
        print(f"[CreateKeyframes] 第一个关键帧: 帧位置 {first_frame} → 分段索引 {first_index}")
        
        # 转换其他关键帧的帧位置（如果存在）
        second_index = None
        third_index = None
        fourth_index = None
        
        if second_keyframes is not None:
            second_index = self.frame_to_section_index(second_frame, fps, video_length_seconds, window_size)
            print(f"[CreateKeyframes] 第二个关键帧: 帧位置 {second_frame} → 分段索引 {second_index}")
        
        if third_keyframes is not None:
            third_index = self.frame_to_section_index(third_frame, fps, video_length_seconds, window_size)
            print(f"[CreateKeyframes] 第三个关键帧: 帧位置 {third_frame} → 分段索引 {third_index}")
        
        if fourth_keyframes is not None:
            fourth_index = self.frame_to_section_index(fourth_frame, fps, video_length_seconds, window_size)
            print(f"[CreateKeyframes] 第四个关键帧: 帧位置 {fourth_frame} → 分段索引 {fourth_index}")
        
        # 处理之前的关键帧
        if prev_keyframes is not None and prev_keyframe_indices:
            try:
                # 提取之前的关键帧潜变量
                prev_kf_samples = prev_keyframes["samples"]
                print(f"[CreateKeyframes] 检测到prev_keyframes输入，形状: {prev_kf_samples.shape}")
                
                # 确保潜变量有正确的形状
                if prev_kf_samples.ndim != 5:  # 必须是 [B, C, T, H, W]
                    if prev_kf_samples.ndim == 4:  # 如果是 [B, C, H, W]
                        prev_kf_samples = prev_kf_samples.unsqueeze(2)  # 添加时间维度
                        print(f"[CreateKeyframes] 为prev_keyframes添加时间维度，新形状: {prev_kf_samples.shape}")
                    else:
                        print(f"[CreateKeyframes] 警告: prev_keyframes形状错误 {prev_kf_samples.shape}")
                        prev_kf_samples = None
                
                # 解析之前的索引
                prev_indices = [int(idx.strip()) for idx in prev_keyframe_indices.split(',') if idx.strip()]
                print(f"[CreateKeyframes] 解析的prev_keyframe_indices: {prev_indices}")
                
                if prev_kf_samples is not None and len(prev_indices) > 0:
                    # 检查维度匹配
                    if prev_kf_samples.shape[2] == len(prev_indices):
                        # 将之前的关键帧添加到列表
                        keyframes_list.append(prev_kf_samples)
                        indices_list.extend(prev_indices)
                        print(f"[CreateKeyframes] 添加之前的{len(prev_indices)}个关键帧，索引: {prev_indices}")
                    else:
                        print(f"[CreateKeyframes] 警告: prev_keyframes帧数({prev_kf_samples.shape[2]})与prev_keyframe_indices长度({len(prev_indices)})不匹配")
            except Exception as e:
                print(f"[CreateKeyframes] 处理prev_keyframes时出错: {e}")
        
        # 处理新的关键帧
        new_kfs = []
        new_indices = []
        
        # 添加第一个关键帧（必需）
        first_kf_samples = first_keyframes["samples"]
        print(f"[CreateKeyframes] 处理first_keyframes，形状: {first_kf_samples.shape}, 帧位置: {first_frame}, 分段索引: {first_index}")
        
        # 确保是5D张量
        if first_kf_samples.ndim == 4:
            first_kf_samples = first_kf_samples.unsqueeze(2)
            print(f"[CreateKeyframes] 为first_keyframes添加时间维度，新形状: {first_kf_samples.shape}")
        elif first_kf_samples.ndim != 5:
            raise ValueError(f"关键帧形状错误: {first_kf_samples.shape}, 应为[B, C, H, W]或[B, C, T, H, W]")
        
        # 如果是多帧潜变量，只取第一帧
        if first_kf_samples.shape[2] > 1:
            first_kf_samples = first_kf_samples[:, :, :1, :, :]
            print(f"[CreateKeyframes] first_keyframes包含多帧，仅使用第一帧, 新形状: {first_kf_samples.shape}")
            
        new_kfs.append(first_kf_samples)
        new_indices.append(first_index)
        
        # 添加可选的关键帧
        optional_kfs = [
            ("second_keyframes", second_keyframes, second_index, second_frame), 
            ("third_keyframes", third_keyframes, third_index, third_frame), 
            ("fourth_keyframes", fourth_keyframes, fourth_index, fourth_frame)
        ]
        
        for name, kf, idx, frame in optional_kfs:
            if kf is not None and idx is not None:
                kf_samples = kf["samples"]
                print(f"[CreateKeyframes] 处理{name}，形状: {kf_samples.shape}, 帧位置: {frame}, 分段索引: {idx}")
                
                # 确保是5D张量
                if kf_samples.ndim == 4:
                    kf_samples = kf_samples.unsqueeze(2)
                    print(f"[CreateKeyframes] 为{name}添加时间维度，新形状: {kf_samples.shape}")
                elif kf_samples.ndim != 5:
                    print(f"[CreateKeyframes] 警告: {name}形状错误 {kf_samples.shape}, 跳过")
                    continue
                
                # 如果是多帧潜变量，只取第一帧
                if kf_samples.shape[2] > 1:
                    kf_samples = kf_samples[:, :, :1, :, :]
                    print(f"[CreateKeyframes] {name}包含多帧，仅使用第一帧, 新形状: {kf_samples.shape}")
                
                new_kfs.append(kf_samples)
                new_indices.append(idx)
                print(f"[CreateKeyframes] 成功添加{name}，帧位置: {frame}, 分段索引: {idx}")
            elif kf is not None:
                print(f"[CreateKeyframes] {name}存在，但索引计算错误，跳过")
            else:
                print(f"[CreateKeyframes] {name}为None，跳过")
        
        # 如果没有添加新的关键帧，直接返回之前的关键帧
        if not new_kfs and keyframes_list:
            # 合并所有之前的索引为字符串
            indices_str = ",".join(map(str, indices_list))
            print(f"[CreateKeyframes] 没有新关键帧，返回之前的{len(indices_list)}个关键帧")
            
            # 将视频参数直接作为返回值
            result = prev_keyframes.copy() if isinstance(prev_keyframes, dict) else {"samples": prev_keyframes["samples"].clone()}
            # 返回关键帧、索引和视频参数
            return result, indices_str, video_length_seconds, fps, window_size
        elif not new_kfs and not keyframes_list:
            print("[CreateKeyframes] 警告: 没有有效的关键帧输入")
            empty_latent = torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)
            result = {"samples": empty_latent}
            # 返回关键帧、索引和视频参数
            return result, "", video_length_seconds, fps, window_size
        
        # 将所有新关键帧添加到列表
        keyframes_list.extend(new_kfs)
        indices_list.extend(new_indices)
        
        print(f"[CreateKeyframes] 总共有{len(indices_list)}个关键帧待处理，索引列表: {indices_list}")
        
        # 按索引排序
        if indices_list:
            # 创建索引-关键帧对列表，并按索引排序
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
            combined_latent = torch.cat(combined_samples, dim=2).to(device=device, dtype=dtype)
            
            # 解决顺序问题：在返回前逆转关键帧顺序，但保持索引值不变
            # 沿时间维度 (dim=2) 逆转关键帧顺序
            combined_latent = torch.flip(combined_latent, dims=[2])
            print(f"[CreateKeyframes] 已逆转关键帧顺序，但保持原始索引值不变: {sorted_indices}")
            
            # 转换索引为字符串
            indices_str = ",".join(map(str, sorted_indices))
            
            print(f"[CreateKeyframes] 成功创建{len(sorted_indices)}个关键帧, 形状: {combined_latent.shape}, 索引字符串: {indices_str}")
            
            # 在最终返回前更新结果
            result = {"samples": combined_latent}
            # 返回关键帧、索引和视频参数
            return result, indices_str, video_length_seconds, fps, window_size
        else:
            # 没有关键帧时返回空结果
            print("[CreateKeyframes] 警告: 没有有效的关键帧")
            empty_latent = torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)
            result = {"samples": empty_latent}
            # 返回关键帧、索引和视频参数
            return result, "", video_length_seconds, fps, window_size


NODE_CLASS_MAPPINGS = {
    "CreateKeyframes_HY": CreateKeyframes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateKeyframes_HY": "FramePack CreateKeyframes (HY)",
} 