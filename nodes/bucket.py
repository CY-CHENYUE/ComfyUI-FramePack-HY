# ComfyUI-FramePack-HY/nodes/bucket.py
import torch
import torch.nn.functional as F
import numpy as np
import comfy.utils
import comfy.model_management as model_management
from ..diffusers_helper.bucket_tools import find_nearest_bucket, bucket_options

class FramePackBucketResize_HY:
    @classmethod
    def INPUT_TYPES(s):
        # 获取可用的分辨率列表并转换为字符串
        available_resolutions = list(bucket_options.keys())
        available_resolutions.sort()
        available_resolutions_str = [str(res) for res in available_resolutions]
        
        return {"required": {
            "image": ("IMAGE", {"tooltip": "要调整到最佳分桶分辨率的图像"}),
            "base_resolution": (available_resolutions_str, {"default": "640", "tooltip": "用于分桶的基础分辨率"}),
            },
            "optional": {
                "resize_mode": (["lanczos", "bilinear", "bicubic", "nearest"], {"default": "lanczos", 
                                 "tooltip": "调整图像尺寸的插值方法"}),
                "alignment": (["center", "top_left"], {"default": "center", 
                               "tooltip": "图像调整大小时的对齐方式"})
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("resized_image", "width", "height")
    FUNCTION = "process"
    CATEGORY = "FramePack"
    DESCRIPTION = "查找并将图像调整到最佳分辨率桶，输出调整后的图像、宽度和高度"

    def process(self, image, base_resolution, resize_mode="lanczos", alignment="center"):
        try:
            # 记录原始图像信息
            print(f"[分桶调整] 输入图像形状: {image.shape}, 类型: {image.dtype}")
            
            # 1. 确保输入图像格式正确 - ComfyUI 图像一般是 [B, H, W, C] 格式
            if len(image.shape) == 3 and image.shape[0] == 3:  # [C, H, W]
                # 转换为 [1, H, W, C]
                image = image.permute(1, 2, 0).unsqueeze(0)
                print(f"[分桶调整] 转换 [C, H, W] 到 [B, H, W, C]: {image.shape}")
            elif len(image.shape) == 4 and image.shape[1] == 3:  # [B, C, H, W]
                # 转换为 [B, H, W, C]
                image = image.permute(0, 2, 3, 1)
                print(f"[分桶调整] 转换 [B, C, H, W] 到 [B, H, W, C]: {image.shape}")
            
            # 验证我们现在有正确的 [B, H, W, C] 格式
            if len(image.shape) != 4 or image.shape[3] not in [1, 3, 4]:
                raise ValueError(f"图像格式异常: {image.shape}，ComfyUI需要[B, H, W, C]格式")
            
            # 2. 提取图像尺寸
            batch_size, original_height, original_width, channels = image.shape
            print(f"[分桶调整] 原始尺寸: {original_width}x{original_height}, 通道数: {channels}")
            
            # 3. 查找最佳分桶尺寸
            base_resolution = int(base_resolution)
            bucket_height, bucket_width = find_nearest_bucket(original_height, original_width, resolution=base_resolution)
            print(f"[分桶调整] 最优分桶尺寸: {bucket_width}x{bucket_height}")
            
            # 4. 为 PyTorch 调整大小，需要先转换为 [B, C, H, W] 格式
            tensor_for_resize = image.permute(0, 3, 1, 2).float()  # 转为 [B, C, H, W]
            
            # 映射resize_mode到PyTorch支持的模式
            mode_map = {
                "lanczos": "bicubic",  # PyTorch没有lanczos，使用bicubic替代
                "bilinear": "bilinear",
                "bicubic": "bicubic",
                "nearest": "nearest"
            }
            mode = mode_map.get(resize_mode, "bicubic")
            align_corners = None if mode == "nearest" else False
            
            # 5. 调整图像大小
            with torch.no_grad():
                resized_tensor = F.interpolate(
                    tensor_for_resize,
                    size=(bucket_height, bucket_width),
                    mode=mode,
                    align_corners=align_corners
                )
            
            # 6. 转回 [B, H, W, C] 格式，这是 ComfyUI 期望的格式
            resized_image = resized_tensor.permute(0, 2, 3, 1).contiguous()
            
            # 7. 确保数值范围在 0-1 之间
            resized_image = torch.clamp(resized_image, 0.0, 1.0)
            
            # 8. 分离计算图并确保是浮点数
            resized_image = resized_image.detach().float()
            
            # 9. 验证和修正最终输出格式
            if resized_image.shape != (batch_size, bucket_height, bucket_width, channels):
                print(f"[分桶调整] 警告: 输出尺寸不符合预期，进行修正")
                # 创建一个新的正确尺寸的张量
                correct_image = torch.zeros(
                    (batch_size, bucket_height, bucket_width, channels),
                    dtype=torch.float32,
                    device=resized_image.device
                )
                
                # 拷贝尽可能多的数据
                try:
                    h = min(resized_image.shape[1], bucket_height)
                    w = min(resized_image.shape[2], bucket_width)
                    c = min(resized_image.shape[3], channels)
                    correct_image[:, :h, :w, :c] = resized_image[:, :h, :w, :c]
                except Exception as e:
                    print(f"[分桶调整] 尺寸修正出错: {e}")
                
                resized_image = correct_image
            
            # 10. 进行 PIL 图像验证测试
            try:
                # 转换为 numpy 数组 (取第一张图片)
                test_image = resized_image[0].cpu().numpy()
                # 确保值范围为 0-255 的 uint8
                test_image = (test_image * 255).astype(np.uint8)
                
                # 如果是单通道图像且形状是 (H, W, 1)，移除最后一个维度
                if channels == 1 and test_image.shape[2] == 1:
                    test_image = test_image.squeeze(2)
                
                # 导入 PIL 并验证
                from PIL import Image
                pil_image = Image.fromarray(test_image)
                print(f"[分桶调整] PIL图像验证成功: {pil_image.size}")
            except Exception as e:
                print(f"[分桶调整] PIL验证警告 (仅供参考): {e}")
            
            # 11. 输出最终图像
            print(f"[分桶调整] 最终输出格式: {resized_image.shape}, 类型: {resized_image.dtype}")
            return resized_image, bucket_width, bucket_height
            
        except Exception as e:
            print(f"[分桶调整] 处理过程出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建一个安全的回退图像 (ComfyUI格式 [B, H, W, C])
            try:
                # 确保格式为 [B, H, W, C]
                safe_image = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.5
                
                # 验证
                test_numpy = safe_image[0].cpu().numpy()
                test_numpy = (test_numpy * 255).astype(np.uint8)
                
                from PIL import Image
                test_image = Image.fromarray(test_numpy)
                print(f"[分桶调整] 回退图像验证通过: {test_image.size}")
                
                return safe_image, 64, 64
            except Exception as backup_error:
                print(f"[分桶调整] 创建回退图像失败: {backup_error}")
                blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return blank, 64, 64


NODE_CLASS_MAPPINGS = {
    "FramePackBucketResize_HY": FramePackBucketResize_HY
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FramePackBucketResize_HY": "FramePack 分桶调整(HY)"
} 