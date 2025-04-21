# ComfyUI-FramePack-HY/nodes/loader.py

import os
import torch
import folder_paths
import comfy.model_management as mm

# 从diffusers_helper导入相关函数和类
from ..diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from ..diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from ..diffusers_helper.memory import DynamicSwapInstaller

SUPPORTED_PRECISIONS = ["auto", "fp16", "bf16", "fp32"]

class LoadFramePackDiffusersPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "lllyasviel/FramePackI2V_HY"}),
                "precision": (SUPPORTED_PRECISIONS, {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("FP_DIFFUSERS_PIPELINE",)
    RETURN_NAMES = ("fp_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "FramePack"

    def load_pipeline(self, model_path, precision):
        print(f"[FramePack Loader] 加载模型，输入路径: '{model_path}'")

        # 确定 torch_dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "auto": torch.bfloat16  # 默认使用bf16
        }
        torch_dtype = dtype_map.get(precision)
        
        # 自动检测模型精度
        if precision == "auto" and "bf16" in model_path.lower():
            print("[FramePack Loader] 自动检测到bf16模型，设置dtype为bfloat16")
            torch_dtype = torch.bfloat16
        elif precision == "auto" and "fp16" in model_path.lower():
            print("[FramePack Loader] 自动检测到fp16模型，设置dtype为float16")
            torch_dtype = torch.float16

        # 处理模型路径
        try:
            # 确保folder_paths.models_dir是可用的
            if hasattr(folder_paths, 'models_dir'):
                direct_model_path = os.path.join(folder_paths.models_dir, "diffusers", model_path)
                print(f"[FramePack Loader] 尝试直接路径: {direct_model_path}")
                if os.path.isdir(direct_model_path):
                    model_path = direct_model_path
                    print(f"[FramePack Loader] 使用直接构建的路径: {model_path}")
            else:
                print("[FramePack Loader] folder_paths.models_dir 不可用")
        except Exception as e:
            print(f"[FramePack Loader] 直接构建路径异常: {e}")

        # 尝试使用get_full_path
        if not os.path.isdir(model_path):
            try:
                full_path = folder_paths.get_full_path("diffusers", model_path)
                if full_path and os.path.isdir(full_path):
                    model_path = full_path
                    print(f"[FramePack Loader] 使用folder_paths.get_full_path找到路径: {model_path}")
            except Exception as e:
                print(f"[FramePack Loader] 使用get_full_path异常: {e}")

        # 硬编码的常见路径
        if not os.path.isdir(model_path):
            common_paths = [
                r"E:\ComfyUI_windows_portable\ComfyUI\models\diffusers\lllyasviel\FramePackI2V_HY",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "diffusers", "lllyasviel", "FramePackI2V_HY")
            ]
            for path in common_paths:
                print(f"[FramePack Loader] 尝试常见路径: {path}")
                if os.path.isdir(path):
                    model_path = path
                    print(f"[FramePack Loader] 使用常见路径: {model_path}")
                    break

        # 最后检查模型路径是否有效
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"无法找到模型目录: {model_path}。请确保路径正确或将模型放在ComfyUI的diffusers模型文件夹中。")

        # 检查目录内容
        try:
            dir_contents = os.listdir(model_path)
            print(f"[FramePack Loader] 找到的模型目录内容: {dir_contents[:5]}{'...' if len(dir_contents) > 5 else ''}")
        except Exception as e:
            print(f"[FramePack Loader] 读取目录内容异常: {e}")

        # 加载模型 - 使用与用户提供的代码相同的方式直接加载
        try:
            print(f"[FramePack Loader] 开始加载模型 (直接加载)...")
            
            device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()
            
            # 直接加载模型
            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                model_path, 
                torch_dtype=torch_dtype, 
                attention_mode="sdpa"
            ).cpu()
            
            # 安装动态交换
            DynamicSwapInstaller.install_model(transformer, device=device)
            
            # 创建简单的pipeline结构
            pipe = {
                "transformer": transformer.eval(),
                "dtype": torch_dtype,
            }
                
            print("[FramePack Loader] 模型加载成功。")
        except Exception as e:
            print(f"[FramePack Loader] 加载模型错误: {e}")
            raise RuntimeError(f"从 {model_path} 加载FramePack模型失败: {e}")

        # 返回模型对象
        return (pipe,)

NODE_CLASS_MAPPINGS = {
    "LoadFramePackDiffusersPipeline_HY": LoadFramePackDiffusersPipeline
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFramePackDiffusersPipeline_HY": "Load FramePack Pipeline (HY)"
} 