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

        # 模型路径查找
        model_found = False
        base_models_dir = getattr(folder_paths, 'models_dir', None)
        
        # 1. 直接检查绝对路径
        if os.path.isdir(model_path):
            model_found = True
            
        # 2. 尝试get_folder_paths方法
        if not model_found:
            try:
                if hasattr(folder_paths, 'get_folder_paths'):
                    diffusers_folders = folder_paths.get_folder_paths("diffusers")
                    
                    # 在所有diffusers文件夹中查找
                    for folder in diffusers_folders:
                        full_path = os.path.join(folder, model_path)
                        if os.path.isdir(full_path):
                            model_path = full_path
                            model_found = True
                            break
            except Exception as e:
                print(f"[FramePack Loader] 使用get_folder_paths查找出错: {e}")
        
        # 3. 检查标准的ComfyUI路径格式
        if not model_found and base_models_dir:
            # 处理形如 "lllyasviel/FramePackI2V_HY" 的路径
            standard_path = os.path.join(base_models_dir, "diffusers", model_path)
            if os.path.isdir(standard_path):
                model_path = standard_path
                model_found = True
                
            # 如果是斜杠分隔的路径，尝试按层级构建
            elif "/" in model_path:
                parts = model_path.split("/")
                constructed_path = os.path.join(base_models_dir, "diffusers", *parts)
                if os.path.isdir(constructed_path):
                    model_path = constructed_path
                    model_found = True
                    
                # 仅使用最后一部分
                else:
                    last_part = parts[-1]
                    simple_path = os.path.join(base_models_dir, "diffusers", last_part)
                    if os.path.isdir(simple_path):
                        model_path = simple_path
                        model_found = True
        
        # 4. 尝试手动构建一些常见路径
        if not model_found and base_models_dir:
            common_paths = [
                os.path.join(base_models_dir, "diffusers", model_path),
                os.path.join(base_models_dir, model_path),
            ]
            
            if "/" in model_path:
                parts = model_path.split("/")
                if len(parts) > 1:
                    model_name = parts[-1]
                    org_name = parts[0]
                    common_paths.extend([
                        os.path.join(base_models_dir, "diffusers", org_name, model_name),
                        os.path.join(base_models_dir, "diffusers", model_name),
                    ])
            
            for path in common_paths:
                if os.path.isdir(path):
                    model_path = path
                    model_found = True
                    break
            
            # 在当前目录下查找
            if not model_found:
                cwd = os.getcwd()
                for root, dirs, files in os.walk(os.path.join(cwd, "models")):
                    if os.path.basename(root) == "FramePackI2V_HY":
                        model_path = root
                        model_found = True
                        break
        
        # 检查最终的模型路径是否有效
        if not os.path.isdir(model_path):
            error_msg = (
                f"无法找到模型目录: {model_path}\n"
                f"调试信息:\n"
                f"- ComfyUI模型目录: {base_models_dir if base_models_dir else '未知'}\n"
                f"- 当前工作目录: {os.getcwd()}\n"
                f"请确保您已下载模型并放在以下位置:\n"
                f"ComfyUI/models/diffusers/{model_path}\n"
                f"或使用绝对路径直接指定模型位置。"
            )
            raise FileNotFoundError(error_msg)

        # 检查目录内容是否符合预期
        try:
            dir_contents = os.listdir(model_path)
            # 检查是否包含必要的模型文件
            expected_files = ["config.json", "model.safetensors"]
            for file in expected_files:
                if not any(f == file or f.endswith(file) for f in dir_contents):
                    print(f"[FramePack Loader] 警告: 未找到预期的模型文件 {file}")
        except Exception as e:
            print(f"[FramePack Loader] 读取目录内容出错: {e}")

        # 加载模型
        try:
            print(f"[FramePack Loader] 开始加载模型...")
            
            device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()
            
            # 加载模型
            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                model_path, 
                torch_dtype=torch_dtype, 
                attention_mode="sdpa"
            ).cpu()
            
            # 安装动态交换
            DynamicSwapInstaller.install_model(transformer, device=device)
            
            # 创建pipeline结构
            pipe = {
                "transformer": transformer.eval(),
                "dtype": torch_dtype,
            }
                
            print("[FramePack Loader] 模型加载成功")
        except Exception as e:
            print(f"[FramePack Loader] 加载模型错误: {e}")
            raise RuntimeError(f"从 {model_path} 加载FramePack模型失败: {e}")

        # 返回模型
        return (pipe,)

NODE_CLASS_MAPPINGS = {
    "LoadFramePackDiffusersPipeline_HY": LoadFramePackDiffusersPipeline
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFramePackDiffusersPipeline_HY": "Load FramePack Pipeline (HY)"
} 