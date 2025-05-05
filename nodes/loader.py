# ComfyUI-FramePack-HY/nodes/loader.py

import os
import torch
import folder_paths
import comfy.model_management as mm

# 从diffusers_helper导入相关函数和类
from ..diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from ..diffusers_helper.memory import DynamicSwapInstaller

SUPPORTED_PRECISIONS = ["auto", "fp16", "bf16", "fp32"]

class LoadFramePackDiffusersPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "lllyasviel/FramePackI2V_HY"}), # 保留此输入，允许用户指定模型名称或路径
                "precision": (SUPPORTED_PRECISIONS, {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("FP_DIFFUSERS_PIPELINE",)
    RETURN_NAMES = ("fp_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "FramePack"

    def load_pipeline(self, model_path, precision):
        print(f"[FramePack Loader] 尝试加载模型: '{model_path}'") # 保留初始加载信息

        # 确定 torch_dtype
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "auto": torch.bfloat16  # 默认使用bf16
        }
        torch_dtype = dtype_map.get(precision)

        # 自动检测模型精度 (基于文件名或路径，如果未明确指定)
        if precision == "auto":
            if "bf16" in model_path.lower():
                print("[FramePack Loader] 自动检测到bf16模型（基于路径名），设置dtype为bfloat16")
                torch_dtype = torch.bfloat16
            elif "fp16" in model_path.lower():
                print("[FramePack Loader] 自动检测到fp16模型（基于路径名），设置dtype为float16")
                torch_dtype = torch.float16
            else:
                 print("[FramePack Loader] 无法从路径名自动检测精度，使用默认值 bfloat16")
                 torch_dtype = torch.bfloat16

        # 模型路径查找逻辑
        resolved_model_path = None

        # 1. 检查输入是否为绝对路径
        if os.path.isabs(model_path) and os.path.isdir(model_path):
            # print(f"[FramePack Loader] 输入为绝对路径，直接使用: {model_path}") # 移除过程信息
            resolved_model_path = model_path
        else:
            # 2. 遍历 ComfyUI 配置的 diffusers 基础目录
            try:
                base_models_dirs = folder_paths.get_folder_paths("diffusers")
                if not base_models_dirs:
                     print("[FramePack Loader] 警告: 未找到配置的 ComfyUI diffusers 基础目录。") # 保留警告
                else:
                    # print(f"[FramePack Loader] 开始在以下基础目录中查找 '{model_path}': {base_models_dirs}") # 移除过程信息
                    for base_dir in base_models_dirs:
                        potential_path = os.path.join(base_dir, model_path)
                        # print(f"[FramePack Loader] 正在检查路径: {potential_path}") # 移除过程信息
                        if os.path.isdir(potential_path):
                            # print(f"[FramePack Loader] 在 ComfyUI diffusers 目录中找到模型: {potential_path}") # 移除过程信息
                            resolved_model_path = potential_path
                            break # 找到后即停止搜索

            except Exception as e:
                print(f"[FramePack Loader] 遍历基础目录查找模型时出错: {e}") # 保留错误信息

            # 3. 如果上述方法都未找到，最后检查一下输入本身是否是一个有效的相对路径（可能性较低）
            if not resolved_model_path and os.path.isdir(model_path):
                 # print(f"[FramePack Loader] 尝试将输入 '{model_path}' 作为相对路径使用。") # 移除过程信息
                 resolved_model_path = model_path

        # 检查最终的模型路径是否有效
        if resolved_model_path is None or not os.path.isdir(resolved_model_path):
            base_models_dir_list = folder_paths.get_folder_paths("diffusers") # 获取用于错误信息
            error_msg = (
                f"无法找到模型目录: '{model_path}'\n"
                f"尝试解析路径: {resolved_model_path or '无有效解析路径'}\n"
                f"请确保您已下载模型并将其放置在以下 ComfyUI 的 diffusers 目录之一中，且路径与输入 '{model_path}' 匹配:\n"
                + "\n".join([f"- {d}" for d in base_models_dir_list]) + "\n"
                f"或者直接在输入框中使用模型的绝对路径。\n"
                f"例如，对于 '{model_path}', 应该存在类似 '{os.path.join(base_models_dir_list[0] if base_models_dir_list else 'ComfyUI/models/diffusers', model_path)}' 的目录"
            )
            raise FileNotFoundError(error_msg) # 错误信息会终止执行，无需额外打印

        # 更新 model_path 为解析后的绝对路径
        model_path = resolved_model_path
        print(f"[FramePack Loader] 成功定位模型路径: {model_path}") # 保留最终使用的路径信息

        # 检查目录内容是否符合预期
        try:
            dir_contents = os.listdir(model_path)
            expected_file = "config.json"
            if expected_file not in dir_contents:
                 print(f"[FramePack Loader] 警告: 模型目录 {model_path} 中缺少关键文件: {expected_file}") # 保留警告
        except Exception as e:
            print(f"[FramePack Loader] 读取目录内容出错: {e}") # 保留错误信息

        # 加载模型
        try:
            print(f"[FramePack Loader] 开始加载模型...") # 保留加载开始信息

            device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()

            transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attention_mode="sdpa"
            ).cpu()

            if not hasattr(transformer, '_dynamic_swap_installed'):
                 DynamicSwapInstaller.install_model(transformer, device=device)
                 transformer._dynamic_swap_installed = True
            # else:
                 # print("[FramePack Loader] DynamicSwap 已安装，跳过。") # 移除过程信息

            pipe = {
                "transformer": transformer.eval(),
                "dtype": torch_dtype,
            }

            print("[FramePack Loader] 模型加载成功") # 保留加载成功信息
        except Exception as e:
            print(f"[FramePack Loader] 加载模型错误: {e}") # 保留错误信息
            raise RuntimeError(f"从 {model_path} (dtype: {torch_dtype}) 加载FramePack模型失败: {e}") # 错误信息会终止执行

        return (pipe,)

NODE_CLASS_MAPPINGS = {
    "LoadFramePackDiffusersPipeline_HY": LoadFramePackDiffusersPipeline
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFramePackDiffusersPipeline_HY": "Load FramePack Pipeline (HY)"
} 