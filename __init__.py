# ComfyUI-FramePack-HY/__init__.py
import sys
import os
import traceback

# 获取自定义节点根目录的绝对路径
framepack_hy_root = os.path.dirname(os.path.abspath(__file__))

# 检查与导入diffusers_helper相关的路径
diffusers_helper_path = os.path.join(framepack_hy_root, "diffusers_helper")
if os.path.exists(diffusers_helper_path):
    if os.path.isdir(diffusers_helper_path):
        # 列出diffusers_helper目录中的内容
        try:
            helper_contents = os.listdir(diffusers_helper_path)
        except Exception as e:
            print(f"[ComfyUI-FramePack-HY] 无法列出diffusers_helper目录内容: {e}")
else:
    print(f"[ComfyUI-FramePack-HY] 警告: diffusers_helper路径不存在，这可能导致导入错误")

# 临时将根目录添加到sys.path
original_sys_path = sys.path[:] # 保存原始路径
if framepack_hy_root not in sys.path:
    sys.path.insert(0, framepack_hy_root) # 添加到开头

# 在try-except之前初始化映射
loader_mappings = {}
loader_display_mappings = {}
sampler_mappings = {}
sampler_display_mappings = {}
bucket_mappings = {}
bucket_display_mappings = {}

# 导入节点并合并它们的映射
try:
    # 现在这些导入应该能够找到 ..diffusers_helper
    from .nodes.loader import NODE_CLASS_MAPPINGS as loader_mappings, NODE_DISPLAY_NAME_MAPPINGS as loader_display_mappings
except ImportError as e:
    print(f"[ComfyUI-FramePack-HY] 警告: 导入loader节点失败: {e}")
    # 导入错误时重置映射
    loader_mappings = {}
    loader_display_mappings = {}
except Exception as e: # 捕获导入期间的其他潜在错误
    print(f"[ComfyUI-FramePack-HY] loader节点导入期间出错: {e}")
    traceback.print_exc()
    loader_mappings = {}
    loader_display_mappings = {}

try:
    from .nodes.sampler import NODE_CLASS_MAPPINGS as sampler_mappings, NODE_DISPLAY_NAME_MAPPINGS as sampler_display_mappings
except ImportError as e:
    # 更具体地检查原始错误
    if "diffusers_helper" in str(e):
        print(f"[ComfyUI-FramePack-HY] 错误: 仍然无法找到相对于 {framepack_hy_root} 的'diffusers_helper'模块。请检查目录结构。")
        # 尝试直接从根目录导入
        try:
            # 临时修改sys.path
            sys.path.insert(0, os.path.dirname(framepack_hy_root))
            import ComfyUI_windows_portable.ComfyUI.custom_nodes.ComfyUI_FramePack_HY.diffusers_helper
        except ImportError:
            print("[ComfyUI-FramePack-HY] 尝试使用绝对路径导入diffusers_helper失败")
    print(f"[ComfyUI-FramePack-HY] 警告: 导入sampler节点失败: {e}")
    # 导入错误时重置映射
    sampler_mappings = {}
    sampler_display_mappings = {}
except Exception as e: # 捕获导入期间的其他潜在错误
    print(f"[ComfyUI-FramePack-HY] sampler节点导入期间出错: {e}")
    traceback.print_exc()
    sampler_mappings = {}
    sampler_display_mappings = {}

# 添加对bucket模块的导入
try:
    # 先检查一下缓存情况
    import sys
    cached_modules = [m for m in sys.modules.keys() if 'bucket' in m]
    if cached_modules:
        for module in cached_modules:
            if module in sys.modules:
                del sys.modules[module]
    
    # 进行导入
    from .nodes.bucket import NODE_CLASS_MAPPINGS as bucket_mappings, NODE_DISPLAY_NAME_MAPPINGS as bucket_display_mappings
    
    # 验证节点是否有效
    if not bucket_mappings:
        print(f"[ComfyUI-FramePack-HY] 警告: 导入的bucket_mappings为空!")
except ImportError as e:
    print(f"[ComfyUI-FramePack-HY] 警告: 导入bucket节点失败: {e}")
    bucket_mappings = {}
    bucket_display_mappings = {}
except Exception as e:
    print(f"[ComfyUI-FramePack-HY] bucket节点导入期间出错: {e}")
    traceback.print_exc()
    bucket_mappings = {}
    bucket_display_mappings = {}

finally:
    # --- 重要: 清理 sys.path ---
    # 移除我们添加的路径，避免全局污染sys.path
    try:
        sys.path.remove(framepack_hy_root)
    except ValueError:
        pass # 路径未找到或已被移除

# 初始化最终映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 合并映射（处理潜在的导入错误）
if loader_mappings:
    NODE_CLASS_MAPPINGS.update(loader_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(loader_display_mappings)
if sampler_mappings:
    NODE_CLASS_MAPPINGS.update(sampler_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(sampler_display_mappings)
if bucket_mappings:
    NODE_CLASS_MAPPINGS.update(bucket_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(bucket_display_mappings)

# 导出合并的映射
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

if not NODE_DISPLAY_NAME_MAPPINGS:
    print("  - 没有节点被注册。")
 