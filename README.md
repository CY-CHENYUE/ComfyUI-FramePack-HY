# ComfyUI-FramePack-HY

这是一个为 ComfyUI 设计的自定义节点包，旨在通过分段采样（FramePack）的方式，利用 Hunyuan-DiT 类模型高效生成长视频。

<div>
  <video controls width="30%" src="video/CY_00335.mp4"></video>
  <video controls width="30%" src="video/CY_00340.mp4"></video>
  <video controls width="30%" src="video/CY_00341.mp4"></video>
</div>

## 功能特性

*   **分段采样 (FramePack):** 将长视频的生成过程分解为多个重叠的窗口（分段），在每个分段内独立进行采样，有效降低显存消耗，使得在有限的硬件资源下生成更长的视频成为可能。
*   **起始帧定义:** 提供专门的节点来设置视频的起始关键帧，确保视频生成的起点符合预期。
*   **内存优化:** 集成了内存管理策略，在模型加载和采样过程中尝试优化显存占用。
*   **基于 Hunyuan-DiT:** 主要适配腾讯混元 DiT 架构的视频模型。

## 包含节点

### 1. Load FramePack Pipeline (HY) (`LoadFramePackDiffusersPipeline_HY`)

*   **功能:** 加载 FramePack 工作流所需的核心视频模型（目前特指 Hunyuan Transformer）。它会自动查找 ComfyUI 配置的 `diffusers` 文件夹中的模型，并处理精度设置。
*   **输入:**
    *   `model_path` (STRING): 模型名称（例如 `lllyasviel/FramePackI2V_HY`）或模型的本地路径。节点会自动在 ComfyUI 的 `diffusers` 目录中查找。
    *   `precision` (STRING): 模型加载精度 (`auto`, `fp16`, `bf16`, `fp32`)。`auto` 会尝试根据模型名称推断，默认为 `bf16`。
*   **输出:**
    *   `fp_pipeline` (FP_DIFFUSERS_PIPELINE): 一个包含已加载模型 (`transformer`) 和数据类型 (`dtype`) 的 Pipeline 对象，供采样器使用。

### 2. FramePack BucketResize (HY) (`FramePackBucketResize_HY`)

*   **功能:** 将输入图像调整到预定义的分辨率桶 (bucket) 中最接近的尺寸。这有助于优化某些模型的性能（如 SDXL 或 Hunyuan）。
*   **输入:**
    *   `image` (IMAGE): 需要调整尺寸的图像。
    *   `base_resolution` (STRING): 用于选择分桶策略的基础分辨率 (例如 "640", "1024")。
    *   `resize_mode` (STRING, 可选): 调整尺寸时使用的插值方法 (`lanczos`, `bilinear`, `bicubic`, `nearest`)。
    *   `alignment` (STRING, 可选): 调整尺寸时的对齐方式 (`center`, `top_left`)。
*   **输出:**
    *   `resized_image` (IMAGE): 调整到最佳分桶尺寸后的图像。
    *   `width` (INT): 调整后的图像宽度。
    *   `height` (INT): 调整后的图像高度。

### 3. FramePack Create Keyframes (HY) (`CreateKeyframes_HY`)

*   **功能:** 定义视频生成的起点和基本参数。
*   **输入:**
    *   `keyframe_1` (LATENT): 必需的起始潜变量，定义视频的第一帧。通常来自文生图或图生图节点。
    *   `video_length_seconds` (INT): 期望生成的视频总时长（秒）。
    *   `fps` (INT): 视频的帧率（每秒帧数）。
    *   `window_size` (INT): 采样上下文窗口的大小。这个参数非常重要，它决定了模型在生成当前帧时会回顾多少历史帧信息，影响视频的时间连贯性。**此参数必须与 `FramePack Sampler (HY)` 节点的 `window_size` 保持一致。**
*   **输出:**
    *   `start_latent_out` (LATENT): 处理后的起始潜变量，仅包含第一帧。
    *   `video_length_seconds` (video_length_seconds): 传递视频时长。
    *   `fps` (video_fps): 传递视频帧率。
    *   `window_size` (window_size): 传递窗口大小。

### 4. FramePack Sampler (HY) (`FramePackDiffusersSampler_HY`)

*   **功能:** 执行核心的分段视频采样过程。
*   **输入:**
    *   `fp_pipeline` (FP_DIFFUSERS_PIPELINE): 来自加载器节点的 Pipeline 对象。
    *   `positive` / `negative` (CONDITIONING): 正向和负向文本条件。
    *   `clip_vision` (CLIP_VISION_OUTPUT): CLIP Vision 输出，用于图像引导。
    *   `steps` (INT): 每段采样的步数。
    *   `cfg` (FLOAT): Classifier-Free Guidance (CFG) 强度。
    *   `guidance_scale` (FLOAT): 蒸馏引导强度 (针对 Hunyuan)。
    *   `seed` (INT): 随机种子。
    *   `width` / `height` (INT): 视频宽高。
    *   `gpu_memory_preservation` (FLOAT): GPU 显存保留量 (GB)，用于内存管理。
    *   `sampler` (STRING): 采样器类型（目前仅支持 `unipc`）。
    *   `start_latent_out` (LATENT): 来自 `CreateKeyframes_HY` 节点的起始潜变量。
    *   `video_length_seconds` / `video_fps` / `window_size`: 从 `CreateKeyframes_HY` 节点连接，用于确定总帧数、分段数等。
    *   `shift` (FLOAT): 影响运动幅度的参数。
    *   `use_teacache` (BOOLEAN): 是否启用 teacache 加速。
    *   `teacache_thresh` (FLOAT): teacache 阈值。
    *   `denoise_strength` (FLOAT): 去噪强度 (未来可能用于图生视频)。
*   **输出:**
    *   `LATENT`: 生成的包含所有视频帧的潜变量序列。

## 安装方法

**方法一：通过 ComfyUI Manager (推荐)**

1.  打开 ComfyUI Manager。
2.  点击 "Install Custom Nodes"。
3.  搜索 "FramePack-HY" 或 "CY-CHENYUE"。
4.  找到 `ComfyUI-FramePack-HY`，点击 "Install"。
5.  等待安装完成，然后**重启 ComfyUI**。

**方法二：手动安装 (Manual Installation)**

1.  **克隆仓库:** 打开终端或命令行窗口，导航到你的 ComfyUI `custom_nodes` 目录，然后执行：
    ```bash
    cd /path/to/ComfyUI/custom_nodes
    git clone https://github.com/CY-CHENYUE/ComfyUI-FramePack-HY.git
    ```
    *(请将 `/path/to/ComfyUI/custom_nodes` 替换为你的实际路径)*

2.  **安装依赖:** 克隆完成后，切换到新克隆的 `ComfyUI-FramePack-HY` 目录，并根据你的系统和环境执行相应的命令：

    *   **Windows 系统:**
        ```bash
        cd ComfyUI-FramePack-HY
        
        # 如果你使用 ComfyUI 便携版 (Portable version):
        ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
        
        # 如果你使用自己的 Python 环境 (venv, conda 等):
        # 确保已激活你的环境
        # C:\path\to\your\python.exe -m pip install -r requirements.txt (替换为你的 Python 路径)
        pip install -r requirements.txt # 通常在激活环境后可以直接使用 pip
        ```

    *   **Linux / Mac 系统:**
        ```bash
        cd ComfyUI-FramePack-HY
        
        # 如果 ComfyUI 使用内置 Python (通常在便携版或特定安装脚本中):
        # ../../python_embeded/bin/python -m pip install -r requirements.txt
        
        # 如果你使用系统的 Python 或虚拟环境 (venv, conda 等):
        # 确保已激活你的环境
        python -m pip install -r requirements.txt # 或者 pip install -r requirements.txt
        ```

3.  **重启 ComfyUI:** 安装完依赖后，**重启 ComfyUI**。

**注意 (Troubleshooting):**

*   如果遇到安装问题：
    *   确保你已经安装了 `git`。
    *   尝试更新 `pip`： `path/to/python -m pip install --upgrade pip` (使用你的 Python 路径)。
    *   如果你使用代理，请确保 `git` 和 `pip` 已正确配置代理。
    *   **最重要：确保你用来安装依赖的 Python 环境与运行 ComfyUI 的 Python 环境是同一个。**

## 基本使用流程

1.  **加载模型:** 使用 `Load FramePack Pipeline (HY)` 节点加载所需的视频模型（例如 `lllyasviel/FramePackI2V_HY`），设置好模型路径和精度，得到 `fp_pipeline` 输出。
2.  **准备起始帧:**
    *   使用文生图（例如 `KSampler`）或加载图像节点生成/加载一张图像。
    *   **(可选但推荐)** 将该图像连接到 `FramePack BucketResize (HY)` 节点，选择合适的基础分辨率，获得尺寸优化后的图像 `resized_image`。
    *   将原始图像或 `resized_image` 连接到 VAE 编码器 (`VAE Encode`)，得到起始潜变量。
3.  **定义视频参数:**
    *   将 VAE 编码器输出的潜变量连接到 `FramePack Create Keyframes (HY)` 的 `keyframe_1` 输入。
    *   在 `FramePack Create Keyframes (HY)` 上设置期望的视频时长、帧率和**关键的 `window_size`**。
4.  **采样:**
    *   将 `Load FramePack Pipeline (HY)` 的 `fp_pipeline` 输出连接到 `FramePack Sampler (HY)` 的 `fp_pipeline` 输入。
    *   连接文本编码器的正负向条件 (`positive`, `negative`)。
    *   连接 CLIP Vision 输出（如果需要图像引导）。
    *   将 `FramePack Create Keyframes (HY)` 的 `start_latent_out`, `video_length_seconds`, `fps`, `window_size` 输出连接到 `FramePack Sampler (HY)` 对应的输入。
    *   设置采样参数（步数、CFG、种子、宽度/高度 - **注意：这里的宽高应与 BucketResize 输出的宽高或 VAE Encode 使用的宽高一致**）。
5.  **解码:** 将 `FramePack Sampler (HY)` 输出的 `LATENT` 连接到 VAE 解码器 (`VAE Decode`)，得到最终的视频帧序列。
6.  **组合视频:** 使用 `Video Combine` 或类似节点将解码后的图像帧序列合成为视频文件。

## 注意事项

*   `window_size` 参数对视频连贯性和显存占用有显著影响。较大的窗口会考虑更长的历史信息，可能带来更好的连贯性，但也会增加计算负担和显存需求。请确保 `CreateKeyframes` 和 `Sampler` 节点的 `window_size` 设置一致。
*   `gpu_memory_preservation` 参数用于尝试保留一定的 GPU 显存，防止在采样过程中因显存耗尽而崩溃。如果遇到显存不足的问题，可以尝试适当增大此值，但这可能会降低采样速度。
*   如果使用了 `FramePack BucketResize (HY)`，请确保后续节点（特别是 VAE Encode 和 `FramePack Sampler (HY)`）使用的宽度和高度与 `BucketResize` 节点的输出一致。
*   确保你的 ComfyUI 环境安装了必要的依赖库（如 `diffusers`, `transformers`, `torch` 等）。

## Contact Me
- X (Twitter): [@cychenyue](https://x.com/cychenyue)
- TikTok: [@cychenyue](https://www.tiktok.com/@cychenyue)
- YouTube: [@CY-CHENYUE](https://www.youtube.com/@CY-CHENYUE)
- BiliBili: [@CY-CHENYUE](https://space.bilibili.com/402808950)
- 小红书: [@CY-CHENYUE](https://www.xiaohongshu.com/user/profile/6360e61f000000001f01bda0)