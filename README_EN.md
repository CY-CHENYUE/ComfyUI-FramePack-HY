# ComfyUI-FramePack-HY

This is a custom node package designed for ComfyUI, aimed at efficiently generating long videos using Hunyuan-DiT class models through segmented sampling (FramePack).

https://github.com/user-attachments/assets/01f20678-64fb-4ae0-a9db-c8a15fc09e70
https://github.com/user-attachments/assets/d1d7b687-ae00-44eb-8ed7-02b426b32231
https://github.com/user-attachments/assets/0124aba0-f7a9-445b-bb33-6516c106abf1


## Features

*   **Segmented Sampling (FramePack):** Decomposes the generation process of long videos into multiple overlapping windows (segments). Sampling is performed independently within each segment, effectively reducing VRAM consumption and enabling the generation of longer videos on hardware with limited resources.
*   **Start Frame Definition:** Provides dedicated nodes to set the starting keyframe of the video, ensuring the generation begins as expected.
*   **Memory Optimization:** Integrates memory management strategies to optimize VRAM usage during model loading and sampling.
*   **Based on Hunyuan-DiT:** Primarily adapted for Tencent Hunyuan DiT architecture video models.

## Included Nodes

### 1. Load FramePack Pipeline (HY) (`LoadFramePackDiffusersPipeline_HY`)

*   **Function:** Loads the core video model (currently specifically the Hunyuan Transformer) required for the FramePack workflow. It automatically searches for the model in the `diffusers` folder configured in ComfyUI and handles precision settings.
*   **Inputs:**
    *   `model_path` (STRING): The model name (e.g., `lllyasviel/FramePackI2V_HY`) or the local path to the model. The node automatically searches within ComfyUI's `diffusers` directory.
    *   `precision` (STRING): Model loading precision (`auto`, `fp16`, `bf16`, `fp32`). `auto` attempts to infer from the model name, defaulting to `bf16`.
*   **Outputs:**
    *   `fp_pipeline` (FP_DIFFUSERS_PIPELINE): A Pipeline object containing the loaded model (`transformer`) and data type (`dtype`), intended for use by the sampler.

### 2. FramePack BucketResize (HY) (`FramePackBucketResize_HY`)

*   **Function:** Resizes the input image to the nearest dimensions within predefined resolution buckets. This helps optimize performance for certain models (like SDXL or Hunyuan).
*   **Inputs:**
    *   `image` (IMAGE): The image to be resized.
    *   `base_resolution` (STRING): The base resolution used to select the bucketing strategy (e.g., "640", "1024").
    *   `resize_mode` (STRING, optional): The interpolation method used for resizing (`lanczos`, `bilinear`, `bicubic`, `nearest`).
    *   `alignment` (STRING, optional): The alignment method during resizing (`center`, `top_left`).
*   **Outputs:**
    *   `resized_image` (IMAGE): The image resized to the optimal bucket dimensions.
    *   `width` (INT): The width of the resized image.
    *   `height` (INT): The height of the resized image.

### 3. FramePack Create Keyframes (HY) (`CreateKeyframes_HY`)

*   **Function:** Defines the starting point and basic parameters for video generation.
*   **Inputs:**
    *   `keyframe_1` (LATENT): The required starting latent variable, defining the first frame of the video. Typically comes from a text-to-image or image-to-image node.
    *   `video_length_seconds` (INT): The desired total duration of the generated video (in seconds).
    *   `fps` (INT): The frame rate of the video (frames per second).
    *   `window_size` (INT): The size of the sampling context window. This parameter is crucial as it determines how much historical information the model considers when generating the current frame, affecting temporal coherence. **This parameter must be consistent with the `window_size` of the `FramePack Sampler (HY)` node.**
*   **Outputs:**
    *   `start_latent_out` (LATENT): The processed starting latent variable, containing only the first frame.
    *   `video_length_seconds` (video_length_seconds): Passes the video duration.
    *   `fps` (video_fps): Passes the video frame rate.
    *   `window_size` (window_size): Passes the window size.

### 4. FramePack Sampler (HY) (`FramePackDiffusersSampler_HY`)

*   **Function:** Executes the core segmented video sampling process.
*   **Inputs:**
    *   `fp_pipeline` (FP_DIFFUSERS_PIPELINE): The Pipeline object from the loader node.
    *   `positive` / `negative` (CONDITIONING): Positive and negative text conditions.
    *   `clip_vision` (CLIP_VISION_OUTPUT): CLIP Vision output for image guidance.
    *   `steps` (INT): The number of sampling steps per segment.
    *   `cfg` (FLOAT): Classifier-Free Guidance (CFG) strength.
    *   `guidance_scale` (FLOAT): Distilled guidance strength (for Hunyuan).
    *   `seed` (INT): Random seed.
    *   `width` / `height` (INT): Video width and height.
    *   `gpu_memory_preservation` (FLOAT): GPU VRAM reservation amount (in GB) for memory management.
    *   `sampler` (STRING): Sampler type (currently only `unipc` is supported).
    *   `start_latent_out` (LATENT): The starting latent variable from the `CreateKeyframes_HY` node.
    *   `video_length_seconds` / `video_fps` / `window_size`: Connected from the `CreateKeyframes_HY` node, used to determine total frames, number of segments, etc.
    *   `shift` (FLOAT): Parameter affecting motion intensity.
    *   `use_teacache` (BOOLEAN): Whether to enable teacache acceleration.
    *   `teacache_thresh` (FLOAT): teacache threshold.
    *   `denoise_strength` (FLOAT): Denoising strength (potentially for future image-to-video use).
*   **Outputs:**
    *   `LATENT`: The generated latent sequence containing all video frames.

## Installation

**Method 1: Via ComfyUI Manager (Recommended)**

1.  Open ComfyUI Manager.
2.  Click "Install Custom Nodes".
3.  Search for "FramePack-HY" or "CY-CHENYUE".
4.  Find `ComfyUI-FramePack-HY` and click "Install".
5.  Wait for the installation to complete, then **restart ComfyUI**.

**Method 2: Manual Installation**

1.  **Clone the repository:** Open a terminal or command prompt, navigate to your ComfyUI `custom_nodes` directory, and execute:
    ```bash
    cd /path/to/ComfyUI/custom_nodes
    git clone https://github.com/CY-CHENYUE/ComfyUI-FramePack-HY.git
    ```
    *(Replace `/path/to/ComfyUI/custom_nodes` with your actual path)*

2.  **Install dependencies:** After cloning, change to the newly cloned `ComfyUI-FramePack-HY` directory and execute the appropriate command based on your system and environment:

    *   **Windows:**
        ```bash
        cd ComfyUI-FramePack-HY
        
        # If using the ComfyUI Portable version:
        ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
        
        # If using your own Python environment (venv, conda, etc.):
        # Ensure your environment is activated
        # C:\path\to\your\python.exe -m pip install -r requirements.txt (Replace with your Python path)
        pip install -r requirements.txt # Usually works directly after activating the environment
        ```

    *   **Linux / Mac:**
        ```bash
        cd ComfyUI-FramePack-HY
        
        # If ComfyUI uses its embedded Python (common in portable versions or specific install scripts):
        # ../../python_embeded/bin/python -m pip install -r requirements.txt
        
        # If using system Python or a virtual environment (venv, conda, etc.):
        # Ensure your environment is activated
        python -m pip install -r requirements.txt # Or just: pip install -r requirements.txt
        ```

3.  **Restart ComfyUI:** After installing dependencies, **restart ComfyUI**.

**Troubleshooting:**

*   If you encounter installation issues:
    *   Ensure you have `git` installed.
    *   Try updating `pip`: `path/to/python -m pip install --upgrade pip` (use your Python path).
    *   If you use a proxy, ensure `git` and `pip` are configured correctly.
    *   **Most importantly: Ensure the Python environment used to install dependencies is the same one used to run ComfyUI.**

## Basic Usage Workflow

1.  **Load Model:** Use the `Load FramePack Pipeline (HY)` node to load the required video model (e.g., `lllyasviel/FramePackI2V_HY`), set the model path and precision, and get the `fp_pipeline` output.
2.  **Prepare Start Frame:**
    *   Generate or load an image using a text-to-image node (e.g., `KSampler`) or an image loading node.
    *   **(Optional but recommended)** Connect this image to the `FramePack BucketResize (HY)` node, select an appropriate base resolution, and get the size-optimized `resized_image`.
    *   Connect the original or `resized_image` to a VAE Encoder (`VAE Encode`) to get the starting latent variable.
3.  **Define Video Parameters:**
    *   Connect the latent output from the VAE Encoder to the `keyframe_1` input of `FramePack Create Keyframes (HY)`.
    *   Set the desired video duration, frame rate, and the **crucial `window_size`** on the `FramePack Create Keyframes (HY)` node.
4.  **Sampling:**
    *   Connect the `fp_pipeline` output from `Load FramePack Pipeline (HY)` to the `fp_pipeline` input of `FramePack Sampler (HY)`.
    *   Connect the positive and negative conditions from your text encoder (`positive`, `negative`).
    *   Connect CLIP Vision output if image guidance is needed.
    *   Connect the `start_latent_out`, `video_length_seconds`, `fps`, and `window_size` outputs from `FramePack Create Keyframes (HY)` to the corresponding inputs on `FramePack Sampler (HY)`.
    *   Set sampling parameters (steps, CFG, seed, width/height - **Note: The width and height here should match the output of BucketResize or the dimensions used in VAE Encode**).
5.  **Decode:** Connect the `LATENT` output from `FramePack Sampler (HY)` to a VAE Decoder (`VAE Decode`) to get the final sequence of video frames.
6.  **Combine Video:** Use a `Video Combine` node or similar to combine the decoded image frame sequence into a video file.

## Notes

*   The `window_size` parameter significantly impacts video coherence and VRAM usage. Larger windows consider more historical information, potentially improving coherence but increasing computational load and VRAM requirements. Ensure the `window_size` setting is consistent between the `CreateKeyframes` and `Sampler` nodes.
*   The `gpu_memory_preservation` parameter attempts to reserve a certain amount of GPU VRAM to prevent crashes due to memory exhaustion during sampling. If you encounter out-of-memory issues, try increasing this value, but it might reduce sampling speed.
*   If using `FramePack BucketResize (HY)`, ensure subsequent nodes (especially VAE Encode and `FramePack Sampler (HY)`) use the width and height consistent with the `BucketResize` node's output.
*   Ensure your ComfyUI environment has the necessary dependency libraries installed (e.g., `diffusers`, `transformers`, `torch`, etc., as listed in `requirements.txt`).

## Contact Me
- X (Twitter): [@cychenyue](https://x.com/cychenyue)
- TikTok: [@cychenyue](https://www.tiktok.com/@cychenyue)
- YouTube: [@CY-CHENYUE](https://www.youtube.com/@CY-CHENYUE)
- BiliBili: [@CY-CHENYUE](https://space.bilibili.com/402808950)
- 小红书: [@CY-CHENYUE](https://www.xiaohongshu.com/user/profile/6360e61f000000001f01bda0) 