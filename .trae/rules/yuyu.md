# yuyu API 节点功能说明

## 1. Nano Banana 2 (Gemini 生图/编辑)
- **智能模式**：
  - **不连图片** = **文生图** (Text-to-Image)
  - **连接图片** = **图片编辑** (Image Editing / Style Transfer)
- **模型选择**：
  - `gemini-3-pro-image-preview`：画质最强，逻辑理解深，但速度稍慢。
  - `gemini-3.1-flash-image-preview`：生成速度极快，适合快速验证。
- **尺寸控制**：
  - 严格按照选择的 `aspect_ratio` (比例) 和 `resolution` (1K/2K/4K) 输出。
  - 自动处理模型返回尺寸不一致的问题（自动居中裁剪+缩放）。
- **参数说明**：
  - `max_ref_images`：限制参考图上传数量（默认15），调小可显著提升速度。

## 2. Grok3 Video
- 支持 720P/1080P 视频生成。
- 自动重试机制应对网络波动。

## 3. Gemini API
- 多模态对话，支持图/文/视频/音频输入。
- `strip_thought` 可隐藏思维链。

## 4. Doubao 4.5
- 支持文生图与图生图。
- 组图模式可一次生成多张。

## 5. Veo Video
- **模型**：`veo_3_1-fast` (极速生成) / `veo-2.0-generate-video-preview`
- **功能**：
  - **文生视频**：直接输入 Prompt。
  - **图生视频**：连接 Image 输入作为参考图。
- **特性**：
  - 智能下载：自动解析重定向、JSON 包装的下载地址。
  - 自动重试：智能切换 Auth 头策略，兼容 S3/OSS 等无需鉴权的存储链接。
  - 实时预览：输出 `preview_image` (视频首帧/预览帧)，方便在 ComfyUI 中直接查看结果。
