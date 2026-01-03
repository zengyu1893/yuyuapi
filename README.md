# ComfyUI-yuyuAPI

这是一个集成了多种高质量 AI 模型的 ComfyUI 节点包，目前包含 **Nano Banana 2 (Gemini 3 Pro)** 图像生成节点、**Grok3 Video** 视频生成节点、**yuyu Gemini API** 多模态对话节点以及 **yuyu doubao4.5** 图像生成节点。

## ✨ 主要功能

### 1. Ⓨ Nano Banana 2 (图像生成)
*   **极速生成**：基于 Google Gemini 3 Pro Image Preview 模型。
*   **多模态支持**：支持最多 **15张** 参考图输入 (image_1 ~ image_15)，实现复杂的图像编辑和风格迁移。
*   **高清输出**：支持 1K、2K、4K 分辨率选项。
*   **灵活画幅**：内置多种常用宽高比 (1:1, 16:9, 9:16, 4:3, 3:4 等)。

### 2. yuyu Grok3 Video (视频生成)
*   **前沿模型**：基于 Grok 3 最新视频生成模型。
*   **高清视频**：支持 720P 和 1080P 分辨率。
*   **灵活比例**：支持 1:1, 2:3, 3:2 等常用视频比例。
*   **自动下载**：任务完成后自动下载视频到本地，并输出视频路径。
*   **智能重试**：内置 5 次指数级重试机制，自动应对网络波动和 502/Bad Gateway 错误。
*   **多维输出**：提供视频路径、任务ID、完整响应 JSON 和 视频下载链接。

### 3. yuyu Gemini API (多模态对话)
*   **智能核心**：集成 **Gemini 3 Pro Preview** 和 **Gemini 3 Flash Preview** 模型，具备顶级的多模态理解与推理能力。
*   **全能输入**：
    *   **图像**：支持同时输入 4 张图片进行联合分析。
    *   **视频**：支持视频输入（自动抽帧分析）。
    *   **音频**：支持音频文件输入，实现听觉理解。
*   **思维链控制**：独家 `strip_thought` 开关，可自动屏蔽模型的内部思考过程（Thinking Process），仅输出最终结果，让回复更加清爽直接。
*   **精细控制**：支持系统指令 (System Instruction)、Temperature、Top P、Max Tokens 等参数调节，满足专业级提示工程需求。

### 4. yuyu doubao4.5 (豆包生图)
*   **旗舰模型**：集成火山引擎最新 **Doubao Seedream 4.5** 模型 (doubao-seedream-4-5-251128)。
*   **文生图/图生图**：支持基础文本生成及多达 **14张** 参考图的多图编辑/风格融合。
*   **组图模式**：支持 `Group Mode` (自动组图)，可一次生成多张连贯插画，并实时流式返回预览。
*   **精准画幅**：支持 1K/2K/4K 及多种比例 (1:1, 16:9, 9:21 等)，自动计算符合官方规范的像素尺寸。
*   **双源切换**：支持切换官方 API 源 (`official`) 或 Yuli 代理源 (`yuli`)。

## 📦 安装说明

1.  进入 ComfyUI 的 `custom_nodes` 目录。
2.  克隆本项目：
    ```bash
    git clone https://github.com/yuyu/ComfyUI-yuyuAPI.git
    ```
3.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ 配置 API Key

本项目支持多种方式配置 API Key（优先级从高到低）：

1.  **节点组件 (Widget)**：直接在节点界面的 `api_key` 输入框填写。
2.  **环境变量**：设置系统环境变量 `YUYU_API_KEY`。
3.  **配置文件**：
    *   复制 `config.json.example` 为 `config.json`。
    *   在 `config.json` 中填入你的 Key：
        ```json
        {
            "YUYU_API_KEY": "sk-xxxxxxxx"
        }
        ```

## 🛠️ 使用方法

### Nano Banana 2
1.  右键点击 -> **玉玉API** -> **Nano Banana** -> **Ⓨ Nano Banana 2 (Yuyu)**。
2.  输入提示词，选择参数即可生成。

### Grok3 Video
1.  右键点击 -> **玉玉API** -> **grok** -> **yuyu Grok3 Video**。
2.  配置提示词、分辨率等参数，连接参考图（可选），即可生成视频。

### Gemini API
1.  右键点击 -> **玉玉API** -> **gemini** -> **yuyu Gemini API**。
2.  **输入**：
    *   `system_instruction`: 设定 AI 的角色和任务目标。
    *   `user_prompt`: 用户的具体问题或指令。
    *   `model`: 选择 Gemini 3 Pro 或 Flash 模型。
    *   `strip_thought`: 开启后将隐藏模型的思考过程，只看结果。
    *   `image/video/audio` (可选): 连接多模态输入。
3.  **输出**：
    *   `response`: AI 的最终回复文本。

### Doubao
1.  右键点击 -> **玉玉API** -> **豆包** -> **yuyu doubao4.5**。
2.  **输入**：
    *   `prompt`: 描述画面内容。
    *   `aspect_ratio`/`resolution`: 设置画幅和清晰度。
    *   `group_mode`: 开启 "auto" 可生成组图 (需配合 Max Images)。
    *   `image_1`...: 连接参考图进行风格迁移或编辑。

## ⚠️ 注意事项

*   请确保你的 API Key 有效且支持对应的模型服务。
*   本项目默认使用 `yuli.host` 提供的 API 服务，支持自动直连。
*   Grok 视频生成需要一定时间，请耐心等待节点运行完成（后台会有进度日志）。

## License

MIT
