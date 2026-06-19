# ComfyUI-yuyuAPI

集成 GPT Image、Gemini、Doubao、Grok、Veo 多模型生图/对话/视频的 ComfyUI 自定义节点包。

API 服务由 [玉玉API](https://yuli.host) 提供，支持直连。

## 节点

| 节点 | 模型 | 功能 |
|------|------|------|
| **Nano Banana** | Gemini 3 Pro / 3.1 Flash | 文生图 + 图生图 |
| **yuyu Gemini LLM** | gemini-3.1-pro / 3-pro / 3-flash / 自定义 | 多模态对话 |
| **yuyu GPT Image 2** | gpt-image-2 / gpt-image-2-all | 文生图 + 图生图 |
| **yuyu doubao4.5** | doubao-seedream-4-5 | 文生图 + 图生图 + 组图 |
| **yuyu Grok3 Video** | Grok 3 | 视频生成 |
| **yuyu Veo Video** | veo-3.1-fast / veo-2.0 | 文/图生视频 |

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yuyu/ComfyUI-yuyuAPI.git
pip install -r requirements.txt
```

## 配置 API Key

支持三种方式，优先级从高到低：

**方式一：节点填写**

每个节点都有 `api_key` 输入框，直接粘贴即可，优先使用。

**方式二：文件夹 config.json**

复制 `config.json.example` 为 `config.json`，填入 Key，整个插件通用：

```json
{
    "YUYU_API_KEY": "sk-xxxxxxxx"
}
```

**方式三：环境变量**

```bash
set YUYU_API_KEY=sk-xxxxxxxx
```

节点不填 `api_key` 时，自动按 config.json → 环境变量 顺序查找。

## 节点说明

### Nano Banana

- 不连参考图 → 文生图，连接参考图（最多 15 张）→ 图生图
- 模型：Gemini 3 Pro（画质）/ 3.1 Flash（速度）
- 支持 1K / 2K / 4K，10 种宽高比

### yuyu Gemini LLM

- 多模态对话：图片（最多 4 张）+ 视频 + 音频
- `custom_model`：自由填写任意 Gemini 模型名
- `strip_thought`：隐藏模型思考过程，只输出最终结果

### yuyu GPT Image 2

- 比例选择：auto / 1:1 / 4:3 / 3:4 / 3:2 / 2:3 / 16:9 / 9:16
- 分辨率：1K / 2K / 4K
- 图生图：连接参考图（14 槽位）或 prompt 中含图片链接均可触发
- 输出格式：png / jpeg / webp

### yuyu doubao4.5

- 豆包 Seedream 4.5，14 张参考图
- 组图模式 + 官方/代理双源切换

### yuyu Grok3 Video

- 720P / 1080P，多种宽高比，自动下载 + 重试

### yuyu Veo Video

- veo-3.1-fast / veo-2.0，文/图生视频，输出预览帧

## License

MIT
