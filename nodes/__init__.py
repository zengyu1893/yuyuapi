"""
ComfyUI-yuyuAPI 节点集合。
"""

from .nano_banana import YuyuNanoBananaNode
from .grok_video import YuyuGrok3VideoNode
from .gemini_chat import YuyuGeminiNode
from .gpt_image import YuyuGPTImageNode
from .doubao import YuyuDoubaoNode
from .veo import YuyuVeoNode

NODE_CLASS_MAPPINGS = {
    "YuyuNanoBananaNode": YuyuNanoBananaNode,
    "YuyuGrok3VideoNode": YuyuGrok3VideoNode,
    "YuyuGeminiNode": YuyuGeminiNode,
    "YuyuGPTImageNode": YuyuGPTImageNode,
    "YuyuDoubaoNode": YuyuDoubaoNode,
    "YuyuVeoNode": YuyuVeoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuyuNanoBananaNode": "Nano Banana",
    "YuyuGrok3VideoNode": "yuyu Grok3 Video",
    "YuyuGeminiNode": "yuyu Gemini LLM",
    "YuyuGPTImageNode": "yuyu GPT Image 2",
    "YuyuDoubaoNode": "yuyu doubao4.5",
    "YuyuVeoNode": "yuyu Veo Video",
}
