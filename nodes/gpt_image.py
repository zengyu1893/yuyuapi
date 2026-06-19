"""
yuyu GPT Image 2 — gpt-image-2 文生图 / gpt-image-2-all 多图融合节点。
"""

import io
import json
import re
import time

import numpy as np
import torch
from PIL import Image

from .core import (
    session,
    resolve_api_key,
    tensor_to_data_url,
    make_instance_id,
)

# 比例 × 分辨率 → 像素尺寸（gpt-image-2：3840 单边上限、16倍数）
ASPECT_SIZE_MAP = {
    "1:1":  {"1K": "1024x1024",  "2K": "2048x2048",  "4K": "2880x2880"},
    "4:3":  {"1K": "1024x768",   "2K": "2048x1536",  "4K": "3264x2448"},
    "3:4":  {"1K": "768x1024",   "2K": "1536x2048",  "4K": "2448x3264"},
    "3:2":  {"1K": "1536x1024",  "2K": "3072x2048",  "4K": "3504x2336"},
    "2:3":  {"1K": "1024x1536",  "2K": "2048x3072",  "4K": "2336x3504"},
    "16:9": {"1K": "1792x1024",  "2K": "3584x2048",  "4K": "3840x2160"},
    "9:16": {"1K": "1024x1792",  "2K": "2048x3584",  "4K": "2160x3840"},
}

ASPECT_RATIO_LIST = ["auto", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]
RESOLUTION_LIST = ["1K", "2K", "4K"]
QUALITY_LIST = ["auto", "low", "medium", "high"]
FORMAT_LIST = ["png", "jpeg", "webp"]


class YuyuGPTImageNode:
    def __init__(self):
        self._instance_id = make_instance_id(self)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": (
                    ["gpt-image-2", "gpt-image-2-all"],
                    {"default": "gpt-image-2"},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (ASPECT_RATIO_LIST, {"default": "auto"}),
                "resolution": (RESOLUTION_LIST, {"default": "2K"}),
                "quality": (QUALITY_LIST, {"default": "auto"}),
                "format": (FORMAT_LIST, {"default": "png"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }
        # gpt-image-2-all 支持最多 14 张参考图
        for i in range(1, 15):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "玉玉API/GPT Image"

    # ——— 响应解析 ——————————————————————————————————————————————

    def _extract_url_from_content(self, content: str) -> str | None:
        """从 chat completion 的 content 中提取图片 URL。"""
        if not content:
            return None
        # Markdown 图片语法 ![alt](url)
        md_match = re.search(r"!\[.*?\]\(([^)]+)\)", content)
        if md_match:
            return md_match[1]
        # 直接的 http(s) URL
        url_match = re.search(r"https?://\S+", content)
        if url_match:
            return url_match[0].rstrip(".,;:'\"")
        return None

    def _download_image(self, url: str) -> Image.Image:
        """从 URL 下载图片，返回 PIL Image。"""
        resp = session.get(url, timeout=120, verify=False)
        if resp.status_code != 200:
            resp = session.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=120,
                verify=False,
            )
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def _parse_chat_response(self, res_json: dict) -> list[Image.Image]:
        """解析 chat completion 格式的响应。"""
        images: list[Image.Image] = []
        choices = res_json.get("choices", [])
        for choice in choices:
            content = (choice.get("message", {}) or {}).get("content", "")
            if not content:
                continue
            # 尝试 base64
            if "data:image" in content or content.startswith("/9j/") or content.startswith("iVBOR"):
                import base64
                parts = content.split("base64,", 1)
                b64_data = parts[1] if len(parts) > 1 else content
                img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                images.append(img)
                continue
            # 尝试 URL
            url = self._extract_url_from_content(content)
            if url:
                images.append(self._download_image(url))
                continue
        return images

    def _parse_standard_response(self, res_json: dict) -> list[Image.Image]:
        """解析标准 images API 格式的响应。"""
        images: list[Image.Image] = []
        data = res_json.get("data", [])
        if not data:
            return images
        for item in data:
            url = item.get("url", "")
            b64_data = item.get("b64_json", "")
            if b64_data:
                import base64
                img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                images.append(img)
                continue
            if url:
                images.append(self._download_image(url))
                continue
        return images

    # ——— 尺寸计算 ——————————————————————————————————————————————

    def _get_size(self, aspect_ratio: str, resolution: str) -> str:
        """比例 + 分辨率 → 像素尺寸。auto 时返回 auto 让 API 自动决定。"""
        if aspect_ratio == "auto" or not aspect_ratio:
            return "auto"
        size = ASPECT_SIZE_MAP.get(aspect_ratio, {}).get(resolution, "")
        if size:
            return size
        # fallback: 2K 1:1
        return ASPECT_SIZE_MAP.get("1:1", {}).get("2K", "2048x2048")

    def _extract_urls_from_prompt(self, prompt: str) -> list[str]:
        """从 prompt 文本中提取所有图片 URL（http/https 或 Markdown 语法）。"""
        urls: list[str] = []
        if not prompt:
            return urls
        # Markdown 图片语法 ![alt](url)
        md_urls = re.findall(r"!\[.*?\]\(([^)]+)\)", prompt)
        for u in md_urls:
            if u.startswith("http://") or u.startswith("https://"):
                urls.append(u)
        # 直接的 http(s) URL（排除 Markdown/HTML 闭合字符）
        raw_urls = re.findall(r"https?://[^\s,;:'\"<>()\[\]]+", prompt)
        for u in raw_urls:
            # 去掉末尾标点/括号，去重
            u = u.rstrip(".,;:'\"()[]")
            if u not in urls:
                urls.append(u)
        return urls

    # ——— 主逻辑 ————————————————————————————————————————————————

    def generate(self, model, prompt, aspect_ratio, resolution, quality, format, n, seed, api_key=None, **kwargs):
        """文生图 / 多图融合。"""
        size = self._get_size(aspect_ratio, resolution)
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt 不能为空")

        api_key = resolve_api_key(api_key or "")

        # 收集参考图
        input_images: list[str] = []
        for i in range(1, 15):
            key = f"image_{i}"
            img_tensor = kwargs.get(key)
            if img_tensor is not None:
                data_url = tensor_to_data_url(img_tensor)
                input_images.append(data_url)

        # 没有参考图输入但有图片链接 → 从 prompt 提取 URL 作为参考图
        url_refs: list[str] = []
        if not input_images:
            url_refs = self._extract_urls_from_prompt(prompt)

        is_edit = len(input_images) > 0 or len(url_refs) > 0

        url = "https://yuli.host/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "format": format,
            "n": n,
        }

        if seed and int(seed) != 0:
            payload["seed"] = abs(int(seed)) % 2147483647

        if is_edit:
            all_refs = input_images + url_refs
            if all_refs:
                # gpt-image-2 的 image 参数接受 URL / data: URL 数组
                payload["image"] = all_refs

        print(
            f"【yuyu】[{self._instance_id}] GPT Image 请求: "
            f"model={model} aspect_ratio={aspect_ratio} resolution={resolution} "
            f"size={size} quality={quality} fmt={format} "
            f"n={n} refs={len(input_images)} url_refs={len(url_refs)}"
        )

        try:
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=300,
                verify=False,
                proxies={"http": None, "https": None},
            )
        except Exception as e:
            print(f"【yuyu】[{self._instance_id}] 直连异常: {e}，尝试默认代理重试...")
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=300,
                verify=False,
            )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        res_json = response.json()

        # 智能判断响应格式
        images: list[Image.Image] = []
        if "choices" in res_json:
            # chat completion 格式 (gpt-image-2)
            images = self._parse_chat_response(res_json)
        elif "data" in res_json:
            # 标准 images API 格式 (gpt-image-2-all)
            images = self._parse_standard_response(res_json)
        else:
            raise Exception(f"无法解析响应格式: {json.dumps(res_json, ensure_ascii=False)[:500]}")

        if not images:
            raise Exception(f"未返回任何图片。响应: {json.dumps(res_json, ensure_ascii=False)[:300]}")

        # 转为 ComfyUI tensor 输出
        tensors: list[torch.Tensor] = []
        for img in images:
            arr = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr)[None,])

        if len(tensors) > 1:
            try:
                return (torch.cat(tensors, dim=0),)
            except Exception:
                first_shape = tensors[0].shape
                for t in tensors[1:]:
                    if t.shape != first_shape:
                        print("【yuyu】Warning: Image sizes mismatch in batch. Returning first image only.")
                        return (tensors[0],)
                return (torch.cat(tensors, dim=0),)
        else:
            return (tensors[0],)
