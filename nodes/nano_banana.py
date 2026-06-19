"""
Ⓨ Nano Banana 2 — Gemini 文生图 / 多图编辑节点。
"""

import base64
import io
import time
import numpy as np
import torch
from PIL import Image

from .core import (
    session,
    resolve_api_key,
    tensor_to_base64,
    target_size,
    fit_to_size,
    extract_image_from_response,
    request_with_progress,
)


class YuyuNanoBananaNode:
    def __init__(self):
        self._instance_id = ""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    ["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"],
                    {"default": "gemini-3-pro-image-preview"},
                ),
                "aspect_ratio": (
                    ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                ),
                "resolution": (["1K", "2K", "4K"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "max_ref_images": ("INT", {"default": 15, "min": 0, "max": 15}),
            },
        }
        for i in range(1, 16):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "玉玉API/Nano Banana"

    def generate(self, prompt, model, aspect_ratio, resolution, seed, **kwargs):
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        api_key = resolve_api_key(kwargs.get("api_key", ""))

        max_ref_images = int(kwargs.get("max_ref_images", 15) or 0)
        input_images: list[str] = []
        for i in range(1, 16):
            if max_ref_images >= 0 and len(input_images) >= max_ref_images:
                break
            key = f"image_{i}"
            if kwargs.get(key) is not None:
                img_tensor = kwargs[key][0]
                i_img = 255.0 * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
                max_side = 1568
                if img.width > max_side or img.height > max_side:
                    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=90)
                b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                input_images.append(b64)

        is_edit = len(input_images) > 0
        url = f"https://yuli.host/v1beta/models/{model}:generateContent"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload: dict = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt.strip()}]}
            ],
            "generationConfig": {
                "candidateCount": 1,
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution,
                },
            },
        }

        if input_images:
            for b64 in input_images:
                payload["contents"][0]["parts"].append(
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
                )

        if seed and int(seed) != 0:
            payload["generationConfig"]["seed"] = abs(int(seed)) % 2147483647

        req_started = time.time()
        print(
            f"【yuyu】NanoBanana 请求: mode={'edit' if is_edit else 'generate'} "
            f"model={model} ratio={aspect_ratio} res={resolution} refs={len(input_images)}/{max_ref_images}"
        )

        # 直连
        try:
            response = request_with_progress(
                "POST",
                url,
                timeout=500,
                log_interval=10,
                params={"key": api_key},
                headers=headers,
                json=payload,
                verify=False,
                proxies={"http": None, "https": None},
            )
        except Exception as e:
            print(f"【yuyu】直连请求异常: {e}，尝试使用系统默认代理重试...")
            try:
                response = request_with_progress(
                    "POST",
                    url,
                    timeout=500,
                    log_interval=10,
                    params={"key": api_key},
                    headers=headers,
                    json=payload,
                    verify=False,
                )
            except Exception as final_e:
                raise Exception(f"请求最终失败: {final_e}")

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        res_json = response.json()
        elapsed = time.time() - req_started
        cand_count = len(res_json.get("candidates") or []) if isinstance(res_json, dict) else 0
        print(f"【yuyu】NanoBanana 响应: candidates={cand_count} cost_s={elapsed:.2f}")

        img_out = extract_image_from_response(res_json)
        if img_out is None:
            raise Exception("No image returned")

        target_w, target_h = target_size(aspect_ratio, resolution)
        got_w, got_h = img_out.size
        target_ratio = target_w / target_h
        got_ratio = got_w / got_h

        if abs(target_ratio - got_ratio) > 0.1:
            img_out = fit_to_size(img_out, target_w, target_h)

        arr = np.array(img_out).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]
        return (tensor,)
