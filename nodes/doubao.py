"""
yuyu doubao4.5 — 豆包 Seedream 4.5 图像生成节点。
"""

import base64
import io
import json
import time

import numpy as np
import torch
from PIL import Image

from .core import session, resolve_api_key, tensor_to_data_url, SIZE_TABLE_DOUBAO, make_instance_id


class YuyuDoubaoNode:
    def __init__(self):
        self._instance_id = make_instance_id(self)

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "api_source": (["official", "yuli"], {"default": "official"}),
                "model": ("STRING", {"default": "doubao-seedream-4-5-251128"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"],
                    {"default": "1:1"},
                ),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
                "group_mode": (["disable", "auto"], {"default": "disable"}),
                "max_images": ("INT", {"default": 15, "min": 1, "max": 15}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "stream": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 600}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }
        for i in range(1, 15):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "玉玉API/豆包"

    def _get_resolution_size(self, aspect_ratio: str, resolution_tag: str) -> str:
        w, h = SIZE_TABLE_DOUBAO.get(aspect_ratio, (2048, 2048))
        if resolution_tag == "1K":
            w, h = w // 2, h // 2
        elif resolution_tag == "4K":
            w, h = w * 2, h * 2
        return f"{w}x{h}"

    def generate(
        self,
        api_source,
        model,
        prompt,
        aspect_ratio,
        resolution,
        group_mode,
        max_images,
        seed,
        stream,
        watermark,
        timeout,
        api_key=None,
        **kwargs,
    ):
        api_key = resolve_api_key(api_key or "")

        input_images = []
        for i in range(1, 15):
            key = f"image_{i}"
            if kwargs.get(key) is not None:
                img_b64 = tensor_to_data_url(kwargs[key][0])
                input_images.append(img_b64)

        size_str = self._get_resolution_size(aspect_ratio, resolution)

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "size": size_str,
            "watermark": watermark,
            "stream": stream,
        }

        if seed != -1:
            payload["seed"] = abs(seed) % 2147483647

        if input_images:
            if len(input_images) == 1:
                payload["image"] = input_images[0]
            else:
                payload["image"] = input_images

        if group_mode == "auto":
            payload["sequential_image_generation"] = "auto"
            payload["sequential_image_generation_options"] = {"max_images": max_images}
            payload["stream"] = True
            payload["response_format"] = "b64_json"
        elif stream:
            payload["response_format"] = "b64_json"

        if api_source == "official":
            submit_url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        else:
            submit_url = "https://yuli.host/v1/images/generations"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        print(f"【yuyu】[{self._instance_id}] Doubao Request: {submit_url}")
        print(f"【yuyu】Params: Model={model}, Size={size_str}, Images={len(input_images)}")

        try:
            is_streaming = payload.get("stream", False)
            response = session.post(
                submit_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=is_streaming,
                verify=False,
            )

            if response.status_code != 200:
                err_text = ""
                try:
                    err_text = response.text
                except Exception:
                    pass
                raise Exception(f"API Error: {response.status_code} - {err_text}")

            image_tensors: list[torch.Tensor] = []

            if is_streaming:
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            b64_data = data_json.get("b64_json")
                            if b64_data:
                                img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                                img = img.convert("RGB")
                                arr = np.array(img).astype(np.float32) / 255.0
                                image_tensors.append(torch.from_numpy(arr)[None,])
                        except Exception as e:
                            print(f"【yuyu】Stream parsing error: {e}")
            else:
                res_json = response.json()
                if "data" in res_json and isinstance(res_json["data"], list):
                    for item in res_json["data"]:
                        img_url = item.get("url")
                        b64_data = item.get("b64_json") or item.get("binary_data")

                        img = None
                        if img_url:
                            print(f"【yuyu】Downloading: {img_url}")
                            img_res = session.get(img_url, timeout=120, verify=False)
                            img = Image.open(io.BytesIO(img_res.content))
                        elif b64_data:
                            img = Image.open(io.BytesIO(base64.b64decode(b64_data)))

                        if img:
                            img = img.convert("RGB")
                            arr = np.array(img).astype(np.float32) / 255.0
                            image_tensors.append(torch.from_numpy(arr)[None,])

            if not image_tensors:
                if is_streaming:
                    raise Exception("Stream finished but no images collected.")
                else:
                    raise Exception(f"No images returned. Response: {res_json}")

            if len(image_tensors) > 1:
                try:
                    return (torch.cat(image_tensors, dim=0),)
                except Exception:
                    first_shape = image_tensors[0].shape
                    for t in image_tensors[1:]:
                        if t.shape != first_shape:
                            print("【yuyu】Warning: Image sizes mismatch in batch. Returning first image only.")
                            return (image_tensors[0],)
                    return (torch.cat(image_tensors, dim=0),)
            else:
                return (image_tensors[0],)

        except Exception as e:
            print(f"【yuyu】Error: {e}")
            raise
