"""
共享工具模块 — HTTP 会话、配置加载、图像处理辅助函数。
"""

import os
import json
import time
import base64
import io
import hashlib
from typing import Any

import numpy as np
import torch
from PIL import Image
import folder_paths
import requests
import urllib3

# 屏蔽 SSL 证书警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 复用 TCP 连接的全局 Session
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount("http://", adapter)
session.mount("https://", adapter)


def load_config() -> dict[str, Any]:
    """加载 config.json 配置（若存在）。"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def resolve_api_key(widget_key: str = "", env_var: str = "YUYU_API_KEY") -> str:
    """按优先级解析 API Key：Widget > 环境变量 > config.json。"""
    config = load_config()
    key = widget_key or ""
    if not key:
        key = os.environ.get(env_var, "")
    if not key:
        key = config.get(env_var, "")
    if not key:
        raise ValueError("API Key 未配置，请在 Widget、环境变量或 config.json 中设置。")
    return key.strip()


def tensor_to_pil(image_tensor: torch.Tensor, index: int = 0) -> Image.Image:
    """将 ComfyUI 的 image tensor 转为 PIL Image。"""
    i = 255.0 * image_tensor[index].cpu().numpy()
    return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))


def tensor_to_base64(image_tensor: torch.Tensor, fmt: str = "JPEG", quality: int = 90) -> str:
    """将 ComfyUI image tensor 转为 base64 字符串（不含 data: 前缀）。"""
    img = tensor_to_pil(image_tensor)
    buf = io.BytesIO()
    save_fmt = "JPEG" if fmt.upper() == "JPEG" else "PNG"
    if save_fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=save_fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensor_to_data_url(image_tensor: torch.Tensor) -> str:
    """将 ComfyUI image tensor 转为 data: URL 字符串。"""
    b64 = tensor_to_base64(image_tensor, fmt="PNG")
    return f"data:image/png;base64,{b64}"


# ——— 分辨率 & 尺寸表 ——————————————————————————————————————————————

# NanoBanana / GeminiEdit 共用的 2K 基准尺寸
SIZE_TABLE_2K: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "2:3": (1664, 2496),
    "3:2": (2496, 1664),
    "3:4": (1728, 2304),
    "4:3": (2304, 1728),
    "4:5": (1792, 2240),
    "5:4": (2240, 1792),
    "9:16": (1440, 2560),
    "16:9": (2560, 1440),
    "21:9": (2688, 1152),
}

# Doubao 使用的尺寸表
SIZE_TABLE_DOUBAO: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2560, 1440),
    "9:16": (1440, 2560),
    "4:3": (2304, 1728),
    "3:4": (1728, 2304),
    "3:2": (2496, 1664),
    "2:3": (1664, 2496),
    "21:9": (3024, 1296),
    "9:21": (1296, 3024),
}


def target_size(aspect_ratio: str, resolution: str, table: dict | None = None) -> tuple[int, int]:
    """根据比例和分辨率标签计算目标像素尺寸。"""
    if table is None:
        table = SIZE_TABLE_2K
    w, h = table.get(aspect_ratio, (2048, 2048))
    if resolution == "1K":
        w, h = max(1, w // 2), max(1, h // 2)
    elif resolution == "4K":
        w, h = w * 2, h * 2
    return int(w), int(h)


def fit_to_size(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """居中裁剪 + 缩放，使图像严格匹配目标尺寸。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 若尺寸接近（误差 < 64px），不裁剪
    if abs(src_w - target_w) < 64 and abs(src_h - target_h) < 64:
        return img

    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    return img.crop((left, top, left + target_w, top + target_h))


def extract_image_from_response(res_json: dict) -> Image.Image | None:
    """从 Gemini API 响应中提取第一张图片（base64 或 URL）。"""
    candidates = res_json.get("candidates") if isinstance(res_json, dict) else None
    if not isinstance(candidates, list) or len(candidates) == 0:
        return None

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    for part in parts:
        b64_data = None
        if isinstance(part, dict):
            if "inline_data" in part:
                b64_data = (part.get("inline_data") or {}).get("data")
            elif "inlineData" in part:
                b64_data = (part.get("inlineData") or {}).get("data")

        if b64_data:
            return Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")

        if isinstance(part, dict) and "text" in part:
            text = part["text"]
            if isinstance(text, str) and "http" in text:
                image_url = text.strip()
                if image_url.startswith("data:"):
                    try:
                        _, encoded = image_url.split(",", 1)
                        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                    except Exception:
                        pass
                img_res = session.get(image_url, timeout=300, verify=False)
                if img_res.status_code != 200:
                    img_res = session.get(
                        image_url,
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=300,
                        verify=False,
                    )
                img_res.raise_for_status()
                return Image.open(io.BytesIO(img_res.content)).convert("RGB")
    return None


def request_with_progress(method: str, url: str, timeout: int = 500, log_interval: int = 10, **kwargs) -> requests.Response:
    """带进度日志的阻塞请求包装器——用后台线程定时打印等待时间。"""
    import threading

    stop_event = threading.Event()
    start_time = time.time()

    def log_worker() -> None:
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            if elapsed > timeout:
                break
            if elapsed > 1:
                print(f"【yuyu】 已等待 {int(elapsed)}s ...")
            time.sleep(log_interval)

    t = threading.Thread(target=log_worker, daemon=True)
    t.start()
    try:
        return session.request(method, url, timeout=timeout, **kwargs)
    finally:
        stop_event.set()
        t.join(timeout=1)


def make_instance_id(obj: object) -> str:
    """为节点实例生成短的唯一标识。"""
    return hashlib.sha256(str(id(obj)).encode()).hexdigest()[:8]
