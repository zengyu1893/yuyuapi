import os
import json
import time
import requests
import base64
import io
import hashlib
import torch
import numpy as np
import wave
from PIL import Image
import folder_paths
import urllib3

# 【优化】屏蔽 SSL 证书警告，还你一个清爽的控制台
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 【优化】使用 Session 复用 TCP 连接，提升连续请求速度
session = requests.Session()
# 设置重试机制
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount('http://', adapter)
session.mount('https://', adapter)

# 简单的配置加载
def load_config():
    config = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except:
            pass
    return config

class YuyuNanoBananaNode:
    def __init__(self):
        self.config = load_config()
        self._size_overrides = {}

    def _target_size(self, aspect_ratio: str, resolution: str):
        base_2k = {
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
        w, h = base_2k.get(aspect_ratio, (2048, 2048))
        if resolution == "1K":
            w, h = max(1, w // 2), max(1, h // 2)
        elif resolution == "4K":
            w, h = w * 2, h * 2
        return int(w), int(h)

    def _fit_to_size(self, img: Image.Image, target_w: int, target_h: int):
        if img.mode != "RGB":
            img = img.convert("RGB")
        src_w, src_h = img.size
        # 允许一定范围的误差（比如 +/- 64px），如果在这个范围内就不裁剪，直接 resize 或原样返回
        if abs(src_w - target_w) < 64 and abs(src_h - target_h) < 64:
            print(f"【yuyu】NanoBanana 尺寸接近目标 ({src_w}x{src_h} vs {target_w}x{target_h})，不进行裁剪")
            return img
            
        # 如果需要强制尺寸匹配
        if src_w <= 0 or src_h <= 0:
            return img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        scale = max(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        right = left + target_w
        bottom = top + target_h
        return img.crop((left, top, right, bottom))

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "max_ref_images": ("INT", {"default": 15, "min": 0, "max": 15}),
            }
        }
        for i in range(1, 16):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "玉玉API/Nano Banana"

    def _request_with_timeout_log(self, method, url, timeout=500, log_interval=10, **kwargs):
        """
        带日志的请求包装器：每隔 log_interval 秒打印一次等待日志，直到请求完成或超时
        注意：requests 库本身是阻塞的，无法在单线程中同时等待和打印日志。
        为了实现“每10秒显示一次”，我们需要用一个单独的线程来打印，或者直接让 requests 阻塞等待。
        但在 Python requests 中，read timeout 是指“等待服务器返回数据的时间”，一旦开始传输就不会超时。
        对于 Gemini 这种长生成任务，服务器可能几百秒不返回任何数据，这会导致 requests一直阻塞。
        
        真正的“每10秒打印”需要异步 IO 或多线程。这里为了兼容性和简单性，我们采用多线程打印进度。
        """
        import threading
        
        stop_event = threading.Event()
        start_time = time.time()
        
        def log_worker():
            while not stop_event.is_set():
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    break
                if elapsed > 1: # 刚开始不打印
                    print(f"【yuyu】NanoBanana 已等待 {int(elapsed)}s ...")
                time.sleep(log_interval)
        
        t = threading.Thread(target=log_worker, daemon=True)
        t.start()
        
        try:
            # 执行真正的请求
            response = session.request(method, url, timeout=timeout, **kwargs)
            return response
        finally:
            stop_event.set()
            t.join(timeout=1)

    def generate(self, prompt, model, aspect_ratio, resolution, seed, **kwargs):
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        api_key = kwargs.get("api_key", "")
        if not api_key:
            api_key = os.environ.get("YUYU_API_KEY", "")
        if not api_key:
            api_key = self.config.get("YUYU_API_KEY", "")
        if not api_key:
            raise ValueError("API Key is missing")

        max_ref_images = int(kwargs.get("max_ref_images", 15) or 0)
        input_images = []
        for i in range(1, 16):
            if max_ref_images >= 0 and len(input_images) >= max_ref_images:
                break
            key = f"image_{i}"
            if kwargs.get(key) is not None:
                img_tensor = kwargs[key][0]
                i_img = 255. * img_tensor.cpu().numpy()
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

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt.strip()}
                    ]
                }
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
        print(f"【yuyu】NanoBanana 请求: mode={'edit' if is_edit else 'generate'} model={model} ratio={aspect_ratio} res={resolution} refs={len(input_images)}/{max_ref_images}")
        
        try:
            # 使用带日志的请求包装器
            response = self._request_with_timeout_log(
                "POST",
                url,
                timeout=500, # 500秒超时
                log_interval=10, # 每10秒打印一次
                params={"key": api_key},
                headers=headers,
                json=payload,
                verify=False,
                proxies={"http": None, "https": None},
            )
        except Exception as e:
            print(f"【yuyu】直连请求异常: {e}，尝试使用系统默认代理重试...")
            try:
                response = self._request_with_timeout_log(
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
        
        try:
            img_out = None
            candidates = res_json.get("candidates") if isinstance(res_json, dict) else None
            if isinstance(candidates, list) and len(candidates) > 0:
                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []
                for part in parts:
                    b64_data = None
                    if isinstance(part, dict) and "inline_data" in part:
                        inline = part.get("inline_data") or {}
                        b64_data = inline.get("data")
                    elif isinstance(part, dict) and "inlineData" in part:
                        inline = part.get("inlineData") or {}
                        b64_data = inline.get("data")

                    if b64_data:
                        print(f"【yuyu】NanoBanana 返回: base64_chars={len(b64_data)}")
                        img_out = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                        break

                    if isinstance(part, dict) and "text" in part and isinstance(part["text"], str) and "http" in part["text"]:
                        image_url = part["text"].strip()
                        print(f"【yuyu】NanoBanana 图片链接: {image_url}")
                        
                        # 64位转换：如果链接是Base64（虽然text里很少见，但以防万一）
                        if image_url.startswith("data:"):
                             try:
                                 header, encoded = image_url.split(",", 1)
                                 img_out = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                             except:
                                 pass
                        
                        if img_out is None:
                            img_res = session.get(image_url, timeout=300, verify=False)
                            if img_res.status_code != 200:
                                img_res = session.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=300, verify=False)
                            img_res.raise_for_status()
                            img_out = Image.open(io.BytesIO(img_res.content)).convert("RGB")
                        break

            if img_out is None:
                raise Exception("No image returned")

            target_w, target_h = self._target_size(aspect_ratio, resolution)
            got_w, got_h = img_out.size
            
            # 如果尺寸完全不一致（误差超过64px），才考虑裁剪
            # Gemini 原生出的图（例如 9:16）通常就是我们想要的，即使像素数略有差异，也最好保留全图
            # 所以这里放宽策略：只有当比例严重失调时才裁剪，否则直接使用原图（或者简单resize）
            
            # 计算长宽比差异
            target_ratio = target_w / target_h
            got_ratio = got_w / got_h
            
            if abs(target_ratio - got_ratio) > 0.1: # 比例差异较大，必须裁剪
                img_out = self._fit_to_size(img_out, target_w, target_h)
                # print(f"【yuyu】NanoBanana 比例差异大，已裁剪: {got_w}x{got_h} -> {target_w}x{target_h}")
            else:
                pass
                # print(f"【yuyu】NanoBanana 原生输出: {got_w}x{got_h} (目标 {target_w}x{target_h})，比例接近，不裁剪")

            arr = np.array(img_out).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr)[None,]
            return (tensor,)
            
        except Exception as e:
            raise Exception(f"Parsing Failed: {e}")

class YuyuGeminiImageEditNode:
    def __init__(self):
        self.config = load_config()
        self._instance_id = hashlib.sha256(str(id(self)).encode()).hexdigest()[:8]

    def _target_size(self, aspect_ratio: str, resolution: str):
        base_2k = {
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
        w, h = base_2k.get(aspect_ratio, (2048, 2048))
        if resolution == "1K":
            w, h = max(1, w // 2), max(1, h // 2)
        elif resolution == "4K":
            w, h = w * 2, h * 2
        return int(w), int(h)

    def _fit_to_size(self, img: Image.Image, target_w: int, target_h: int):
        if img.mode != "RGB":
            img = img.convert("RGB")
        src_w, src_h = img.size
        if src_w <= 0 or src_h <= 0:
            return img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        scale = max(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        right = left + target_w
        bottom = top + target_h
        return img.crop((left, top, right, bottom))

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "玉玉API/gemini"

    def edit(self, prompt, image, aspect_ratio, resolution, seed, api_key=None):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt不能为空")

        api_key = api_key or ""
        if not api_key:
            api_key = os.environ.get("YUYU_API_KEY", "")
        if not api_key:
            api_key = self.config.get("YUYU_API_KEY", "")
        if not api_key:
            raise ValueError("API Key未配置，请在Widget、环境变量或config.json中设置。")

        img_tensor = image[0]
        i_img = 255.0 * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        url = "https://yuli.host/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    ],
                }
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
        if seed and int(seed) != 0:
            payload["generationConfig"]["seed"] = abs(int(seed)) % 2147483647

        started = time.time()
        print(f"【yuyu】GeminiEdit 请求: ratio={aspect_ratio} res={resolution}")
        try:
            response = session.post(
                url,
                params={"key": api_key},
                headers=headers,
                json=payload,
                timeout=500, # 限制500秒
                verify=False,
                proxies={"http": None, "https": None},
            )
        except Exception as e:
            print(f"【yuyu】GeminiEdit 直连异常: {e}，尝试默认代理重试...")
            response = session.post(
                url,
                params={"key": api_key},
                headers=headers,
                json=payload,
                timeout=500, # 限制500秒
                verify=False,
            )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        res_json = response.json()
        elapsed = time.time() - started
        cand_count = len(res_json.get("candidates") or []) if isinstance(res_json, dict) else 0
        print(f"【yuyu】GeminiEdit 响应: candidates={cand_count} cost_s={elapsed:.2f}")

        img_out = None
        candidates = res_json.get("candidates") if isinstance(res_json, dict) else None
        if isinstance(candidates, list) and len(candidates) > 0:
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                b64_data = None
                if isinstance(part, dict) and "inline_data" in part:
                    inline = part.get("inline_data") or {}
                    b64_data = inline.get("data")
                elif isinstance(part, dict) and "inlineData" in part:
                    inline = part.get("inlineData") or {}
                    b64_data = inline.get("data")

                if b64_data:
                    img_out = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                    break

                if isinstance(part, dict) and "text" in part and isinstance(part["text"], str) and "http" in part["text"]:
                    image_url = part["text"].strip()
                    
                    if image_url.startswith("data:"):
                         try:
                             header, encoded = image_url.split(",", 1)
                             img_out = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                         except:
                             pass
                    
                    if img_out is None:
                        img_res = session.get(image_url, timeout=300, verify=False)
                        if img_res.status_code != 200:
                            img_res = session.get(image_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=300, verify=False)
                        img_res.raise_for_status()
                        img_out = Image.open(io.BytesIO(img_res.content)).convert("RGB")
                    break

        if img_out is None:
            raise Exception("No image returned")

        target_w, target_h = self._target_size(aspect_ratio, resolution)
        got_w, got_h = img_out.size
        
        target_ratio = target_w / target_h
        got_ratio = got_w / got_h
        
        if abs(target_ratio - got_ratio) > 0.1:
            img_out = self._fit_to_size(img_out, target_w, target_h)
            # print(f"【yuyu】GeminiEdit 比例差异大，已裁剪: {got_w}x{got_h} -> {target_w}x{target_h}")
        else:
            pass
            # print(f"【yuyu】GeminiEdit 原生输出: {got_w}x{got_h} (目标 {target_w}x{target_h})，比例接近，不裁剪")

        arr = np.array(img_out).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None,]
        return (tensor,)

class YuyuGrok3VideoNode:
    def __init__(self):
        self.config = load_config()
        self._instance_id = hashlib.sha256(str(id(self)).encode()).hexdigest()[:8]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["grok-video-3"], {"default": "grok-video-3"}),
                "aspect_ratio": (["1:1", "2:3", "3:2"], {"default": "1:1"}),
                "size": (["720P", "1080P"], {"default": "720P"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": "randomize"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_path", "task_id", "response", "video_url")
    FUNCTION = "generate"
    CATEGORY = "玉玉API/grok"

    def _request_with_retry(self, method, url, headers=None, json_payload=None, params=None, timeout=60, allow_redirects=False):
        last_exc = None
        for attempt in range(5):
            if attempt > 0:
                time.sleep(2 ** attempt)
            try:
                # 【优化】移除proxies, 使用session, verify=False
                if method == "POST":
                    res = session.post(
                        url,
                        headers=headers,
                        json=json_payload,
                        timeout=timeout,
                        allow_redirects=allow_redirects,
                        verify=False
                    )
                else:
                    res = session.get(
                        url,
                        headers=headers,
                        params=params,
                        timeout=timeout,
                        allow_redirects=allow_redirects,
                        verify=False
                    )
                if res.status_code in (502, 503, 504):
                    msg = f"API请求失败: {res.status_code}"
                    last_exc = Exception(msg)
                    print(f"【yuyu】[{self._instance_id}] {msg} (尝试 {attempt+1}/5)")
                    continue
                return res
            except Exception as e:
                last_exc = e
                print(f"【yuyu】[{self._instance_id}] 请求异常: {e} (尝试 {attempt+1}/5)")
        raise last_exc

    def _parse_json_response(self, res, context: str):
        content_type = (res.headers.get("Content-Type") or "").lower()
        text = (res.text or "").strip()
        if not text:
            raise Exception(f"{context}返回空响应: {res.status_code}")
        if "application/json" not in content_type and not (text.startswith("{") or text.startswith("[")):
            snippet = text[:300].replace("\r", " ").replace("\n", " ")
            raise Exception(f"{context}返回非JSON: {res.status_code} - {content_type} - {snippet}")
        try:
            return json.loads(text)
        except Exception as e:
            snippet = text[:300].replace("\r", " ").replace("\n", " ")
            raise Exception(f"{context}解析JSON失败: {e} - {res.status_code} - {content_type} - {snippet}")

    def _get_api_key(self, api_key: str):
        api_key = api_key or ""
        if not api_key:
            api_key = os.environ.get("YUYU_API_KEY", "")
        if not api_key:
            api_key = self.config.get("YUYU_API_KEY", "")
        if not api_key:
            raise ValueError("API Key未配置，请在Widget、环境变量或config.json中设置。")
        return api_key

    def _image_to_data_url(self, image):
        if image is None:
            return None
        img_tensor = image[0]
        i_img = 255.0 * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        raw_bytes = buffered.getvalue()
        img_str = base64.b64encode(raw_bytes).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def generate(self, prompt, model, aspect_ratio, size, seed, **kwargs):
        prompt = prompt or ""
        if not prompt.strip():
            raise ValueError("Prompt不能为空")

        api_key = self._get_api_key(kwargs.get("api_key", ""))
        
        images_list = []
        for i in range(1, 5):
            img_input = kwargs.get(f"image_{i}")
            if img_input is not None:
                data_url = self._image_to_data_url(img_input)
                if data_url:
                    images_list.append(data_url)

        submit_url = "https://yuli.host/v1/video/create"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-yuyuAPI",
        }
        
        payload = {
            "model": model,
            "prompt": prompt.strip(),
            "aspect_ratio": aspect_ratio,
            "size": size,
        }
        
        if seed and int(seed) != 0:
            payload["seed"] = abs(int(seed)) % 2147483647
            
        if images_list:
            payload["images"] = images_list

        trace_id = str(time.time_ns())
        print(f"【yuyu】[{self._instance_id}] Grok3Video提交({trace_id}) -> {submit_url}")

        submit_res = self._request_with_retry(
            "POST",
            submit_url,
            headers=headers,
            json_payload=payload,
            timeout=120, # 优化超时
            allow_redirects=False,
        )
        if submit_res.status_code != 200:
            raise Exception(f"API请求失败: {submit_res.status_code} - {submit_res.text}")
        
        submit_json = self._parse_json_response(submit_res, "提交任务")
        
        task_id = submit_json.get("id") or submit_json.get("task_id")
        if not task_id and isinstance(submit_json.get("data"), dict):
            task_id = submit_json["data"].get("id") or submit_json["data"].get("task_id")
            
        if not task_id:
            raise Exception(f"无法从API响应中获取id: {submit_json}")

        print(f"【yuyu】[{self._instance_id}] 开始查询任务: {task_id}")

        status_url = "https://yuli.host/v1/video/query"
        deadline = time.time() + 600
        poll_json = None
        video_url = None
        
        query_headers = headers.copy()
        if "Content-Type" in query_headers:
            del query_headers["Content-Type"]

        while time.time() < deadline:
            query_params = {"id": task_id}
            
            try:
                poll_res = self._request_with_retry(
                    "GET",
                    status_url,
                    headers=query_headers,
                    params=query_params,
                    timeout=30,
                    allow_redirects=False
                )
            except Exception as e:
                print(f"【yuyu】查询请求异常: {e}")
                time.sleep(2)
                continue
            
            if poll_res.status_code != 200:
                print(f"【yuyu】查询失败: {poll_res.status_code} - {poll_res.text}")
                time.sleep(2)
                continue

            poll_json = self._parse_json_response(poll_res, "查询任务")
            
            status = poll_json.get("status")
            if not status and isinstance(poll_json.get("data"), dict):
                status = poll_json["data"].get("status")

            success_statuses = (
                "completed", "video_generation_completed", "video_upsampling_completed",
                "succeeded", "success", "done"
            )
            
            fail_statuses = (
                "failed", "error", "video_generation_failed", "video_upsampling_failed"
            )

            current_status_lower = str(status).lower() if status else ""
            
            if current_status_lower in success_statuses:
                video_url = poll_json.get("video_url")
                if not video_url and isinstance(poll_json.get("data"), dict):
                    video_url = poll_json["data"].get("video_url")
                
                if video_url:
                    print(f"【yuyu】[{self._instance_id}] 视频链接: {video_url}")
                    break
            
            if current_status_lower in fail_statuses:
                raise Exception(f"任务失败: {poll_json}")
                
            if status:
                 print(f"【yuyu】[{self._instance_id}] 任务状态: {status}")
                 
            time.sleep(2)

        if not video_url:
            raise Exception(f"任务超时或未返回视频URL: {poll_json}")

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        safe_task_id = str(task_id).replace(":", "_").replace("/", "_").replace("\\", "_")
        video_path = os.path.join(output_dir, f"yuyu_grok3video_{safe_task_id}_{int(time.time())}.mp4")

        # 【优化】下载视频移除代理限制并忽略证书
        video_res = session.get(video_url, timeout=600, verify=False)
        video_res.raise_for_status()
        with open(video_path, "wb") as f:
            f.write(video_res.content)

        return (video_path, str(task_id), json.dumps(poll_json, ensure_ascii=False), video_url)

class YuyuGeminiNode:
    def __init__(self):
        self.config = load_config()
        self._instance_id = hashlib.sha256(str(id(self)).encode()).hexdigest()[:8]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system_instruction": ("STRING", {"default": "You are a helpful AI assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["gemini-3-pro-preview", "gemini-3-flash-preview"], {"default": "gemini-3-pro-preview"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": "https://yuli.host", "multiline": False}),
                "output_language": (["Auto", "中文", "English", "Japanese", "Korean"], {"default": "中文"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "strip_thought": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "玉玉API/gemini"

    def _get_api_key(self, api_key):
        api_key = api_key or ""
        if not api_key:
            api_key = os.environ.get("YUYU_API_KEY", "")
        if not api_key:
            api_key = self.config.get("YUYU_API_KEY", "")
        if not api_key:
            raise ValueError("API Key is missing.")
        return api_key

    def _process_image(self, image_tensor):
        i_img = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _process_audio(self, audio_dict):
        waveform = audio_dict['waveform']
        sample_rate = audio_dict['sample_rate']
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        audio_np = waveform.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        buffered = io.BytesIO()
        with wave.open(buffered, 'wb') as wav_file:
            wav_file.setnchannels(audio_int16.shape[0])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.T.tobytes())
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def chat(self, system_instruction, user_prompt, model, api_key, base_url, output_language, temperature, top_p, max_tokens, strip_thought, **kwargs):
        api_key = self._get_api_key(api_key)
        
        final_system_instruction = system_instruction
        magic_separator = "|||OUTPUT_START|||"
        
        if strip_thought:
             strict_instruction = f"CRITICAL: You must start your final actual response with the exact text: {magic_separator}\nEverything before this tag will be considered as thinking process and will be hidden. Output ONLY the final result after the tag."
             if strict_instruction not in final_system_instruction:
                final_system_instruction = f"{strict_instruction}\n{final_system_instruction}"

        parts = []
        parts.append({"text": user_prompt})
        if output_language != "Auto":
            parts.append({"text": f"\nPlease answer in {output_language}."})
        
        for i in range(1, 5):
            key = f"image_{i}"
            if kwargs.get(key) is not None:
                img_b64 = self._process_image(kwargs[key][0])
                parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        
        if kwargs.get("video") is not None:
            video_frames = kwargs["video"]
            total_frames = video_frames.shape[0]
            step = max(1, total_frames // 15)
            for idx in range(0, total_frames, step):
                img_b64 = self._process_image(video_frames[idx])
                parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})
        
        if kwargs.get("audio") is not None:
            audio_b64 = self._process_audio(kwargs["audio"])
            parts.append({"inline_data": {"mime_type": "audio/wav", "data": audio_b64}})
            
        url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": final_system_instruction}]},
            "generationConfig": {"temperature": temperature, "topP": top_p, "maxOutputTokens": max_tokens}
        }
        
        print(f"【yuyu】[{self._instance_id}] Requesting {url}")
        # 【优化】移除proxies, 使用session, verify=False, 增加超时
        response = session.post(url, headers=headers, json=payload, timeout=300, verify=False)
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        res_json = response.json()
        
        try:
            print(f"【yuyu】[{self._instance_id}] Response keys: {list(res_json.keys())}")
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                print(f"【yuyu】[{self._instance_id}] Candidate parts: {len(res_json['candidates'][0].get('content', {}).get('parts', []))}")
        except:
            pass

        try:
            candidate = res_json["candidates"][0]
            content_parts = candidate["content"]["parts"]
            
            full_text = ""
            for part in content_parts:
                if "text" in part:
                    full_text += part["text"]
            
            if strip_thought and magic_separator in full_text:
                print(f"【yuyu】[{self._instance_id}] Detected magic separator, stripping thought process.")
                split_parts = full_text.split(magic_separator)
                if len(split_parts) > 1:
                    full_text = split_parts[-1].strip()
            
            return (full_text,)
        except Exception as e:
            print(f"【yuyu】Response parsing error: {e}, Response: {res_json}")
            return (str(res_json),)

class YuyuDoubaoNode:
    def __init__(self):
        self.config = load_config()
        self._instance_id = hashlib.sha256(str(id(self)).encode()).hexdigest()[:8]

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "api_source": (["official", "yuli"], {"default": "official"}),
                "model": ("STRING", {"default": "doubao-seedream-4-5-251128"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"], {"default": "1:1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
                "group_mode": (["disable", "auto"], {"default": "disable"}),
                "max_images": ("INT", {"default": 15, "min": 1, "max": 15}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "stream": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 600}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
        for i in range(1, 15):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "玉玉API/豆包"

    def _get_api_key(self, api_key):
        api_key = api_key or ""
        if not api_key:
            api_key = os.environ.get("YUYU_API_KEY", "")
        if not api_key:
            api_key = self.config.get("YUYU_API_KEY", "")
        if not api_key:
            raise ValueError("API Key未配置，请在Widget、环境变量或config.json中设置。")
        return api_key

    def _tensor_to_base64(self, image_tensor):
        if image_tensor is None:
            return None
        i_img = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_resolution_size(self, aspect_ratio, resolution_tag):
        base_map = {
            "1:1": (2048, 2048),
            "16:9": (2560, 1440),
            "9:16": (1440, 2560),
            "4:3": (2304, 1728),
            "3:4": (1728, 2304),
            "3:2": (2496, 1664),
            "2:3": (1664, 2496),
            "21:9": (3024, 1296),
            "9:21": (1296, 3024)
        }
        w, h = base_map.get(aspect_ratio, (2048, 2048))
        if resolution_tag == "1K":
            w, h = w // 2, h // 2
        elif resolution_tag == "4K":
            w, h = w * 2, h * 2
        return f"{w}x{h}"

    def generate(self, api_source, model, prompt, aspect_ratio, resolution, group_mode, max_images, seed, stream, watermark, timeout, api_key=None, **kwargs):
        api_key = self._get_api_key(api_key)
        
        input_images = []
        for i in range(1, 15):
            key = f"image_{i}"
            if kwargs.get(key) is not None:
                img_b64 = self._tensor_to_base64(kwargs[key][0])
                input_images.append(img_b64)
        
        size_str = self._get_resolution_size(aspect_ratio, resolution)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size_str,
            "watermark": watermark,
            "stream": stream 
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
            payload["sequential_image_generation_options"] = {
                "max_images": max_images
            }
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
            "Content-Type": "application/json"
        }

        print(f"【yuyu】[{self._instance_id}] Doubao Request: {submit_url}")
        print(f"【yuyu】Params: Model={model}, Size={size_str}, Images={len(input_images)}")

        try:
            is_streaming = payload.get("stream", False)
            # 【优化】移除proxies, 使用session, verify=False
            response = session.post(
                submit_url, 
                headers=headers, 
                json=payload, 
                timeout=timeout, 
                stream=is_streaming,
                verify=False
            )
            
            if response.status_code != 200:
                err_text = ""
                try:
                    err_text = response.text
                except:
                    pass
                raise Exception(f"API Error: {response.status_code} - {err_text}")
                
            image_tensors = []

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
                                i = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                                i = i.convert("RGB")
                                i = np.array(i).astype(np.float32) / 255.0
                                i = torch.from_numpy(i)[None,]
                                image_tensors.append(i)
                        except Exception as e:
                            print(f"【yuyu】Stream parsing error: {e}")
                            pass
            else:
                res_json = response.json()
                if "data" in res_json and isinstance(res_json["data"], list):
                    for item in res_json["data"]:
                        img_url = item.get("url")
                        b64_data = item.get("b64_json") or item.get("binary_data")
                        
                        i = None
                        if img_url:
                            print(f"【yuyu】Downloading: {img_url}")
                            # 【优化】下载图片优化
                            img_res = session.get(img_url, timeout=120, verify=False)
                            i = Image.open(io.BytesIO(img_res.content))
                        elif b64_data:
                            i = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                        
                        if i:
                            i = i.convert("RGB")
                            i = np.array(i).astype(np.float32) / 255.0
                            i = torch.from_numpy(i)[None,]
                            image_tensors.append(i)
            
            if not image_tensors:
                if is_streaming:
                     raise Exception("Stream finished but no images collected.")
                else:
                     raise Exception(f"No images returned. Response: {res_json}")
                
            if len(image_tensors) > 1:
                try:
                    final_tensor = torch.cat(image_tensors, dim=0)
                    return (final_tensor,)
                except:
                    first_shape = image_tensors[0].shape
                    resized_tensors = [image_tensors[0]]
                    for t in image_tensors[1:]:
                        if t.shape != first_shape:
                             print("【yuyu】Warning: Image sizes mismatch in batch. Returning first image only.")
                             return (image_tensors[0],)
                        resized_tensors.append(t)
                    return (torch.cat(resized_tensors, dim=0),)
            else:
                return (image_tensors[0],)

        except Exception as e:
            print(f"【yuyu】Error: {e}")
            raise e

class YuyuVeoNode:
    def __init__(self):
        self.config = load_config()
        self._instance_id = hashlib.sha256(str(id(self)).encode()).hexdigest()[:8]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["veo-2.0-generate-video-preview", "veo_3_1-fast"], {"default": "veo_3_1-fast"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "16:9"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            },
            "optional": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "task_id", "response", "video_url", "preview_image")
    FUNCTION = "generate"
    CATEGORY = "玉玉API/Veo"

    def _get_api_key(self, api_key):
        api_key = api_key or ""
        if not api_key:
            api_key = os.environ.get("YUYU_API_KEY", "")
        if not api_key:
            api_key = self.config.get("YUYU_API_KEY", "")
        if not api_key:
            raise ValueError("API Key未配置，请在Widget、环境变量或config.json中设置。")
        return api_key.strip()

    def _image_to_bytes(self, image):
        if image is None:
            return None
        i_img = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()

    def _load_video_preview(self, video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            max_frames = 60 # Preview limit
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frame = torch.from_numpy(frame)
                frames.append(frame)
                count += 1
            cap.release()
            if frames:
                return torch.stack(frames)
        except Exception as e:
            print(f"【yuyu】Video preview generation failed: {e}")
        
        # Return empty tensor if failed
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    def generate(self, prompt, model, aspect_ratio, fps, image=None, api_key=None):
        api_key = self._get_api_key(api_key)
        
        submit_url = "https://yuli.host/v1/videos"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        
        data = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "fps": str(fps)
        }
        
        files = {}
        if image is not None:
            img_bytes = self._image_to_bytes(image)
            files["input_reference"] = ("input_image.png", img_bytes, "image/png")

        print(f"【yuyu】[{self._instance_id}] Veo Submission: {submit_url}")
        
        try:
            res = session.post(
                submit_url,
                headers=headers,
                data=data,
                files=files if files else None,
                timeout=120,
                verify=False,
                allow_redirects=True 
            )
        except Exception as e:
            raise Exception(f"API Request Failed: {e}")

        if res.status_code != 200:
             raise Exception(f"API Error: {res.status_code} - {res.text}")

        res_json = res.json()
        task_id = res_json.get("id", "")
        video_url = ""
        
        if "url" in res_json:
            video_url = res_json["url"]
        elif "data" in res_json and isinstance(res_json["data"], list):
             video_url = res_json["data"][0].get("url", "")
             
        if not video_url and task_id:
            print(f"【yuyu】[{self._instance_id}] Task ID: {task_id}, polling...")
            deadline = time.time() + 600
            while time.time() < deadline:
                time.sleep(2)
                try:
                    poll_res = session.get(
                        f"https://yuli.host/v1/videos/{task_id}",
                        headers=headers,
                        timeout=30,
                        verify=False
                    )
                    if poll_res.status_code == 200:
                        poll_json = poll_res.json()
                        
                        status = poll_json.get("status", "")
                        if not status and "detail" in poll_json:
                            status = poll_json["detail"].get("status", "")
                        
                        status = status.lower()
                        
                        print(f"【yuyu】[{self._instance_id}] Polling... Status: {status}")
                        if "detail" in poll_json and "pending_info" in poll_json["detail"]:
                             progress = poll_json["detail"]["pending_info"].get("progress_pct", 0)
                             print(f"【yuyu】[{self._instance_id}] Progress: {progress*100:.1f}%")

                        if status in ("succeeded", "completed", "success"):
                            if "video_url" in poll_json:
                                video_url = poll_json["video_url"]
                            elif "url" in poll_json:
                                video_url = poll_json["url"]
                            elif "output" in poll_json and "url" in poll_json["output"]:
                                video_url = poll_json["output"]["url"]
                            elif "download_url" in poll_json:
                                video_url = poll_json["download_url"]
                            elif "video" in poll_json and "url" in poll_json["video"]:
                                video_url = poll_json["video"]["url"]
                            elif "id" in poll_json and str(poll_json["id"]).startswith("video_"):
                                video_url = f"https://yuli.host/v1/videos/{poll_json['id']}/content"
                            elif "video" in poll_json and "id" in poll_json["video"]:
                                video_url = f"https://yuli.host/v1/videos/{poll_json['video']['id']}/content"
                            break
                        elif status in ("failed", "error"):
                            raise Exception(f"Task Failed: {poll_json}")
                except Exception as e:
                    if "Task Failed" in str(e):
                        raise e
                    print(f"【yuyu】Polling Error: {e}")
                    
        if not video_url:
            raise Exception(f"No video URL returned. Response: {poll_json if 'poll_json' in locals() else res_json}")

        # 如果URL是content结尾，尝试解析出真实下载地址
        if video_url.endswith("/content"):
            print(f"【yuyu】正在解析Content URL: {video_url}")
            try:
                resolve_headers = headers.copy()
                resolve_headers["Accept"] = "application/json, text/plain, */*"
                
                # 禁止自动重定向，以便我们手动处理
                r_res = session.get(video_url, headers=resolve_headers, timeout=30, verify=False, allow_redirects=False)
                
                if r_res.status_code in (301, 302, 303, 307, 308):
                    video_url = r_res.headers.get("Location")
                    print(f"【yuyu】捕获重定向，真实下载地址: {video_url}")
                elif r_res.status_code == 200:
                    # 尝试判断是否为JSON
                    content_type = r_res.headers.get("Content-Type", "").lower()
                    is_json = "application/json" in content_type
                    
                    if not is_json:
                        # 即使header不是json，也尝试peek一下内容
                        try:
                            peek_content = r_res.content[:512].strip()
                            if peek_content.startswith(b"{") and b"url" in peek_content:
                                is_json = True
                        except:
                            pass
                    
                    if is_json:
                        try:
                            r_json = r_res.json()
                            if "video_url" in r_json:
                                video_url = r_json["video_url"]
                                print(f"【yuyu】解析JSON成功，真实下载地址: {video_url}")
                            elif "url" in r_json:
                                video_url = r_json["url"]
                                print(f"【yuyu】解析JSON成功，真实下载地址: {video_url}")
                            elif "download_url" in r_json:
                                video_url = r_json["download_url"]
                                print(f"【yuyu】解析JSON成功，真实下载地址: {video_url}")
                            elif "data" in r_json and isinstance(r_json["data"], dict) and "url" in r_json["data"]:
                                video_url = r_json["data"]["url"]
                                print(f"【yuyu】解析JSON成功，真实下载地址: {video_url}")
                        except Exception as e:
                            print(f"【yuyu】尝试解析JSON失败: {e}")
                    else:
                        print(f"【yuyu】Content解析直接返回文件类型: {content_type}，将直接下载")
                else:
                    print(f"【yuyu】解析Content URL返回状态码: {r_res.status_code}")
                    print(f"【yuyu】响应内容: {r_res.text[:200]}")
            except Exception as e:
                print(f"【yuyu】解析Content URL异常: {e}")

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"yuyu_veo_{task_id}_{int(time.time())}.mp4"
        video_path = os.path.join(output_dir, filename)
        
        print(f"【yuyu】Downloading video: {video_url}")
        
        # 下载视频增加Header验证，防止直接GET被拒绝
        dl_headers = headers.copy()
        # 很多S3链接不需要Auth头，且带了会报错，所以尝试移除
        if "yuli.host" not in video_url:
            dl_headers = {}

        try:
            video_res = session.get(video_url, headers=dl_headers, timeout=600, verify=False, stream=True)
            if video_res.status_code != 200:
                 print(f"【yuyu】下载失败({video_res.status_code})，尝试切换Auth头策略重试...")
                 if dl_headers:
                     dl_headers = {}
                 else:
                     dl_headers = headers.copy()
                 video_res = session.get(video_url, headers=dl_headers, timeout=600, verify=False, stream=True)
            
            video_res.raise_for_status()
        except Exception as e:
             raise Exception(f"下载失败: {e}")

        with open(video_path, "wb") as f:
            for chunk in video_res.iter_content(chunk_size=8192):
                f.write(chunk)
        
        preview_image = self._load_video_preview(video_path)
        return (video_path, str(task_id), json.dumps(res_json), video_url, preview_image)

NODE_CLASS_MAPPINGS = {
    "YuyuNanoBananaNode": YuyuNanoBananaNode,
    "YuyuGrok3VideoNode": YuyuGrok3VideoNode,
    "YuyuGeminiNode": YuyuGeminiNode,
    "YuyuDoubaoNode": YuyuDoubaoNode,
    "YuyuVeoNode": YuyuVeoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuyuNanoBananaNode": "Ⓨ Nano Banana 2 (Yuyu)",
    "YuyuGrok3VideoNode": "yuyu Grok3 Video",
    "YuyuGeminiNode": "yuyu Gemini API",
    "YuyuDoubaoNode": "yuyu doubao4.5",
    "YuyuVeoNode": "yuyu Veo Video",
}
