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
        self._last_request_sig = None
        self._last_request_at = 0.0
        self._last_image = None
        self._result_cache = {}
        self._instance_id = hashlib.sha256(str(id(self)).encode()).hexdigest()[:8]

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],),
                "resolution": (["1K", "2K", "4K"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "model": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
        # 添加15个参考图接口
        for i in range(1, 16):
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "玉玉API/Nano Banana"

    def _log_response(self, json_obj):
        try:
            if isinstance(json_obj, dict):
                keys = list(json_obj.keys())
                print(f"【yuyu】[{self._instance_id}] 响应字段: {keys}")
                if "candidates" in json_obj and isinstance(json_obj.get("candidates"), list):
                    print(f"【yuyu】[{self._instance_id}] 候选数量: {len(json_obj.get('candidates'))}")
            else:
                print(f"【yuyu】[{self._instance_id}] 响应类型: {type(json_obj)}")
        except:
            print(f"【yuyu】[{self._instance_id}] 响应: <复杂JSON>")

    def generate(self, prompt, aspect_ratio, resolution, seed, **kwargs):
        prompt = prompt or ""
        if not prompt.strip():
            now = time.time()
            if self._last_image is not None and (now - self._last_request_at) < 60.0:
                print(f"【yuyu】[{self._instance_id}] Prompt为空，复用上一次结果")
                return (self._last_image,)
            raise ValueError("Prompt不能为空")

        # 获取API Key
        api_key = kwargs.get("api_key", "")
        model = kwargs.get("model", "gemini-3-pro-image-preview")

        if not api_key:
            api_key = os.environ.get("YUYU_API_KEY", "")
        if not api_key:
            api_key = self.config.get("YUYU_API_KEY", "")
        
        if not api_key:
            raise ValueError("API Key未配置，请在Widget、环境变量或config.json中设置。")

        # 处理参考图
        input_images = []
        image_hashes = []
        for i in range(1, 16):
            img_key = f"image_{i}"
            if img_key in kwargs and kwargs[img_key] is not None:
                img_tensor = kwargs[img_key][0]
                i_img = 255. * img_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                raw_bytes = buffered.getvalue()
                img_str = base64.b64encode(raw_bytes).decode("utf-8")
                input_images.append(f"data:image/png;base64,{img_str}")
                image_hashes.append(hashlib.sha256(raw_bytes).hexdigest()[:16])

        request_sig = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "seed": seed,
                "image_hashes": image_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        now = time.time()
        cached = self._result_cache.get(request_sig)
        if cached and (now - cached.get("at", 0.0)) < 60.0 and cached.get("image") is not None:
            print(f"【yuyu】[{self._instance_id}] 检测到重复执行，直接复用缓存结果")
            return (cached["image"],)
        if self._last_request_sig == request_sig and (now - self._last_request_at) < 60.0 and self._last_image is not None:
            print(f"【yuyu】[{self._instance_id}] 检测到重复执行，直接复用上一次结果")
            self._result_cache[request_sig] = {"at": now, "image": self._last_image}
            return (self._last_image,)
        self._last_request_sig = request_sig
        self._last_request_at = now

        # 默认使用通用OpenAI/Fal兼容接口
        submit_url = "https://yuli.host/v1/images/generations" 
        
        # 构建请求参数
        if model == "gemini-3-pro-image-preview":
            submit_url = f"https://yuli.host/v1beta/models/{model}:generateContent"

            payload = {
                "contents": [{
                    "parts": [{"text": prompt.strip()}]
                }],
                "generationConfig": {
                    "candidateCount": 1,
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": resolution,
                    },
                }
            }
            if seed and int(seed) != 0:
                # Gemini API seed 限制为 32 位整数，而 ComfyUI 是 64 位
                # 需要进行截断或取模处理，确保在 INT32 范围内 (-2147483648 到 2147483647)
                # 这里使用 abs(seed) % 2147483647 确保正数且在范围内
                payload["generationConfig"]["seed"] = abs(int(seed)) % 2147483647
            
            # 添加参考图 (Multimodal)
            if input_images:
                # Gemini 接受 inline_data
                for img_data_url in input_images:
                    # 移除 data:image/png;base64, 前缀
                    b64_data = img_data_url.split(",")[1]
                    payload["contents"][0]["parts"].append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": b64_data
                        }
                    })
        else:
            # Fal / OpenAI 兼容格式
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "seed": seed
            }
            if input_images:
                payload["image_urls"] = input_images

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 提交任务
        trace_id = str(time.time_ns())
        print(f"【yuyu】[{self._instance_id}] 开始请求({trace_id}) -> {submit_url}")
        print(f"【yuyu】[{self._instance_id}] 参数({trace_id}): 模型={model}, 比例={aspect_ratio}, 分辨率={resolution}, 参考图={len(input_images)}")

        # 直接禁用系统代理进行直连，避免本地代理配置错误导致连接失败
        response = requests.post(submit_url, headers=headers, json=payload, timeout=60, proxies={"http": None, "https": None})
        
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")

        res_json = response.json()
        self._log_response(res_json)

        image_url = None
        
        # 解析 Gemini 响应
        if "candidates" in res_json:
            # Gemini 原生响应
            # 通常图片不会直接返回 URL，而是 Base64，或者通过 output 字段
            # 如果是 Preview 模型，可能返回结构有所不同
            # 假设 Yuli Host 做了处理返回 URL，或者返回 Base64
            # 注意：REST API 返回的 JSON 字段通常是驼峰命名 (inlineData)，而 Python SDK 是下划线 (inline_data)
            candidate = res_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                got_image = False
                for part in candidate["content"]["parts"]:
                    # 检查 inline_data (SDK风格) 或 inlineData (REST风格)
                    b64_data = None
                    if "inline_data" in part:
                        b64_data = part["inline_data"]["data"]
                    elif "inlineData" in part:
                        b64_data = part["inlineData"]["data"]
                    
                    if b64_data:
                        got_image = True
                        try:
                            i = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                            i = i.convert("RGB")
                            i = np.array(i).astype(np.float32) / 255.0
                            i = torch.from_numpy(i)[None,]
                            i = i[:1]
                            self._last_image = i
                            self._result_cache[request_sig] = {"at": time.time(), "image": i}
                            return (i,)
                        except Exception as e:
                            print(f"【yuyu】[{self._instance_id}] Base64解码失败: {e}")
                    
                    elif "text" in part and "http" in part["text"]:
                        # 极其罕见：URL在文本里
                        image_url = part["text"] # 需要提取
                if not got_image and not image_url:
                    raise Exception(f"Gemini未返回图片，请检查imageConfig(imageSize/aspectRatio)与responseModalities。响应: {res_json}")
            
        # 解析 OpenAI/Fal 响应
        if not image_url:
            if "data" in res_json and isinstance(res_json["data"], list) and "url" in res_json["data"][0]:
                image_url = res_json["data"][0]["url"]
            elif "request_id" in res_json or "id" in res_json:
                # 异步轮询 (Fal 风格)
                task_id = res_json.get("request_id") or res_json.get("id")
                # 注意：如果 submit_url 变了，status_url 也得变
                # 简单处理：如果用了 v1beta，大概率是同步的；如果是 v1/images，是异步的
                if "v1/images" in submit_url:
                    status_url = f"https://yuli.host/v1/images/generations/{task_id}" 
                    while True:
                        time.sleep(2)
                        poll_res = requests.get(status_url, headers=headers, timeout=30)
                        poll_json = poll_res.json()
                        print(f"【yuyu】[{self._instance_id}] Poll Status: {poll_json.get('status', 'unknown')}")
                        
                        status = poll_json.get("status")
                        if status == "COMPLETED" or status == "SUCCEEDED":
                            if "output" in poll_json and "url" in poll_json["output"]:
                                image_url = poll_json["output"]["url"]
                            elif "data" in poll_json and isinstance(poll_json["data"], list):
                                image_url = poll_json["data"][0]["url"]
                            break
                        elif status == "FAILED":
                            raise Exception(f"任务失败: {poll_json}")

        if not image_url:
            if "url" in res_json:
                image_url = res_json["url"]
            elif "data" in res_json and isinstance(res_json["data"], list) and "b64_json" in res_json["data"][0]:
                b64_data = res_json["data"][0]["b64_json"]
                i = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                i = i.convert("RGB")
                i = np.array(i).astype(np.float32) / 255.0
                i = torch.from_numpy(i)[None,]
                i = i[:1]
                self._last_image = i
                self._result_cache[request_sig] = {"at": time.time(), "image": i}
                return (i,)
            else:
                raise Exception(f"无法从API响应中获取图片: {res_json}")

        # 下载图片
        print(f"【yuyu】[{self._instance_id}] 下载图片: {image_url}")
        img_res = requests.get(image_url)
        img_res.raise_for_status()
        
        i = Image.open(io.BytesIO(img_res.content))
        i = i.convert("RGB")
        i = np.array(i).astype(np.float32) / 255.0
        i = torch.from_numpy(i)[None,]
        i = i[:1]
        self._last_image = i
        self._result_cache[request_sig] = {"at": time.time(), "image": i}
        
        return (i,)

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

    def _request_with_retry(self, method, url, headers=None, json_payload=None, timeout=60, allow_redirects=False):
        last_exc = None
        for attempt in range(5):
            if attempt > 0:
                time.sleep(2 ** attempt)
            try:
                if method == "POST":
                    res = requests.post(
                        url,
                        headers=headers,
                        json=json_payload,
                        timeout=timeout,
                        proxies={"http": None, "https": None},
                        allow_redirects=allow_redirects,
                    )
                else:
                    res = requests.get(
                        url,
                        headers=headers,
                        timeout=timeout,
                        proxies={"http": None, "https": None},
                        allow_redirects=allow_redirects,
                    )
                if res.status_code in (502, 503, 504):
                    msg = f"API请求失败: {res.status_code}"
                    if res.text and len(res.text) < 100:
                        msg += f" - {res.text}"
                    else:
                        msg += " - Bad Gateway / Service Unavailable"
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
        
        # 处理多张图片输入
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
            # API文档未明确seed参数，但通常为了复现可以保留，或者根据用户示例调整
            # 用户示例中没有seed，但为了兼容性先保留，如果不生效也不会报错
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
            timeout=60,
            allow_redirects=False,
        )
        if submit_res.status_code != 200:
            raise Exception(f"API请求失败: {submit_res.status_code} - {submit_res.text}")
        
        submit_json = self._parse_json_response(submit_res, "提交任务")
        
        # 获取任务ID
        task_id = submit_json.get("id") or submit_json.get("task_id")
        if not task_id and isinstance(submit_json.get("data"), dict):
            task_id = submit_json["data"].get("id") or submit_json["data"].get("task_id")
            
        if not task_id:
            raise Exception(f"无法从API响应中获取id: {submit_json}")

        print(f"【yuyu】[{self._instance_id}] 开始查询任务: {task_id}")

        # 查询状态
        status_url = "https://yuli.host/v1/video/query"
        deadline = time.time() + 600
        poll_json = None
        video_url = None
        
        query_headers = headers.copy()
        if "Content-Type" in query_headers:
            del query_headers["Content-Type"]

        while time.time() < deadline:
            query_params = {"id": task_id}
            
            poll_res = requests.get(
                status_url,
                headers=query_headers,
                params=query_params,
                timeout=30,
                proxies={"http": None, "https": None},
            )
            
            if poll_res.status_code != 200:
                print(f"【yuyu】查询失败: {poll_res.status_code} - {poll_res.text}")
                time.sleep(2)
                continue

            poll_json = self._parse_json_response(poll_res, "查询任务")
            
            # 解析状态
            status = poll_json.get("status")
            if not status and isinstance(poll_json.get("data"), dict):
                status = poll_json["data"].get("status")

            # 成功状态判断
            success_statuses = (
                "completed", "video_generation_completed", "video_upsampling_completed",
                "succeeded", "success", "done"
            )
            
            # 失败状态判断
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

        # 下载视频
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        safe_task_id = str(task_id).replace(":", "_").replace("/", "_").replace("\\", "_")
        video_path = os.path.join(output_dir, f"yuyu_grok3video_{safe_task_id}_{int(time.time())}.mp4")

        video_res = requests.get(video_url, timeout=300, proxies={"http": None, "https": None})
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
        
        # 处理思维链屏蔽 - 使用分隔符策略
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
        response = requests.post(url, headers=headers, json=payload, timeout=120, proxies={"http": None, "https": None})
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        res_json = response.json()
        
        # 打印响应摘要以便调试
        try:
            print(f"【yuyu】[{self._instance_id}] Response keys: {list(res_json.keys())}")
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                print(f"【yuyu】[{self._instance_id}] Candidate parts: {len(res_json['candidates'][0].get('content', {}).get('parts', []))}")
        except:
            pass

        try:
            candidate = res_json["candidates"][0]
            content_parts = candidate["content"]["parts"]
            
            # 拼接所有文本部分
            full_text = ""
            for part in content_parts:
                if "text" in part:
                    full_text += part["text"]
            
            # 处理分隔符过滤
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
        # 添加 data:image/png;base64, 前缀
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_resolution_size(self, aspect_ratio, resolution_tag):
        # 基础尺寸映射 (基于2K)
        # 1K = 0.5x, 4K = 2x
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
        
        # 收集图片
        input_images = []
        for i in range(1, 15):
            key = f"image_{i}"
            if kwargs.get(key) is not None:
                img_b64 = self._tensor_to_base64(kwargs[key][0])
                input_images.append(img_b64)
        
        # 构建Payload
        size_str = self._get_resolution_size(aspect_ratio, resolution)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size_str,
            "watermark": watermark,
            "stream": stream 
        }

        if seed != -1:
            # 限制 seed 在 INT32 范围内 (0 ~ 2147483647)，避免 API 报错 "无效参数"
            payload["seed"] = abs(seed) % 2147483647

        if input_images:
            # 根据示例，如果是单张图，传入单个字符串；如果是多张图，传入列表
            # 这里统一使用列表，Volcengine API 通常支持
            if len(input_images) == 1:
                payload["image"] = input_images[0]
            else:
                payload["image"] = input_images
            
        if group_mode == "auto":
            payload["sequential_image_generation"] = "auto"
            payload["sequential_image_generation_options"] = {
                "max_images": max_images
            }
            # 组图模式下，强制开启流式返回 b64_json，以便实时处理事件
            payload["stream"] = True
            payload["response_format"] = "b64_json"
        elif stream:
            # 如果用户手动开启了 stream
            payload["response_format"] = "b64_json"

        # 确定 Endpoint
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
            # 如果开启了 Stream，需要流式处理
            is_streaming = payload.get("stream", False)
            response = requests.post(
                submit_url, 
                headers=headers, 
                json=payload, 
                timeout=timeout, 
                proxies={"http": None, "https": None},
                stream=is_streaming
            )
            
            if response.status_code != 200:
                # 尝试读取错误信息
                err_text = ""
                try:
                    err_text = response.text
                except:
                    pass
                raise Exception(f"API Error: {response.status_code} - {err_text}")
                
            image_tensors = []

            if is_streaming:
                # 处理 SSE 流
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
                            # 检查事件类型
                            # 示例: event.type == "image_generation.partial_succeeded"
                            # 或者直接看 b64_json 是否存在
                            # 注意：Raw API 返回的 JSON 结构可能与 SDK 的 event 对象不同
                            # 通常是 { "b64_json": "...", "type": "..." } 或类似的
                            
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
                # 普通 JSON 响应
                res_json = response.json()
                if "data" in res_json and isinstance(res_json["data"], list):
                    for item in res_json["data"]:
                        img_url = item.get("url")
                        b64_data = item.get("b64_json") or item.get("binary_data")
                        
                        i = None
                        if img_url:
                            print(f"【yuyu】Downloading: {img_url}")
                            img_res = requests.get(img_url, timeout=60)
                            i = Image.open(io.BytesIO(img_res.content))
                        elif b64_data:
                            i = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                        
                        if i:
                            i = i.convert("RGB")
                            i = np.array(i).astype(np.float32) / 255.0
                            i = torch.from_numpy(i)[None,]
                            image_tensors.append(i)
            
            if not image_tensors:
                # 如果是流式，可能所有都失败了或者没解析到
                if is_streaming:
                     raise Exception("Stream finished but no images collected.")
                else:
                     raise Exception(f"No images returned. Response: {res_json}")
                
            # 如果有多张图，ComfyUI怎么返回？
            # 通常返回 Batch Tensor (N, H, W, C)
            # 这里我们需要把它们拼起来
            if len(image_tensors) > 1:
                # 假设尺寸一致
                # 需要检查尺寸是否一致，如果不一致需要 Resize
                # Seedream 组图通常尺寸一致
                try:
                    final_tensor = torch.cat(image_tensors, dim=0)
                    return (final_tensor,)
                except:
                    # 如果尺寸不一致，只返回第一张或报错
                    # 这里尝试 Resize 到第一张的尺寸
                    first_shape = image_tensors[0].shape
                    resized_tensors = [image_tensors[0]]
                    for t in image_tensors[1:]:
                        if t.shape != first_shape:
                             # 使用 torch.nn.functional.interpolate 或其他方式
                             # 简单起见，这里仅打印警告并返回第一张
                             print("【yuyu】Warning: Image sizes mismatch in batch. Returning first image only.")
                             return (image_tensors[0],)
                        resized_tensors.append(t)
                    return (torch.cat(resized_tensors, dim=0),)
            else:
                return (image_tensors[0],)

        except Exception as e:
            print(f"【yuyu】Error: {e}")
            raise e

NODE_CLASS_MAPPINGS = {
    "YuyuNanoBananaNode": YuyuNanoBananaNode,
    "YuyuGrok3VideoNode": YuyuGrok3VideoNode,
    "YuyuGeminiNode": YuyuGeminiNode,
    "YuyuDoubaoNode": YuyuDoubaoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuyuNanoBananaNode": "Ⓨ Nano Banana 2 (Yuyu)",
    "YuyuGrok3VideoNode": "yuyu Grok3 Video",
    "YuyuGeminiNode": "yuyu Gemini API",
    "YuyuDoubaoNode": "yuyu doubao4.5",
}
