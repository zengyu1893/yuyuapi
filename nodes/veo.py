"""
yuyu Veo Video — Veo 3.1 / 2.0 视频生成节点。
"""

import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image

import folder_paths

from .core import session, resolve_api_key, make_instance_id


class YuyuVeoNode:
    def __init__(self):
        self._instance_id = make_instance_id(self)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    ["veo-2.0-generate-video-preview", "veo_3_1-fast"],
                    {"default": "veo_3_1-fast"},
                ),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "16:9"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            },
            "optional": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "task_id", "response", "video_url", "preview_image")
    FUNCTION = "generate"
    CATEGORY = "玉玉API/Veo"

    @staticmethod
    def _image_to_bytes(image):
        if image is None:
            return None
        i_img = 255.0 * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i_img, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _load_video_preview(video_path):
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            frames: list[torch.Tensor] = []
            max_frames = 60
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(torch.from_numpy(frame))
                count += 1
            cap.release()
            if frames:
                return torch.stack(frames)
        except Exception as e:
            print(f"【yuyu】Video preview generation failed: {e}")

        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    def generate(self, prompt, model, aspect_ratio, fps, image=None, api_key=None):
        api_key = resolve_api_key(api_key or "")

        submit_url = "https://yuli.host/v1/videos"
        headers = {"Authorization": f"Bearer {api_key}"}

        data = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "fps": str(fps),
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
                allow_redirects=True,
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
            poll_json: dict = {}
            while time.time() < deadline:
                time.sleep(2)
                try:
                    poll_res = session.get(
                        f"https://yuli.host/v1/videos/{task_id}",
                        headers=headers,
                        timeout=30,
                        verify=False,
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
                            print(f"【yuyu】[{self._instance_id}] Progress: {progress * 100:.1f}%")

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
                        raise
                    print(f"【yuyu】Polling Error: {e}")

        if not video_url:
            raise Exception(f"No video URL returned. Response: {poll_json if 'poll_json' in locals() else res_json}")

        # 解析 /content 结尾的 URL
        if video_url.endswith("/content"):
            print(f"【yuyu】正在解析 Content URL: {video_url}")
            try:
                resolve_headers = headers.copy()
                resolve_headers["Accept"] = "application/json, text/plain, */*"
                r_res = session.get(
                    video_url,
                    headers=resolve_headers,
                    timeout=30,
                    verify=False,
                    allow_redirects=False,
                )

                if r_res.status_code in (301, 302, 303, 307, 308):
                    video_url = r_res.headers.get("Location", video_url)
                    print(f"【yuyu】捕获重定向，真实下载地址: {video_url}")
                elif r_res.status_code == 200:
                    content_type = r_res.headers.get("Content-Type", "").lower()
                    is_json = "application/json" in content_type

                    if not is_json:
                        try:
                            peek = r_res.content[:512].strip()
                            if peek.startswith(b"{") and b"url" in peek:
                                is_json = True
                        except Exception:
                            pass

                    if is_json:
                        try:
                            r_json = r_res.json()
                            for key in ("video_url", "url", "download_url"):
                                if key in r_json:
                                    video_url = r_json[key]
                                    print(f"【yuyu】解析 JSON 成功，真实下载地址: {video_url}")
                                    break
                            if "data" in r_json and isinstance(r_json["data"], dict) and "url" in r_json["data"]:
                                video_url = r_json["data"]["url"]
                                print(f"【yuyu】解析 JSON 成功，真实下载地址: {video_url}")
                        except Exception as e:
                            print(f"【yuyu】尝试解析 JSON 失败: {e}")
                    else:
                        print(f"【yuyu】Content 解析直接返回文件类型: {content_type}，将直接下载")
            except Exception as e:
                print(f"【yuyu】解析 Content URL 异常: {e}")

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"yuyu_veo_{task_id}_{int(time.time())}.mp4"
        video_path = os.path.join(output_dir, filename)

        print(f"【yuyu】Downloading video: {video_url}")

        dl_headers = headers.copy()
        if "yuli.host" not in video_url:
            dl_headers = {}

        try:
            video_res = session.get(video_url, headers=dl_headers, timeout=600, verify=False, stream=True)
            if video_res.status_code != 200:
                print(f"【yuyu】下载失败 ({video_res.status_code})，尝试切换 Auth 头策略重试...")
                dl_headers = {} if dl_headers else headers.copy()
                video_res = session.get(video_url, headers=dl_headers, timeout=600, verify=False, stream=True)
            video_res.raise_for_status()
        except Exception as e:
            raise Exception(f"下载失败: {e}")

        with open(video_path, "wb") as f:
            for chunk in video_res.iter_content(chunk_size=8192):
                f.write(chunk)

        preview_image = self._load_video_preview(video_path)
        return (video_path, str(task_id), json.dumps(res_json, ensure_ascii=False), video_url, preview_image)
