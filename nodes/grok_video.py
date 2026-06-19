"""
yuyu Grok3 Video — 视频生成节点（Grok 3）。
"""

import json
import os
import time

import folder_paths

from .core import session, resolve_api_key, tensor_to_data_url, make_instance_id


class YuyuGrok3VideoNode:
    def __init__(self):
        self._instance_id = make_instance_id(self)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["grok-video-3"], {"default": "grok-video-3"}),
                "aspect_ratio": (["1:1", "2:3", "3:2"], {"default": "1:1"}),
                "size": (["720P", "1080P"], {"default": "720P"}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": "randomize"},
                ),
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
                time.sleep(2**attempt)
            try:
                if method == "POST":
                    res = session.post(
                        url,
                        headers=headers,
                        json=json_payload,
                        timeout=timeout,
                        allow_redirects=allow_redirects,
                        verify=False,
                    )
                else:
                    res = session.get(
                        url,
                        headers=headers,
                        params=params,
                        timeout=timeout,
                        allow_redirects=allow_redirects,
                        verify=False,
                    )
                if res.status_code in (502, 503, 504):
                    msg = f"API 请求失败: {res.status_code}"
                    last_exc = Exception(msg)
                    print(f"【yuyu】[{self._instance_id}] {msg} (尝试 {attempt + 1}/5)")
                    continue
                return res
            except Exception as e:
                last_exc = e
                print(f"【yuyu】[{self._instance_id}] 请求异常: {e} (尝试 {attempt + 1}/5)")
        raise last_exc

    def _parse_json_response(self, res, context: str):
        content_type = (res.headers.get("Content-Type") or "").lower()
        text = (res.text or "").strip()
        if not text:
            raise Exception(f"{context} 返回空响应: {res.status_code}")
        if "application/json" not in content_type and not (text.startswith("{") or text.startswith("[")):
            snippet = text[:300].replace("\r", " ").replace("\n", " ")
            raise Exception(f"{context} 返回非 JSON: {res.status_code} - {content_type} - {snippet}")
        try:
            return json.loads(text)
        except Exception as e:
            snippet = text[:300].replace("\r", " ").replace("\n", " ")
            raise Exception(f"{context} 解析 JSON 失败: {e} - {res.status_code} - {content_type} - {snippet}")

    def generate(self, prompt, model, aspect_ratio, size, seed, **kwargs):
        prompt = prompt or ""
        if not prompt.strip():
            raise ValueError("Prompt 不能为空")

        api_key = resolve_api_key(kwargs.get("api_key", ""))

        images_list = []
        for i in range(1, 5):
            img_input = kwargs.get(f"image_{i}")
            if img_input is not None:
                data_url = tensor_to_data_url(img_input)
                if data_url:
                    images_list.append(data_url)

        submit_url = "https://yuli.host/v1/video/create"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-yuyuAPI",
        }

        payload: dict = {
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
        print(f"【yuyu】[{self._instance_id}] Grok3Video 提交 ({trace_id}) -> {submit_url}")

        submit_res = self._request_with_retry(
            "POST",
            submit_url,
            headers=headers,
            json_payload=payload,
            timeout=120,
            allow_redirects=False,
        )
        if submit_res.status_code != 200:
            raise Exception(f"API 请求失败: {submit_res.status_code} - {submit_res.text}")

        submit_json = self._parse_json_response(submit_res, "提交任务")

        task_id = submit_json.get("id") or submit_json.get("task_id")
        if not task_id and isinstance(submit_json.get("data"), dict):
            task_id = submit_json["data"].get("id") or submit_json["data"].get("task_id")
        if not task_id:
            raise Exception(f"无法从 API 响应中获取 id: {submit_json}")

        print(f"【yuyu】[{self._instance_id}] 开始查询任务: {task_id}")

        status_url = "https://yuli.host/v1/video/query"
        deadline = time.time() + 600
        poll_json = None
        video_url = None

        query_headers = headers.copy()
        query_headers.pop("Content-Type", None)

        while time.time() < deadline:
            try:
                poll_res = self._request_with_retry(
                    "GET",
                    status_url,
                    headers=query_headers,
                    params={"id": task_id},
                    timeout=30,
                    allow_redirects=False,
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
                "completed",
                "video_generation_completed",
                "video_upsampling_completed",
                "succeeded",
                "success",
                "done",
            )
            fail_statuses = ("failed", "error", "video_generation_failed", "video_upsampling_failed")

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
            raise Exception(f"任务超时或未返回视频 URL: {poll_json}")

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        safe_task_id = str(task_id).replace(":", "_").replace("/", "_").replace("\\", "_")
        video_path = os.path.join(output_dir, f"yuyu_grok3video_{safe_task_id}_{int(time.time())}.mp4")

        video_res = session.get(video_url, timeout=600, verify=False)
        video_res.raise_for_status()
        with open(video_path, "wb") as f:
            f.write(video_res.content)

        return (video_path, str(task_id), json.dumps(poll_json, ensure_ascii=False), video_url)
