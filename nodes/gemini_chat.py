"""
yuyu Gemini LLM — 多模态对话节点（图片 / 视频 / 音频输入）。
"""

import base64
import io
import wave

import numpy as np
from PIL import Image

from .core import session, resolve_api_key, tensor_to_pil, make_instance_id


class YuyuGeminiNode:
    def __init__(self):
        self._instance_id = make_instance_id(self)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_instruction": (
                    "STRING",
                    {"default": "You are a helpful AI assistant.", "multiline": True},
                ),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "gemini-3.1-pro-preview",
                        "gemini-3-pro-preview",
                        "gemini-3-flash-preview",
                    ],
                    {"default": "gemini-3.1-pro-preview"},
                ),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "base_url": ("STRING", {"default": "https://yuli.host", "multiline": False}),
                "output_language": (["Auto", "中文", "English", "Japanese", "Korean"], {"default": "中文"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "strip_thought": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_model": ("STRING", {"default": "", "multiline": False}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "玉玉API/gemini"

    @staticmethod
    def _process_image(image_tensor):
        img = tensor_to_pil(image_tensor)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _process_audio(audio_dict):
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        audio_np = waveform.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(audio_int16.shape[0])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.T.tobytes())
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def chat(
        self,
        system_instruction,
        user_prompt,
        model,
        api_key,
        base_url,
        output_language,
        temperature,
        top_p,
        max_tokens,
        strip_thought,
        custom_model="",
        **kwargs,
    ):
        api_key = resolve_api_key(api_key)

        # 自定义模型优先，为空时用下拉选择的模型
        effective_model = custom_model.strip() if custom_model else model

        final_system_instruction = system_instruction
        magic_separator = "|||OUTPUT_START|||"

        if strip_thought:
            strict_instruction = (
                f"CRITICAL: You must start your final actual response with the exact text: {magic_separator}\n"
                "Everything before this tag will be considered as thinking process and will be hidden. "
                "Output ONLY the final result after the tag."
            )
            if strict_instruction not in final_system_instruction:
                final_system_instruction = f"{strict_instruction}\n{final_system_instruction}"

        parts: list = []
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

        url = f"{base_url.rstrip('/')}/v1beta/models/{effective_model}:generateContent"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": final_system_instruction}]},
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
            },
        }

        print(f"【yuyu】[{self._instance_id}] Requesting {effective_model} -> {url}")
        response = session.post(url, headers=headers, json=payload, timeout=300, verify=False)
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        res_json = response.json()

        try:
            print(f"【yuyu】[{self._instance_id}] Response keys: {list(res_json.keys())}")
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                candidate = res_json["candidates"][0]
                parts_count = len(candidate.get("content", {}).get("parts", []))
                print(f"【yuyu】[{self._instance_id}] Candidate parts: {parts_count}")
        except Exception:
            pass

        try:
            candidate = res_json["candidates"][0]
            content_parts = candidate["content"]["parts"]

            full_text = "".join(part.get("text", "") for part in content_parts if "text" in part)

            if strip_thought and magic_separator in full_text:
                print(f"【yuyu】[{self._instance_id}] Detected magic separator, stripping thought process.")
                split_parts = full_text.split(magic_separator)
                if len(split_parts) > 1:
                    full_text = split_parts[-1].strip()

            return (full_text,)
        except Exception as e:
            print(f"【yuyu】Response parsing error: {e}, Response: {res_json}")
            return (str(res_json),)
