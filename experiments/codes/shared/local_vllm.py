"""
Shared helpers for local vLLM-served multimodal models.
"""

from __future__ import annotations

import base64
import io
import os
import time
from types import SimpleNamespace


class LocalVLLMService:
    def __init__(self, display, base_url, model_name, context_len, api_key="EMPTY"):
        from openai import OpenAI

        self.display = display
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.context_len = context_len
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self.config = SimpleNamespace(max_position_embeddings=context_len)


def create_service_from_config(cfg):
    return LocalVLLMService(
        display=cfg["display"],
        base_url=os.environ.get(cfg["base_url_env"], cfg["default_base_url"]),
        model_name=os.environ.get(cfg["model_env"], cfg["default_model_name"]),
        context_len=cfg["context_len"],
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )


def _pil_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def to_openai_content(content):
    api_content = []
    for item in content:
        kind = item.get("type")
        if kind == "text":
            api_content.append({"type": "text", "text": item["text"]})
        elif kind == "image":
            api_content.append({
                "type": "image_url",
                "image_url": {"url": _pil_to_data_url(item["image"])},
            })
        elif kind == "image_url":
            api_content.append(item)
        else:
            raise ValueError(f"Unsupported content type: {kind}")
    return api_content


def infer_from_content(service, system_prompt, content, max_tokens, *, max_attempts=3, sleep_after=0.0):
    out = {"text": "", "n_input_tokens": None, "n_output_tokens": None, "error": None}
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": to_openai_content(content)},
        ]
        resp = None
        for attempt in range(max_attempts):
            try:
                resp = service.client.chat.completions.create(
                    model=service.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0,
                )
                break
            except Exception as e:
                if attempt < max_attempts - 1 and "rate" in str(e).lower():
                    time.sleep(30)
                else:
                    raise
        if resp is None:
            raise RuntimeError("No response received from vLLM service.")
        usage = getattr(resp, "usage", None)
        out["n_input_tokens"] = getattr(usage, "prompt_tokens", None)
        out["n_output_tokens"] = getattr(usage, "completion_tokens", None)
        out["text"] = (resp.choices[0].message.content or "").strip()
        if sleep_after:
            time.sleep(sleep_after)
    except Exception as e:
        out["error"] = str(e)
    return out
