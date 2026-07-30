"""Single-call provider adapters for deliberation stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grading import grade_gemma, grade_llama


@dataclass(frozen=True)
class ProviderCallResult:
    raw_text: str
    raw_response: Any
    metadata: dict[str, Any]


def invoke_model(
    *,
    provider: str,
    model_id: str,
    prompt: str,
    image_base64: str,
    max_tokens: int,
    timeout_seconds: int,
) -> ProviderCallResult:
    normalized = provider.strip().lower()
    if normalized in {"gemma", "openrouter"}:
        client = grade_gemma.create_client()
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=0,
            timeout=timeout_seconds,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        )
        raw_text = grade_gemma.response_text(response)
        raw_response = grade_gemma._response_debug_value(response)
        return ProviderCallResult(
            raw_text,
            raw_response,
            {
                "provider": "OpenRouter",
                "model_id": model_id,
                "timeout_seconds": timeout_seconds,
                "max_tokens": max_tokens,
                "streaming_enabled": False,
                "image_resent": True,
                "prompt_character_count": len(prompt),
            },
        )
    if normalized in {"llama", "nvidia", "nvidia nim"}:
        client = grade_llama.create_client()
        payload = grade_llama._nvidia_payload(
            grade_llama._vision_messages(prompt, image_base64),
            response_format=False,
            temperature=0,
            top_p=1,
            max_tokens=max_tokens,
        )
        response = grade_llama._post_nvidia(
            client,
            payload,
            timeout=timeout_seconds,
        )
        attempts = {"deliberation": grade_llama._response_dump(response)}
        raw_text = grade_llama.response_text(response, attempts)
        return ProviderCallResult(
            raw_text,
            grade_llama._response_dump(response),
            {
                "provider": "NVIDIA NIM",
                "model_id": model_id,
                "timeout_seconds": timeout_seconds,
                "max_tokens": max_tokens,
                "streaming_enabled": False,
                "image_resent": True,
                "prompt_character_count": len(prompt),
                "http_status": response.transport.get("http_status"),
            },
        )
    raise RuntimeError(f"Unsupported consensus provider: {provider}")
