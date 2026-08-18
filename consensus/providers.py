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
    image_base64: str | None,
    max_tokens: int,
    timeout_seconds: int,
    stage: str = "consensus",
    progress_callback: Any | None = None,
) -> ProviderCallResult:
    normalized = provider.strip().lower()
    if normalized in {"gemma", "openrouter"}:
        client = grade_gemma.create_client()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                }
            )
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=0,
            timeout=timeout_seconds,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        raw_text = grade_gemma.response_text(response)
        raw_response = grade_gemma._response_debug_value(response)
        choices = getattr(response, "choices", None) or []
        first_choice = choices[0] if choices else None
        usage = getattr(response, "usage", None)
        return ProviderCallResult(
            raw_text,
            raw_response,
            {
                "provider": "OpenRouter",
                "model_id": model_id,
                "timeout_seconds": timeout_seconds,
                "max_tokens": max_tokens,
                "streaming_enabled": False,
                "image_resent": bool(image_base64),
                "prompt_character_count": len(prompt),
                "finish_reason": getattr(first_choice, "finish_reason", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        )
    if normalized in {"llama", "nvidia", "nvidia nim"}:
        client = grade_llama.create_client()
        messages = (
            grade_llama._vision_messages(prompt, image_base64)
            if image_base64
            else [{"role": "user", "content": prompt}]
        )
        payload = grade_llama._nvidia_payload(
            messages,
            response_format=False,
            temperature=0,
            top_p=1,
            max_tokens=max_tokens,
            stream=True,
        )
        response, retry_debug = grade_llama._request_with_retry(
            lambda timeout: grade_llama._post_nvidia(
                client,
                payload,
                stream=True,
                timeout=timeout,
                progress_callback=progress_callback,
            ),
            progress_callback,
            stage=stage,
        )
        attempts = {"deliberation": grade_llama._response_dump(response)}
        raw_text = grade_llama.response_text(response, attempts)
        response_dump = grade_llama._response_dump(response)
        choices = response_dump.get("choices", []) if isinstance(response_dump, dict) else []
        first_choice = choices[0] if choices else {}
        usage = response_dump.get("usage", {}) if isinstance(response_dump, dict) else {}
        return ProviderCallResult(
            raw_text,
            response_dump,
            {
                "provider": "NVIDIA NIM",
                "model_id": model_id,
                "timeout_seconds": grade_llama.READ_TIMEOUT_SECONDS,
                "connect_timeout_seconds": grade_llama.CONNECT_TIMEOUT_SECONDS,
                "read_timeout_seconds": grade_llama.READ_TIMEOUT_SECONDS,
                "retry_read_timeout_seconds": retry_debug.get(
                    "retry_read_timeout_seconds"
                ),
                "retry_attempted": retry_debug.get("retry_attempted", False),
                "retry_wait_seconds": retry_debug.get("retry_wait_seconds", 0),
                "transport_attempts": retry_debug.get("attempts", []),
                "stage": stage,
                "max_tokens": max_tokens,
                "streaming_enabled": True,
                "image_resent": bool(image_base64),
                "prompt_character_count": len(prompt),
                "http_status": response.transport.get("http_status"),
                "finish_reason": first_choice.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "time_to_first_token_seconds": response.transport.get(
                    "time_to_first_token_seconds"
                ),
                "duration_seconds": response.transport.get(
                    "elapsed_request_seconds"
                ),
            },
        )
    raise RuntimeError(f"Unsupported consensus provider: {provider}")
