"""Focused regression tests for the active Qwen single-pass grading path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grading import grade_llama
from grading import grade_gemma
from interface.grading_runner import MalformedResultError, parse_model_json


def _valid_grading_payload(score: int = 2) -> dict:
    result = {
        "map_file": "map.pdf",
        "model": grade_llama.MODEL,
        "overall_meets_expectations": "Yes",
        "strengths": [],
        "areas_for_improvement": [],
        "grading_notes": "",
    }
    for group, fields in grade_llama.CATEGORY_FIELDS.items():
        result[group] = {
            "overall_decision": "Yes",
            "if_no_explanation": "",
            **{
                field: {"score": score, "explanation": "Concise rubric explanation."}
                for field in fields
            },
        }
    return result


class _HttpResponse:
    status_code = 200
    headers: dict[str, str] = {}


class QwenSinglePassTests(unittest.TestCase):
    def test_prompt_includes_rubric_once_without_extraction_content(self) -> None:
        prompt = grade_llama.build_prompt("map.pdf")
        self.assertEqual(prompt.count(grade_llama.SPRING_2025_RUBRIC), 1)
        self.assertNotIn("EXTRACTED STUDENT CONCEPT MAP CONTENT", prompt)
        self.assertNotIn('"score": 1', prompt)

    def test_grade_pdf_sends_one_multimodal_request(self) -> None:
        calls: list[tuple[str, str, str]] = []
        payload = _valid_grading_payload()
        response = grade_llama.GroqChatCompletion(
            data={"choices": [{"message": {"content": json.dumps(payload)}}]},
            http_response=_HttpResponse(),
            transport={
                "http_status": 200,
                "request_token_settings": {
                    "stage": "single_pass_grading",
                    "max_completion_tokens": 1200,
                },
            },
        )

        def render(_pdf_path: Path, output_path: Path) -> dict:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"jpeg-bytes")
            return {
                "path": output_path,
                "base64": "encoded-image",
                "width": 100,
                "height": 100,
                "bytes": 10,
                "render_matrix": [1, 1],
                "max_width_px": 1400,
                "jpeg_quality": 80,
            }

        def request_grade(_client: object, prompt: str, image_base64: str):
            calls.append(("grading", prompt, image_base64))
            return response

        with tempfile.TemporaryDirectory() as temp_dir:
            debug_prefix = Path(temp_dir) / "qwen"
            with (
                patch.object(grade_llama, "render_pdf_first_page", render),
                patch.object(grade_llama, "create_client", return_value=object()),
                patch.object(grade_llama, "request_grade", request_grade),
                patch.object(
                    grade_llama,
                    "request_extraction",
                    side_effect=AssertionError("Extraction must not run."),
                ),
            ):
                result = grade_llama.grade_pdf(
                    Path(temp_dir) / "map.pdf", "map.pdf", debug_prefix
                )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "encoded-image")
        self.assertEqual(calls[0][1].count(grade_llama.SPRING_2025_RUBRIC), 1)
        self.assertEqual(result["debug"]["pipeline"], "single_pass_multimodal")
        self.assertEqual(result["debug"]["qwen_request_count"], 1)

    def test_direct_request_payload_contains_the_concept_map_image(self) -> None:
        captured: dict = {}

        def post(_client: object, payload: dict, stage: str):
            captured["payload"] = payload
            captured["stage"] = stage
            return object()

        with patch.object(grade_llama, "_post_groq", post):
            grade_llama.request_grade(object(), "rubric prompt", "encoded-image")

        content = captured["payload"]["messages"][0]["content"]
        self.assertEqual(captured["stage"], "single_pass_grading")
        self.assertEqual(captured["payload"]["max_completion_tokens"], 1200)
        self.assertEqual(content[0], {"type": "text", "text": "rubric prompt"})
        self.assertEqual(
            content[1]["image_url"]["url"], "data:image/jpeg;base64,encoded-image"
        )

    def test_response_extractor_supports_content_blocks(self) -> None:
        response = grade_llama.GroqChatCompletion(
            data={
                "id": "response-id",
                "model": grade_llama.MODEL,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [
                                {"type": "text", "text": "first"},
                                {"type": "output_text", "text": "second"},
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
            http_response=_HttpResponse(),
            transport={},
        )
        attempts: dict = {}
        self.assertEqual(grade_llama.response_text(response, attempts), "first\nsecond")
        self.assertEqual(attempts["qwen_response_diagnostics"]["completion_tokens"], 2)

    def test_reasoning_without_complete_schema_is_not_used_as_a_grade(self) -> None:
        response = grade_llama.GroqChatCompletion(
            data={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": None, "reasoning": "I can see the map."},
                    }
                ]
            },
            http_response=_HttpResponse(),
            transport={},
        )
        with self.assertRaisesRegex(
            grade_llama.EmptyLlamaVisionResponseError,
            "reasoning text but no complete grading JSON",
        ):
            grade_llama.response_text(response, {})

    def test_missing_or_invalid_scores_are_not_defaulted(self) -> None:
        missing_score = _valid_grading_payload()
        del missing_score["knowledge_acquisition"]["basic_science"]["score"]
        with self.assertRaises(MalformedResultError):
            parse_model_json(json.dumps(missing_score))

        invalid_score = _valid_grading_payload()
        invalid_score["knowledge_acquisition"]["basic_science"]["score"] = 5
        with self.assertRaises(MalformedResultError):
            parse_model_json(json.dumps(invalid_score))

    def test_429_retries_only_the_single_request(self) -> None:
        request_count = 0
        messages: list[str] = []

        def request():
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                raise grade_llama.GroqQwenHttpError(
                    "Groq HTTP 429",
                    {
                        "http_status": 429,
                        "response_headers": {"Retry-After": "1"},
                        "response_text": "rate_limit_exceeded",
                        "response_json": {"error": {"code": "rate_limit_exceeded"}},
                    },
                )
            return type("Response", (), {"http_response": _HttpResponse()})()

        with patch.object(grade_llama.time, "sleep") as sleep:
            _, debug = grade_llama._request_with_retry(
                request, "single_pass_grading", messages.append
            )

        self.assertEqual(request_count, 2)
        sleep.assert_called_once_with(3.0)
        self.assertTrue(debug["retry_attempted"])
        self.assertEqual(debug["rate_limit_stage"], "single_pass_grading")

    def test_gemma_configuration_is_unchanged(self) -> None:
        self.assertEqual(grade_gemma.MODEL, "google/gemma-4-26b-a4b-it:free")


if __name__ == "__main__":
    unittest.main()
