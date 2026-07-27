"""Regression coverage for NVIDIA Llama 3.2 90B Vision grading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grading import grade_gemma, grade_llama
from interface.grading_runner import MalformedResultError, parse_model_json, selected_model_names
from scripts import generate_evaluation_report


def valid_payload(score: int = 2) -> dict:
    result = {"map_file": "map.pdf", "model": grade_llama.MODEL, "overall_meets_expectations": "Yes", "strengths": [], "areas_for_improvement": [], "grading_notes": ""}
    for group, fields in grade_llama.CATEGORY_FIELDS.items():
        result[group] = {"overall_decision": "Yes", "if_no_explanation": "", **{field: {"score": score, "explanation": "Concise evidence-based explanation."} for field in fields}}
    return result


class HttpResponse:
    status_code = 200
    headers: dict[str, str] = {}


class Llama32SinglePassTests(unittest.TestCase):
    def test_selector_routes_only_active_models(self) -> None:
        self.assertEqual(selected_model_names("Llama 3.2 90B Vision"), ["Llama 3.2 90B Vision"])
        self.assertEqual(selected_model_names("Both"), ["Gemma", "Llama 3.2 90B Vision"])
        with self.assertRaisesRegex(Exception, "Unknown model selection"):
            selected_model_names("Qwen 3.6 27B")

    def test_prompt_has_one_rubric_and_no_score_anchor(self) -> None:
        prompt = grade_llama.build_prompt("map.pdf")
        self.assertEqual(prompt.count(grade_llama.SPRING_2025_RUBRIC), 1)
        self.assertNotIn('"score": 1', prompt)

    def test_nvidia_payload_uses_image_and_expected_model(self) -> None:
        captured: dict = {}
        def post(_client, payload):
            captured["payload"] = payload
            return object()
        with patch.object(grade_llama, "_post_nvidia", post):
            grade_llama.request_grade(object(), "rubric", "encoded-image")
        payload = captured["payload"]
        self.assertEqual(payload["model"], "meta/llama-3.2-90b-vision-instruct")
        self.assertEqual(payload["max_tokens"], 1800)
        self.assertEqual(payload["temperature"], 0.2)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64,encoded-image")

    def test_nvidia_key_and_compact_references_are_used(self) -> None:
        self.assertEqual(grade_llama.API_KEY_ENV, "NVIDIA_API_KEY")
        summary = grade_llama._compress_reference_materials([
            {"filename": "case.pdf", "text": "Patient has pelvic pain.\nCopyright 2026\nPoll: choose one\nLearning objective: explain physiology."}
        ])
        self.assertIn("Patient has pelvic pain.", summary)
        self.assertIn("Learning objective", summary)
        self.assertNotIn("Copyright", summary)
        self.assertNotIn("Poll:", summary)

    def test_grade_uses_one_request_and_never_scores_in_python(self) -> None:
        calls = []
        response = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_payload(3))}}], "usage": {}},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        def render(_pdf, output):
            output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"jpeg")
            return {"path": output, "base64": "image", "width": 10, "height": 10, "bytes": 4, "render_matrix": [1, 1], "max_width_px": 1400, "jpeg_quality": 80}
        def request(_client, _prompt, _image):
            calls.append(1); return response
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(grade_llama, "render_pdf_first_page", render), patch.object(grade_llama, "create_client", return_value=object()), patch.object(grade_llama, "request_grade", request):
                result = grade_llama.grade_pdf(Path(temp) / "map.pdf", "map.pdf", Path(temp) / "debug")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["debug"]["request_count"], 1)
        self.assertEqual(json.loads(result["cleaned_text"])["knowledge_acquisition"]["basic_science"]["score"], 3)

    def test_invalid_or_missing_scores_fail_validation(self) -> None:
        missing = valid_payload(); del missing["knowledge_acquisition"]["basic_science"]["score"]
        with self.assertRaises(MalformedResultError): parse_model_json(json.dumps(missing))
        invalid = valid_payload(); invalid["knowledge_acquisition"]["basic_science"]["score"] = 5
        with self.assertRaises(MalformedResultError): parse_model_json(json.dumps(invalid))

    def test_transient_nvidia_retry_repeats_only_the_llama_request(self) -> None:
        calls = 0
        response = grade_llama.NvidiaChatCompletion(data={"choices": []}, http_response=HttpResponse(), transport={"http_status": 200})
        def request():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise grade_llama.NvidiaHttpError("NVIDIA NIM HTTP 503", {"http_status": 503})
            return response
        with patch.object(grade_llama.time, "sleep") as sleep:
            _, debug = grade_llama._request_with_retry(request)
        self.assertEqual(calls, 2)
        sleep.assert_called_once_with(2)
        self.assertTrue(debug["retry_attempted"])

    def test_gemma_is_unchanged(self) -> None:
        self.assertEqual(grade_gemma.MODEL, "google/gemma-4-26b-a4b-it:free")

    def test_reports_use_the_llama32_label(self) -> None:
        self.assertEqual(
            generate_evaluation_report.MODEL_KEYS["Llama 3.2 90B Vision"],
            "llama32_90b_vision",
        )


if __name__ == "__main__":
    unittest.main()
