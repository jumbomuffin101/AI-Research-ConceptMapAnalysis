"""Regression coverage for NVIDIA Llama 3.2 90B Vision grading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from grading import grade_gemma, grade_llama
from interface import grading_runner
from interface.grading_runner import EvaluationResult, MalformedResultError, parse_model_json, run_evaluation, selected_model_names
from interface.consensus_integration import consensus_ready
from scripts import generate_evaluation_report


def valid_payload(score: int = 2) -> dict:
    result = {"map_file": "map.pdf", "model": grade_llama.MODEL, "overall_meets_expectations": "Yes", "strengths": ["Visible concepts are connected to the working diagnosis."], "areas_for_improvement": ["Make one pathophysiology connection more explicit."], "grading_notes": ""}
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

    def test_prompt_calibrates_three_vs_four_without_weakening_one_or_two(self) -> None:
        prompt = grade_llama.build_prompt("map.pdf")
        self.assertIn("A score of 4 does not require perfection", prompt)
        self.assertIn("meaningful visible limitation", prompt)
        self.assertIn("use 2 for partial, superficial", prompt)
        self.assertIn("use 1 for absent, largely incorrect", prompt)
        self.assertIn("do not treat reference material as student evidence", prompt)

    def test_prompt_calibrates_holistic_yes_no_without_score_thresholds(self) -> None:
        prompt = grade_llama.build_prompt("map.pdf")
        self.assertIn("The final Yes/No decision is holistic", prompt)
        self.assertIn("not an average, threshold, or automatic result", prompt)
        self.assertIn("Return No only for a substantial map-level deficiency", prompt)
        self.assertIn("A successful map may still have areas_for_improvement", prompt)

    def test_prompt_requires_demonstrated_relationships_for_high_scores_and_passing(self) -> None:
        prompt = grade_llama.build_prompt("map.pdf")
        self.assertIn("A concept being present does not itself demonstrate a relationship", prompt)
        self.assertIn("scores of 3 or 4 require meaningful visible relationships", prompt)
        self.assertIn("Prioritized DDx requires multiple plausible diagnoses visibly ranked", prompt)
        self.assertIn("verify every 3 or 4 has specific visible support", prompt)
        self.assertIn("anti-inflation review", prompt)

    def test_prompt_forbids_proximity_and_keyword_inference(self) -> None:
        prompt = grade_llama.build_prompt("map.pdf")
        self.assertIn("Do not infer relationships because concepts are near one another", prompt)
        self.assertIn("do not assume prioritization because one diagnosis is present", prompt)
        self.assertIn("integration merely because arrows exist", prompt)
        self.assertIn("When uncertain whether a relationship is demonstrated, use the lower score", prompt)
        retry = grade_llama.build_full_retry_prompt("map.pdf")
        self.assertIn("Do not infer required relationships from proximity", retry)

    def test_prompt_allows_holistic_domains_without_weakening_list_based_map_rules(self) -> None:
        prompt = grade_llama.build_prompt("map.pdf")
        self.assertIn("A domain decision is holistic", prompt)
        self.assertIn("health-system science and determinants of health support the domain", prompt)
        self.assertIn("a numbered DDx list is not required", prompt)
        self.assertIn("incomplete-but-substantial evidence is 3, not 2", prompt)
        self.assertIn("Terminology, density, and lists without meaningful relationships remain insufficient", prompt)
        retry = grade_llama.build_full_retry_prompt("map.pdf")
        self.assertIn("Domain decisions are holistic", retry)
        self.assertIn("require meaningful visible relationships", retry)

    def test_prompt_distinguishes_incomplete_from_inadequate_evidence(self) -> None:
        prompt = grade_llama.build_prompt("map.pdf")
        self.assertIn("Do not assign 2 merely because a criterion is not fully comprehensive", prompt)
        self.assertIn("'Does not fully explain' usually indicates 3, not 2", prompt)
        self.assertIn("patient-specific finding-to-mechanism links", prompt)
        self.assertIn("Transfer need not be labeled 'previously learned.'", prompt)
        self.assertIn("Evidence quality", prompt)
        self.assertIn("not evidence perfection", prompt)
        retry = grade_llama.build_full_retry_prompt("map.pdf")
        self.assertIn("Incomplete-but-substantial evidence is 3, not 2", retry)

    def test_full_retry_prompt_is_compact_and_contains_every_criterion(self) -> None:
        initial = grade_llama.build_prompt("map.pdf")
        retry = grade_llama.build_full_retry_prompt("map.pdf")
        self.assertLess(len(retry), len(initial))
        self.assertLess(len(retry), 8000)
        self.assertIn("Your previous response was incomplete", retry)
        self.assertNotIn("ANTI-INFLATION REVIEW", retry)
        for group, fields in grade_llama.CATEGORY_FIELDS.items():
            self.assertIn(group, retry)
            for field in fields:
                self.assertIn(field, retry)

    def test_full_retry_uses_streaming_and_its_own_timeout(self) -> None:
        captured: dict = {}
        def post(_client, payload, **kwargs):
            captured["payload"] = payload; captured["kwargs"] = kwargs
            return object()
        with patch.object(grade_llama, "_post_nvidia", post):
            grade_llama.request_complete_grading_retry(object(), "compact prompt", "image", response_format=True)
        self.assertTrue(captured["payload"]["stream"])
        self.assertTrue(captured["kwargs"]["stream"])
        self.assertEqual(captured["kwargs"]["timeout"], (30, 300))
        self.assertEqual(captured["payload"]["max_tokens"], grade_llama.FULL_RETRY_MAX_TOKENS)

    def test_streaming_chunks_are_collected_into_completion_content(self) -> None:
        class StreamResponse:
            status_code = 200
            headers: dict[str, str] = {}
            text = ""
            def iter_lines(self, decode_unicode=True):
                return iter([
                    'data: {"choices":[{"delta":{"content":"{\\\"a\\\":"},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{"content":"1}"},"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ])
        class Requests:
            def post(self, *_args, **_kwargs): return StreamResponse()
        completion = grade_llama._post_nvidia(
            {"requests": Requests(), "headers": {}}, {"stream": True}, stream=True, timeout=(30, 300)
        )
        self.assertEqual(completion.choices[0]["message"]["content"], '{"a":1}')
        self.assertTrue(completion.transport["streaming_enabled"])
        self.assertEqual(completion.transport["read_timeout_seconds"], 300)

    def test_decision_debug_metadata_only_reports_model_values(self) -> None:
        payload = valid_payload(3)
        payload["knowledge_acquisition"]["basic_science"]["score"] = 4
        payload["integration"]["overall_decision"] = "No"
        metadata = grade_llama._decision_debug_metadata(payload)
        self.assertEqual(metadata["domain_decisions"]["integration"], "No")
        self.assertEqual(metadata["overall_decision"], "Yes")
        self.assertIn("knowledge_acquisition.basic_science", metadata["criteria_scored_4"])
        self.assertIn("integration.illness_scripts", metadata["criteria_scored_3"])
        self.assertEqual(metadata["overall_no_substantial_deficiency_identified"], "not independently assessed by Python")

    def test_nvidia_payload_uses_image_and_expected_model(self) -> None:
        captured: dict = {}
        def post(_client, payload, **kwargs):
            captured["payload"] = payload; captured["kwargs"] = kwargs
            return object()
        with patch.object(grade_llama, "_post_nvidia", post):
            grade_llama.request_grade(object(), "rubric", "encoded-image")
        payload = captured["payload"]
        self.assertEqual(payload["model"], "meta/llama-3.2-90b-vision-instruct")
        self.assertEqual(payload["max_tokens"], grade_llama.MAX_TOKENS)
        self.assertEqual(payload["temperature"], 0.2)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64,encoded-image")
        self.assertTrue(payload["stream"])
        self.assertTrue(captured["kwargs"]["stream"])
        self.assertEqual(captured["kwargs"]["timeout"], (30, 300))

    def test_format_repair_payload_is_text_only_and_uses_json_mode_setting(self) -> None:
        captured: dict = {}
        def post(_client, payload, **kwargs):
            captured["payload"] = payload; captured["kwargs"] = kwargs
            return object()
        with patch.object(grade_llama, "_post_nvidia", post):
            grade_llama.request_format_repair(object(), "## Markdown evaluation", response_format=True)
        payload = captured["payload"]
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], grade_llama.MAX_TOKENS)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIsInstance(payload["messages"][0]["content"], str)
        self.assertNotIn("image_url", payload["messages"][0]["content"])
        self.assertTrue(payload["stream"])
        self.assertEqual(captured["kwargs"]["timeout"], (30, 300))

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
        def request(_client, _prompt, _image, **_kwargs):
            calls.append(1); return response
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(grade_llama, "render_pdf_first_page", render), patch.object(grade_llama, "create_client", return_value=object()), patch.object(grade_llama, "request_grade", request):
                result = grade_llama.grade_pdf(Path(temp) / "map.pdf", "map.pdf", Path(temp) / "debug")
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["debug"]["request_count"], 1)
        self.assertEqual(json.loads(result["cleaned_text"])["knowledge_acquisition"]["basic_science"]["score"], 3)

    def test_valid_initial_json_does_not_trigger_format_repair(self) -> None:
        response = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_payload(3))}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        def render(_pdf, output):
            output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"jpeg")
            return {"path": output, "base64": "image", "width": 10, "height": 10, "bytes": 4}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(grade_llama, "render_pdf_first_page", render), \
             patch.object(grade_llama, "create_client", return_value=object()), \
             patch.object(grade_llama, "request_grade", return_value=response), \
             patch.object(grade_llama, "request_format_repair") as repair:
            result = grade_llama.grade_pdf(Path(temp) / "map.pdf", "map.pdf", Path(temp) / "debug")
        repair.assert_not_called()
        self.assertEqual(result["debug"]["final_result_source"], "initial")

    def test_markdown_response_uses_one_text_only_repair_and_preserves_scores(self) -> None:
        markdown = "## Evaluation\n" + json.dumps(valid_payload(3))
        initial = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": markdown}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        repaired = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_payload(3))}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        captured: dict = {}
        def render(_pdf, output):
            output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"jpeg")
            return {"path": output, "base64": "image", "width": 10, "height": 10, "bytes": 4}
        def repair(_client, previous, **kwargs):
            captured["previous"] = previous; captured["kwargs"] = kwargs
            return repaired
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(grade_llama, "render_pdf_first_page", render), \
             patch.object(grade_llama, "create_client", return_value=object()), \
             patch.object(grade_llama, "request_grade", return_value=initial), \
             patch.object(grade_llama, "request_format_repair", repair):
            result = grade_llama.grade_pdf(Path(temp) / "map.pdf", "map.pdf", Path(temp) / "debug")
        self.assertTrue(captured["previous"].startswith("## Evaluation"))
        self.assertEqual(result["debug"]["final_result_source"], "repair")
        self.assertFalse(result["debug"]["image_resent_for_repair"])
        self.assertEqual(json.loads(result["cleaned_text"])["integration"]["illness_scripts"]["score"], 3)

    def test_unrecoverable_content_uses_one_full_retry_not_formatter(self) -> None:
        initial = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        retry = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": "still not JSON"}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        def render(_pdf, output):
            output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"jpeg")
            return {"path": output, "base64": "image", "width": 10, "height": 10, "bytes": 4}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(grade_llama, "render_pdf_first_page", render), \
             patch.object(grade_llama, "create_client", return_value=object()), \
             patch.object(grade_llama, "request_grade", return_value=initial), \
             patch.object(grade_llama, "request_complete_grading_retry", return_value=retry) as full_retry, \
             patch.object(grade_llama, "request_format_repair") as repair_call:
            with self.assertRaises(grade_llama.MalformedLlamaVisionJsonError):
                grade_llama.grade_pdf(Path(temp) / "map.pdf", "map.pdf", Path(temp) / "debug")
        repair_call.assert_not_called()
        full_retry.assert_called_once()

    def test_summary_only_json_uses_full_multimodal_retry_not_formatter(self) -> None:
        incomplete = {
            "map_file": "map.pdf", "model": grade_llama.MODEL, "overall_meets_expectations": "Yes",
            "strengths": ["Clear concepts."], "areas_for_improvement": ["Add connections."], "grading_notes": "Summary only.",
        }
        initial = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(incomplete)}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        full_retry = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_payload(3))}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        images: list[str] = []
        def render(_pdf, output):
            output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"jpeg")
            return {"path": output, "base64": "image", "width": 10, "height": 10, "bytes": 4}
        def initial_request(_client, _prompt, image, **_kwargs):
            images.append(image)
            return initial
        def retry_request(_client, _prompt, image, **_kwargs):
            images.append(image)
            return full_retry
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(grade_llama, "render_pdf_first_page", render), \
             patch.object(grade_llama, "create_client", return_value=object()), \
             patch.object(grade_llama, "request_grade", initial_request), \
             patch.object(grade_llama, "request_complete_grading_retry", retry_request), \
             patch.object(grade_llama, "request_format_repair") as formatter:
            result = grade_llama.grade_pdf(Path(temp) / "map.pdf", "map.pdf", Path(temp) / "debug")
        formatter.assert_not_called()
        self.assertEqual(images, ["image", "image"])
        self.assertEqual(result["debug"]["initial_response_classification"], "incomplete_grading_failure")
        self.assertEqual(result["debug"]["initial_original_score_count"], 0)
        self.assertTrue(result["debug"]["initial_summary_only_response"])
        self.assertFalse(result["debug"]["format_repair_eligible"])
        self.assertTrue(result["debug"]["image_resent_for_full_retry"])
        self.assertEqual(result["debug"]["final_result_source"], "full_multimodal_retry")

    def test_missing_one_domain_uses_full_multimodal_retry(self) -> None:
        incomplete = valid_payload(3)
        del incomplete["transfer"]
        inspection = grade_llama.inspect_grading_completeness(json.dumps(incomplete))
        self.assertEqual(inspection["classification"], "incomplete_grading_failure")
        self.assertIn("transfer", inspection["domains_missing"])

    def test_full_retry_timeout_is_transport_failure_not_json_or_format_repair(self) -> None:
        summary = {
            "map_file": "map.pdf", "model": grade_llama.MODEL, "overall_meets_expectations": "Yes",
            "strengths": ["Clear concepts."], "areas_for_improvement": ["Add connections."], "grading_notes": "Summary only.",
        }
        initial = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(summary)}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        def render(_pdf, output):
            output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"jpeg")
            return {"path": output, "base64": "image", "width": 10, "height": 10, "bytes": 4}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(grade_llama, "render_pdf_first_page", render), \
             patch.object(grade_llama, "create_client", return_value=object()), \
             patch.object(grade_llama, "request_grade", return_value=initial), \
             patch.object(grade_llama, "request_complete_grading_retry", side_effect=TimeoutError("read timed out")), \
             patch.object(grade_llama, "request_format_repair") as formatter:
            with self.assertRaisesRegex(
                grade_llama.LlamaNvidiaTimeoutError,
                "did not return a response before the NVIDIA timeout limit",
            ) as caught:
                grade_llama.grade_pdf(Path(temp) / "map.pdf", "map.pdf", Path(temp) / "debug")
        formatter.assert_not_called()
        self.assertEqual(caught.exception.attempts["request_count"], 2)
        self.assertEqual(caught.exception.attempts["retry_read_timeout_seconds"], 360)

    def test_streamlit_public_runner_uses_full_retry_for_summary_only_llama(self) -> None:
        summary = {
            "map_file": "map.pdf", "model": grade_llama.MODEL, "overall_meets_expectations": "Yes",
            "strengths": ["Clear concepts."], "areas_for_improvement": ["Add connections."], "grading_notes": "Summary only.",
        }
        first = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(summary)}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        second = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_payload(3))}}]},
            http_response=HttpResponse(), transport={"http_status": 200},
        )
        sent_images: list[str] = []
        def render(_pdf, output):
            output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"jpeg")
            return {"path": output, "base64": "image", "width": 10, "height": 10, "bytes": 4}
        def first_request(_client, _prompt, image, **_kwargs):
            sent_images.append(image); return first
        def full_retry(_client, _prompt, image, **_kwargs):
            sent_images.append(image); return second
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf_path = root / "map.pdf"; pdf_path.write_bytes(b"pdf")
            with patch.object(grade_llama, "render_pdf_first_page", render), \
                 patch.object(grade_llama, "create_client", return_value=object()), \
                 patch.object(grade_llama, "request_grade", first_request), \
                 patch.object(grade_llama, "request_complete_grading_retry", full_retry), \
                 patch.object(grade_llama, "request_format_repair") as formatter, \
                 patch.object(grading_runner, "OUTPUT_DIR", root / "outputs"), \
                 patch.object(grading_runner, "DEBUG_DIR", root / "debug"), \
                 patch.object(grading_runner, "FAILURE_EVALUATION_DIR", root / "failures"):
                outcomes = run_evaluation(pdf_path, ["Llama 3.2 90B Vision"], "map.pdf")
        formatter.assert_not_called()
        self.assertEqual(sent_images, ["image", "image"])
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], EvaluationResult)

    def test_missing_narrative_fields_trigger_repair_and_preserve_scores_and_decisions(self) -> None:
        for missing_fields in (("strengths",), ("areas_for_improvement",), ("strengths", "areas_for_improvement")):
            initial_payload = valid_payload(3)
            for field in missing_fields:
                del initial_payload[field]
            initial = grade_llama.NvidiaChatCompletion(
                data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(initial_payload)}}]},
                http_response=HttpResponse(), transport={"http_status": 200},
            )
            repaired = grade_llama.NvidiaChatCompletion(
                data={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(valid_payload(3))}}]},
                http_response=HttpResponse(), transport={"http_status": 200},
            )
            def render(_pdf, output):
                output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"jpeg")
                return {"path": output, "base64": "image", "width": 10, "height": 10, "bytes": 4}
            with tempfile.TemporaryDirectory() as temp, \
                 patch.object(grade_llama, "render_pdf_first_page", render), \
                 patch.object(grade_llama, "create_client", return_value=object()), \
                 patch.object(grade_llama, "request_grade", return_value=initial), \
                 patch.object(grade_llama, "request_format_repair", return_value=repaired) as repair_call:
                result = grade_llama.grade_pdf(Path(temp) / "map.pdf", "map.pdf", Path(temp) / "debug")
            repair_call.assert_called_once()
            parsed = json.loads(result["cleaned_text"])
            self.assertIsInstance(parsed["strengths"], list)
            self.assertIsInstance(parsed["areas_for_improvement"], list)
            self.assertEqual(parsed["knowledge_acquisition"]["basic_science"]["score"], 3)
            self.assertEqual(parsed["overall_meets_expectations"], "Yes")

    def test_empty_narrative_arrays_fail_without_python_fallback(self) -> None:
        payload = valid_payload()
        payload["strengths"] = []
        payload["areas_for_improvement"] = []
        with self.assertRaisesRegex(RuntimeError, "strengths.*1 to 3"):
            grade_llama._validate_narrative_fields(payload)
        self.assertEqual(payload["strengths"], [])
        self.assertEqual(payload["areas_for_improvement"], [])

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
        sleep.assert_called_once_with(5)
        self.assertTrue(debug["retry_attempted"])

    def test_read_timeout_retries_once_with_360_second_read_timeout(self) -> None:
        class ReadTimeout(Exception):
            pass

        timeouts = []
        response = grade_llama.NvidiaChatCompletion(
            data={"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
            http_response=HttpResponse(),
            transport={
                "http_status": 200,
                "streaming_enabled": True,
                "time_to_first_token_seconds": 2.5,
                "elapsed_request_seconds": 15.0,
            },
        )

        def request(timeout):
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise ReadTimeout("read timed out")
            return response

        statuses = []
        with patch.object(grade_llama.time, "sleep") as sleep:
            _, debug = grade_llama._request_with_retry(
                request,
                statuses.append,
                stage="initial_grading",
            )
        self.assertEqual(timeouts, [(30, 300), (30, 360)])
        sleep.assert_called_once_with(5)
        self.assertEqual(debug["request_count"], 2)
        self.assertEqual(debug["retry_read_timeout_seconds"], 360)
        self.assertEqual(len(debug["attempts"]), 2)
        self.assertIn("Retrying Llama once", statuses[-1])

    def test_second_read_timeout_uses_precise_failure_message(self) -> None:
        class ReadTimeout(Exception):
            pass

        calls = 0

        def request(_timeout):
            nonlocal calls
            calls += 1
            raise ReadTimeout("read timed out")

        with patch.object(grade_llama.time, "sleep"):
            with self.assertRaises(grade_llama.LlamaNvidiaTimeoutError) as caught:
                grade_llama._request_with_retry(request, stage="initial_grading")
        self.assertEqual(calls, 2)
        self.assertEqual(
            str(caught.exception),
            "Llama 3.2 90B Vision did not return a response before the NVIDIA timeout limit.",
        )
        self.assertEqual(len(caught.exception.attempts["attempts"]), 2)

    def test_each_retryable_http_status_gets_exactly_one_retry(self) -> None:
        for status in (429, 502, 503, 504):
            with self.subTest(status=status):
                calls = 0
                response = grade_llama.NvidiaChatCompletion(
                    data={"choices": []},
                    http_response=HttpResponse(),
                    transport={"http_status": 200},
                )

                def request(_timeout):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise grade_llama.NvidiaHttpError(
                            f"NVIDIA NIM HTTP {status}", {"http_status": status}
                        )
                    return response

                with patch.object(grade_llama.time, "sleep"):
                    grade_llama._request_with_retry(request, stage="initial_grading")
                self.assertEqual(calls, 2)

    def test_schema_error_does_not_trigger_transport_retry(self) -> None:
        calls = 0

        def request(_timeout):
            nonlocal calls
            calls += 1
            raise MalformedResultError("invalid score")

        with self.assertRaises(MalformedResultError):
            grade_llama._request_with_retry(request, stage="initial_grading")
        self.assertEqual(calls, 1)

    def test_successful_llama_transport_retry_preserves_gemma_and_is_consensus_ready(self) -> None:
        class ReadTimeout(Exception):
            pass

        payload = valid_payload(3)
        llama_response = grade_llama.NvidiaChatCompletion(
            data={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(payload)},
                    }
                ],
                "usage": {"completion_tokens": 100},
            },
            http_response=HttpResponse(),
            transport={
                "http_status": 200,
                "streaming_enabled": True,
                "elapsed_request_seconds": 10,
                "time_to_first_token_seconds": 1,
            },
        )
        llama_calls = 0

        def llama_request(*_args, **_kwargs):
            nonlocal llama_calls
            llama_calls += 1
            if llama_calls == 1:
                raise ReadTimeout("read timed out")
            return llama_response

        def render(_pdf, output):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"jpeg")
            return {
                "path": output,
                "base64": "image",
                "width": 10,
                "height": 10,
                "bytes": 4,
            }

        gemma_payload = valid_payload(3)
        gemma_payload["model"] = grade_gemma.MODEL
        gemma_grade = {
            "cleaned_text": json.dumps(gemma_payload),
            "response": {"gemma": True},
            "debug": {},
            "prompt_path": None,
            "raw_path": None,
            "image_path": None,
        }
        gemma_grader = Mock(return_value=gemma_grade)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(grading_runner, "OUTPUT_DIR", Path(temp) / "outputs"), \
             patch.object(grading_runner, "DEBUG_DIR", Path(temp) / "debug"), \
             patch.object(grading_runner, "FAILURE_EVALUATION_DIR", Path(temp) / "failures"), \
             patch.object(grade_gemma, "grade_pdf", gemma_grader), \
             patch.object(grade_llama, "render_pdf_first_page", render), \
             patch.object(grade_llama, "create_client", return_value=object()), \
             patch.object(grade_llama, "request_grade", llama_request), \
             patch.object(grade_llama.time, "sleep"):
            results = run_evaluation(
                Path(temp) / "map.pdf",
                ["Gemma", "Llama 3.2 90B Vision"],
                "map.pdf",
            )
        gemma_grader.assert_called_once()
        self.assertEqual(llama_calls, 2)
        self.assertTrue(consensus_ready(results))

    def test_gemma_is_unchanged(self) -> None:
        self.assertEqual(grade_gemma.MODEL, "google/gemma-4-26b-a4b-it:free")

    def test_reports_use_the_llama32_label(self) -> None:
        self.assertEqual(
            generate_evaluation_report.MODEL_KEYS["Llama 3.2 90B Vision"],
            "llama32_90b_vision",
        )


if __name__ == "__main__":
    unittest.main()
