from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from grading import grade_gemma
from interface.consensus_integration import consensus_ready, immutable_initial_results
from interface.grading_runner import EvaluationResult, parse_model_json


def complete_grading(score: int = 3) -> dict:
    result = {
        "map_file": "map.pdf",
        "model": grade_gemma.MODEL,
        "overall_meets_expectations": "Yes",
        "strengths": ["Clear synthesis."],
        "areas_for_improvement": ["Expand transfer."],
        "grading_notes": "Complete evaluation.",
    }
    for group, fields in grade_gemma.CATEGORY_FIELDS.items():
        result[group] = {
            field: {"score": score, "explanation": f"Visible support for {field}."}
            for field in fields
        }
        result[group]["overall_decision"] = "Yes"
        result[group]["if_no_explanation"] = ""
    return result


class FakeResponse:
    def __init__(self, content: str, *, finish_reason: str = "stop") -> None:
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
        self._content = content
        self._finish_reason = finish_reason

    def model_dump(self, **_kwargs):
        return {
            "id": "response-id",
            "choices": [
                {
                    "message": {"content": self._content},
                    "finish_reason": self._finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 500,
                "total_tokens": 600,
            },
        }


def malformed_complete_text(data: dict) -> str:
    text = json.dumps(data, indent=2)
    marker = '\n  "overall_meets_expectations"'
    return text.replace("," + marker, marker, 1)


class GemmaFormatRepairTests(unittest.TestCase):
    def _render(self, _pdf: Path, output: Path) -> str:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"jpeg")
        return "encoded-image"

    def _grade(
        self,
        initial_text: str,
        repair_text: str | None = None,
        progress: list[str] | None = None,
        finish_reason: str = "stop",
    ):
        initial_response = FakeResponse(initial_text, finish_reason=finish_reason)
        repair_response = FakeResponse(repair_text) if repair_text is not None else None
        repair_mock = Mock(return_value=repair_response)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(grade_gemma, "render_pdf_first_page", self._render), \
                 patch.object(grade_gemma, "create_client", return_value=object()), \
                 patch.object(grade_gemma, "request_grade", return_value=initial_response), \
                 patch.object(grade_gemma, "request_format_repair", repair_mock):
                result = grade_gemma.grade_pdf(
                    root / "map.pdf",
                    "map.pdf",
                    root / "debug" / "run",
                    progress_callback=(progress.append if progress is not None else None),
                )
                raw_path_contents = Path(result["raw_path"]).read_text(encoding="utf-8")
        return result, repair_mock, raw_path_contents

    def test_valid_json_bypasses_repair(self) -> None:
        raw = json.dumps(complete_grading())
        result, repair, _ = self._grade(raw)
        repair.assert_not_called()
        self.assertEqual(result["debug"]["final_result_source"], "initial")
        self.assertTrue(result["debug"]["initial_json_parse_success"])

    def test_malformed_complete_response_gets_one_text_only_repair(self) -> None:
        original = complete_grading()
        malformed = malformed_complete_text(original)
        progress: list[str] = []
        result, repair, raw_saved = self._grade(
            malformed,
            json.dumps(original),
            progress,
        )
        repair.assert_called_once()
        args = repair.call_args.args
        self.assertEqual(args[1], malformed)
        self.assertEqual(args[2], "map.pdf")
        self.assertNotIn("encoded-image", args)
        self.assertEqual(result["debug"]["initial_score_count"], 15)
        self.assertTrue(result["debug"]["scores_preserved"])
        self.assertTrue(result["debug"]["decisions_preserved"])
        self.assertFalse(result["debug"]["format_repair_image_resent"])
        self.assertEqual(result["debug"]["final_result_source"], "format_repair")
        self.assertEqual(json.loads(raw_saved), original)
        self.assertTrue(any("Repairing the response format" in item for item in progress))

    def test_incomplete_malformed_response_cannot_use_formatter(self) -> None:
        original = complete_grading()
        del original["integration"]["illness_scripts"]
        malformed = malformed_complete_text(original)
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(grade_gemma, "render_pdf_first_page", self._render), \
             patch.object(grade_gemma, "create_client", return_value=object()), \
             patch.object(grade_gemma, "request_grade", return_value=FakeResponse(malformed)), \
             patch.object(grade_gemma, "request_format_repair") as repair:
            with self.assertRaises(grade_gemma.MalformedGemmaJsonError) as raised:
                grade_gemma.grade_pdf(
                    Path(temp_dir) / "map.pdf",
                    "map.pdf",
                    Path(temp_dir) / "debug" / "run",
                )
        repair.assert_not_called()
        self.assertEqual(
            raised.exception.attempts["gemma_response_classification"],
            "incomplete_grading_failure",
        )
        self.assertFalse(raised.exception.attempts["format_repair_attempted"])

    def test_repair_cannot_change_scores(self) -> None:
        original = complete_grading(3)
        changed = copy.deepcopy(original)
        changed["knowledge_acquisition"]["basic_science"]["score"] = 4
        malformed = malformed_complete_text(original)
        with self.assertRaises(grade_gemma.MalformedGemmaJsonError) as raised:
            self._grade(malformed, json.dumps(changed))
        self.assertFalse(raised.exception.attempts["scores_preserved"])
        self.assertEqual(raised.exception.attempts["final_result_source"], "failure")

    def test_repair_cannot_change_decisions(self) -> None:
        original = complete_grading()
        changed = copy.deepcopy(original)
        changed["transfer"]["overall_decision"] = "No"
        changed["transfer"]["if_no_explanation"] = "Transfer was incomplete."
        malformed = malformed_complete_text(original)
        with self.assertRaises(grade_gemma.MalformedGemmaJsonError) as raised:
            self._grade(malformed, json.dumps(changed))
        self.assertTrue(raised.exception.attempts["scores_preserved"])
        self.assertFalse(raised.exception.attempts["decisions_preserved"])

    def test_length_finish_reason_is_not_format_repair_eligible(self) -> None:
        raw = malformed_complete_text(complete_grading())
        diagnostic = grade_gemma.classify_grading_response(
            raw,
            FakeResponse(raw, finish_reason="length"),
        )
        self.assertEqual(
            diagnostic["gemma_response_classification"],
            "truncated_grading_failure",
        )
        self.assertFalse(diagnostic["format_repair_eligible"])

    def test_truncated_response_does_not_call_format_repair(self) -> None:
        raw = malformed_complete_text(complete_grading())
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(grade_gemma, "render_pdf_first_page", self._render), \
             patch.object(grade_gemma, "create_client", return_value=object()), \
             patch.object(
                 grade_gemma,
                 "request_grade",
                 return_value=FakeResponse(raw, finish_reason="length"),
             ), \
             patch.object(grade_gemma, "request_format_repair") as repair:
            with self.assertRaises(grade_gemma.MalformedGemmaJsonError) as raised:
                grade_gemma.grade_pdf(
                    Path(temp_dir) / "map.pdf",
                    "map.pdf",
                    Path(temp_dir) / "debug" / "run",
                )
        repair.assert_not_called()
        self.assertIn("truncated before the complete JSON", str(raised.exception))
        self.assertFalse(raised.exception.attempts["format_repair_attempted"])

    def test_format_repair_request_contains_no_image_block(self) -> None:
        create = Mock(return_value=FakeResponse("{}"))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        grade_gemma.request_format_repair(client, "malformed", "map.pdf")
        messages = create.call_args.kwargs["messages"]
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertTrue(all(isinstance(item["content"], str) for item in messages))
        self.assertNotIn("image_url", json.dumps(messages))
        self.assertNotIn('"score":1', json.dumps(messages))

    def test_successful_repair_is_a_valid_immutable_initial_for_consensus(self) -> None:
        original = complete_grading()
        result, _, _ = self._grade(
            malformed_complete_text(original),
            json.dumps(original),
        )
        parsed = parse_model_json(result["cleaned_text"], normalize_decisions=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "request.jpg"
            image.write_bytes(b"jpeg")
            outcomes = [
                EvaluationResult("Gemma", grade_gemma.MODEL, parsed, image, source_image_path=image),
                EvaluationResult(
                    "Llama 3.2 90B Vision",
                    "llama-model",
                    copy.deepcopy(parsed),
                    image,
                    source_image_path=image,
                ),
            ]
            self.assertTrue(consensus_ready(outcomes))
            immutable = immutable_initial_results(outcomes)
        self.assertEqual(immutable["gemma"], original)


if __name__ == "__main__":
    unittest.main()
