from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from grading.multimodal_feedback import (
    CRITERION_FIELDS,
    merge_recovered_evidence,
    normalize_multimodal_numbers,
    validate_multimodal_feedback,
)
from grading import grade_gemma, grade_llama
from interface import grading_runner
from interface.evidence_renderer import (
    OverlayItem,
    normalized_to_pixels,
    render_evidence_overlay,
)
from interface.grading_runner import EvaluationResult, parse_model_json, run_evaluation
from interface.result_display import category_rows


def core_grading(score: int = 3) -> dict:
    result = {
        "map_file": "map.pdf",
        "model": "test/model",
        "overall_meets_expectations": "Yes",
        "strengths": ["Connected reasoning."],
        "areas_for_improvement": ["Clarify one relationship."],
        "grading_notes": "Model-generated grading.",
    }
    for domain, criteria in CRITERION_FIELDS.items():
        result[domain] = {
            criterion: {"score": score, "explanation": "Visible evidence supports this score."}
            for criterion in criteria
        }
        result[domain]["overall_decision"] = "Yes"
        result[domain]["if_no_explanation"] = ""
    return result


def recovery_payload(*, bbox=None, confidence: float = 0.9) -> dict:
    result: dict = {}
    for domain, criteria in CRITERION_FIELDS.items():
        result[domain] = {}
        for criterion in criteria:
            result[domain][criterion] = {
                "supporting_evidence": [
                    {
                        "evidence_text": "Visible linked concepts.",
                        "location_description": "Upper-left map region.",
                        "bbox": bbox,
                        "relationship_type": "visible concept relationship",
                        "confidence": confidence,
                    }
                ],
                "missing_evidence": [
                    {
                        "missing_relationship": "One bridge is incomplete.",
                        "suggested_connection": "Connect finding to mechanism.",
                        "importance": "moderate",
                    }
                ],
                "criterion_confidence": confidence,
                "human_review_recommended": confidence < 0.60,
            }
        result[domain]["visual_summary"] = {
            "strongest_visible_evidence": ["Visible linked concepts."],
            "most_important_missing_connection": "One bridge is incomplete.",
            "domain_confidence": confidence,
            "human_review_recommended": confidence < 0.60,
        }
    result["multimodal_feedback"] = {
        "strongest_regions": [
            {"description": "Connected region.", "bbox": bbox, "confidence": confidence}
        ],
        "highest_priority_improvements": [
            {
                "current_state": "Two concepts are linked.",
                "missing_bridge": "The mechanism is incomplete.",
                "suggested_revision": "Connect finding to mechanism.",
                "bbox": bbox,
                "importance": "major",
            }
        ],
        "overall_visual_confidence": confidence,
        "human_review_recommended": confidence < 0.60,
    }
    result["learning_feedback"] = [
        {
            "criterion": "patient_data_pathophysiology",
            "observed_evidence": "A finding is linked to the diagnosis.",
            "guiding_question": "What mechanism explains this finding?",
            "hint": "Think about the affected physiology.",
            "bbox": bbox,
            "confidence": confidence,
        }
    ]
    return result


def complete_result(*, bbox=None, confidence: float = 0.9) -> dict:
    merged, _ = merge_recovered_evidence(
        core_grading(),
        recovery_payload(bbox=bbox, confidence=confidence),
    )
    return merged


class MultimodalSchemaTests(unittest.TestCase):
    def test_both_initial_prompts_request_grounded_feedback(self) -> None:
        for prompt in (
            grade_gemma.build_prompt("map.pdf"),
            grade_llama.build_prompt("map.pdf"),
        ):
            self.assertIn("GROUNDED MULTIMODAL FEEDBACK", prompt)
            self.assertIn("supporting_evidence", prompt)
            self.assertIn("normalized [x_min,y_min,x_max,y_max]", prompt)
            self.assertIn("Presence is not demonstration", prompt)
            self.assertIn("learning_feedback", prompt)

    def test_llama_formatter_does_not_require_or_invent_multimodal_fields(self) -> None:
        import inspect

        source = inspect.getsource(grade_llama.request_format_repair)
        self.assertIn("_core_schema_template()", source)
        self.assertNotIn("evidence_recovery_template", source)

    def test_valid_normalized_and_null_bbox(self) -> None:
        for bbox in ([0.1, 0.2, 0.7, 0.8], None):
            result = complete_result(bbox=bbox)
            validation = validate_multimodal_feedback(result)
            self.assertTrue(validation.complete)

    def test_malformed_bbox_does_not_invalidate_core_grading(self) -> None:
        result = complete_result(bbox=[0.8, 0.2, 0.1, 0.7])
        self.assertEqual(
            parse_model_json(json.dumps(result))["integration"]["illness_scripts"]["score"],
            3,
        )
        validation = validate_multimodal_feedback(result)
        self.assertFalse(validation.complete)
        self.assertGreater(validation.invalid_bbox_count, 0)

    def test_out_of_range_confidence_and_invalid_importance_fail_visual_validation(self) -> None:
        result = complete_result()
        result["integration"]["illness_scripts"]["criterion_confidence"] = 1.2
        result["integration"]["illness_scripts"]["missing_evidence"][0]["importance"] = "urgent"
        validation = validate_multimodal_feedback(result)
        self.assertFalse(validation.complete)
        self.assertTrue(any("criterion_confidence" in warning for warning in validation.warnings))
        self.assertTrue(any("invalid importance" in warning for warning in validation.warnings))

    def test_omitted_evidence_arrays_are_incomplete_not_fabricated(self) -> None:
        result = complete_result()
        del result["application"]["patient_data_pathophysiology"]["supporting_evidence"]
        validation = validate_multimodal_feedback(result)
        self.assertFalse(validation.complete)
        self.assertIn(
            "application.patient_data_pathophysiology.supporting_evidence",
            validation.missing_fields,
        )
        self.assertNotIn(
            "supporting_evidence",
            result["application"]["patient_data_pathophysiology"],
        )

    def test_unambiguous_multimodal_number_strings_are_format_normalized(self) -> None:
        result = complete_result()
        item = result["knowledge_acquisition"]["basic_science"]
        item["criterion_confidence"] = "0.82"
        item["supporting_evidence"][0]["bbox"] = ["0.1", "0.2", "0.7", "0.8"]
        changes = normalize_multimodal_numbers(result)
        self.assertEqual(item["criterion_confidence"], 0.82)
        self.assertEqual(item["supporting_evidence"][0]["bbox"], [0.1, 0.2, 0.7, 0.8])
        self.assertGreaterEqual(len(changes), 2)

    def test_recovery_cannot_change_scores_or_decisions(self) -> None:
        original = core_grading(4)
        recovered = recovery_payload()
        recovered["knowledge_acquisition"]["basic_science"]["score"] = 1
        recovered["knowledge_acquisition"]["overall_decision"] = "No"
        recovered["overall_meets_expectations"] = "No"
        merged, ignored = merge_recovered_evidence(original, recovered)
        self.assertEqual(merged["knowledge_acquisition"]["basic_science"]["score"], 4)
        self.assertEqual(merged["knowledge_acquisition"]["overall_decision"], "Yes")
        self.assertEqual(merged["overall_meets_expectations"], "Yes")
        self.assertIn("knowledge_acquisition.basic_science.score", ignored)
        self.assertIn("knowledge_acquisition.overall_decision", ignored)
        self.assertIn("overall_meets_expectations", ignored)


class EvidenceRendererTests(unittest.TestCase):
    def test_coordinate_conversion_and_render_preserve_source(self) -> None:
        from PIL import Image

        self.assertEqual(normalized_to_pixels([0.1, 0.2, 0.5, 0.8], 100, 50), (10, 10, 50, 40))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.jpg"
            Image.new("RGB", (100, 50), "white").save(source, "JPEG")
            before = source.read_bytes()
            rendered, warnings = render_evidence_overlay(
                source,
                [
                    OverlayItem([0.1, 0.2, 0.5, 0.8], "support", 1),
                    OverlayItem(None, "support", 2),
                    OverlayItem([1.2, 0.1, 1.3, 0.5], "improvement", 3),
                ],
            )
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(rendered.size, (100, 50))
            self.assertEqual(len(warnings), 1)


class EvidenceRecoveryIntegrationTests(unittest.TestCase):
    def test_runner_recovers_once_resends_image_and_preserves_grading(self) -> None:
        calls = {"recovery": 0}

        def grade_pdf(_pdf, _map_file, debug_prefix, **_kwargs):
            image_path = Path(f"{debug_prefix}_request.jpg")
            image_path.write_bytes(b"jpeg")
            return {
                "cleaned_text": json.dumps(core_grading(3)),
                "response": {"initial": True},
                "image_base64": "same-image",
                "image_path": image_path,
                "debug": {},
            }

        def recover(image, original, _progress=None):
            calls["recovery"] += 1
            self.assertEqual(image, "same-image")
            self.assertEqual(original["integration"]["illness_scripts"]["score"], 3)
            return {
                "raw_text": json.dumps(recovery_payload(bbox=[0.1, 0.1, 0.4, 0.4])),
                "cleaned_text": json.dumps(recovery_payload(bbox=[0.1, 0.1, 0.4, 0.4])),
            }

        fake = SimpleNamespace(
            MODEL="test/model",
            PROVIDER="Test",
            grade_pdf=grade_pdf,
            recover_multimodal_evidence=recover,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "map.pdf"
            pdf.write_bytes(b"pdf")
            with patch.dict(grading_runner.MODEL_MODULES, {"Gemma": fake}), \
                 patch.object(grading_runner, "OUTPUT_DIR", root / "output"), \
                 patch.object(grading_runner, "DEBUG_DIR", root / "debug"), \
                 patch.object(grading_runner, "FAILURE_EVALUATION_DIR", root / "failures"):
                outcomes = run_evaluation(pdf, ["Gemma"], "map.pdf", learning_mode=True)
        self.assertEqual(calls["recovery"], 1)
        self.assertIsInstance(outcomes[0], EvaluationResult)
        self.assertTrue(outcomes[0].multimodal_available)
        self.assertTrue(outcomes[0].learning_mode)
        self.assertEqual(outcomes[0].data["integration"]["illness_scripts"]["score"], 3)

    def test_invalid_recovery_is_attempted_only_once_and_grading_survives(self) -> None:
        calls = 0

        def grade_pdf(_pdf, _map_file, debug_prefix, **_kwargs):
            image_path = Path(f"{debug_prefix}_request.jpg")
            image_path.write_bytes(b"jpeg")
            return {
                "cleaned_text": json.dumps(core_grading()),
                "response": {},
                "image_base64": "same-image",
                "image_path": image_path,
                "debug": {},
            }

        def recover(*_args):
            nonlocal calls
            calls += 1
            return {"raw_text": "{}", "cleaned_text": "{}"}

        fake = SimpleNamespace(
            MODEL="test/model",
            PROVIDER="Test",
            grade_pdf=grade_pdf,
            recover_multimodal_evidence=recover,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "map.pdf"
            pdf.write_bytes(b"pdf")
            with patch.dict(grading_runner.MODEL_MODULES, {"Gemma": fake}), \
                 patch.object(grading_runner, "OUTPUT_DIR", root / "output"), \
                 patch.object(grading_runner, "DEBUG_DIR", root / "debug"), \
                 patch.object(grading_runner, "FAILURE_EVALUATION_DIR", root / "failures"):
                outcomes = run_evaluation(pdf, ["Gemma"], "map.pdf")
        self.assertEqual(calls, 1)
        self.assertIsInstance(outcomes[0], EvaluationResult)
        self.assertFalse(outcomes[0].multimodal_available)
        self.assertEqual(outcomes[0].data["knowledge_acquisition"]["basic_science"]["score"], 3)

    def test_learning_mode_rows_hide_scores_without_mutating_grading(self) -> None:
        result = core_grading(4)
        before = copy.deepcopy(result)
        rows = category_rows(
            "knowledge_acquisition",
            result["knowledge_acquisition"],
            learning_mode=True,
        )
        self.assertTrue(all("Score" not in row for row in rows))
        self.assertEqual(result, before)
        self.assertTrue(recovery_payload()["learning_feedback"][0]["guiding_question"])


if __name__ == "__main__":
    unittest.main()
