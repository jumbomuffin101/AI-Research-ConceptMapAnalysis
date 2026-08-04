from __future__ import annotations

import copy
import json
import unittest

from consensus.schemas import validate_complete_grading
from grading import grade_gemma, grade_llama
from interface.grading_runner import CATEGORY_FIELDS, parse_model_json


REMOVED_VISUAL_FIELDS = (
    "supporting_evidence",
    "missing_evidence",
    "criterion_confidence",
    "visual_summary",
    "multimodal_feedback",
    "learning_feedback",
    "bbox",
    "evidence_from_map",
)


def compact_grading(score: int = 3, overall: str = "Yes") -> dict:
    result = {
        "map_file": "map.pdf",
        "model": "test-model",
        "overall_meets_expectations": overall,
        "strengths": ["Clear integration."],
        "areas_for_improvement": ["Expand transfer."],
        "grading_notes": "Concise grading note.",
    }
    for domain, fields in CATEGORY_FIELDS.items():
        result[domain] = {
            field: {"score": score, "explanation": "Concise evidence-based sentence."}
            for field in fields
        }
        result[domain]["overall_decision"] = overall
        result[domain]["if_no_explanation"] = (
            "" if overall == "Yes" else "The domain is substantially incomplete."
        )
    return result


class CompactGradingSchemaTests(unittest.TestCase):
    def test_gemma_and_llama_prompts_request_only_compact_grading(self) -> None:
        for prompt in (grade_gemma.build_prompt("map.pdf"), grade_llama.build_prompt("map.pdf")):
            for removed in REMOVED_VISUAL_FIELDS:
                self.assertNotIn(removed, prompt)
            self.assertIn('"score"', prompt)
            self.assertIn('"explanation"', prompt)
        self.assertEqual(grade_gemma.MAX_TOKENS, 1800)
        self.assertEqual(grade_llama.MAX_TOKENS, 1800)

    def test_all_fifteen_scores_and_explanations_remain_required(self) -> None:
        result = compact_grading()
        parsed = parse_model_json(json.dumps(result), normalize_decisions=True)
        self.assertEqual(sum(len(fields) for fields in CATEGORY_FIELDS.values()), 15)
        for domain, fields in CATEGORY_FIELDS.items():
            for field in fields:
                self.assertEqual(set(parsed[domain][field]), {"score", "explanation"})

        incomplete = copy.deepcopy(result)
        del incomplete["integration"]["illness_scripts"]
        with self.assertRaises(Exception):
            parse_model_json(json.dumps(incomplete), normalize_decisions=True)

    def test_provider_extras_are_excluded_without_changing_scores(self) -> None:
        result = compact_grading(4)
        result["knowledge_acquisition"]["basic_science"]["bbox"] = [0, 0, 1, 1]
        result["multimodal_feedback"] = {"unexpected": True}
        parsed = parse_model_json(json.dumps(result), normalize_decisions=True)
        self.assertEqual(parsed["knowledge_acquisition"]["basic_science"]["score"], 4)
        self.assertNotIn("bbox", parsed["knowledge_acquisition"]["basic_science"])
        self.assertNotIn("multimodal_feedback", parsed)

    def test_consensus_review_validation_uses_compact_schema(self) -> None:
        grading = compact_grading()
        validated = validate_complete_grading(grading)
        self.assertEqual(validated, grading)
        for domain, fields in CATEGORY_FIELDS.items():
            for field in fields:
                self.assertEqual(set(validated[domain][field]), {"score", "explanation"})

    def test_evidence_only_recovery_entry_points_are_removed(self) -> None:
        self.assertFalse(hasattr(grade_gemma, "recover_multimodal_evidence"))
        self.assertFalse(hasattr(grade_llama, "recover_multimodal_evidence"))


if __name__ == "__main__":
    unittest.main()
