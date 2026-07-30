from __future__ import annotations

import base64
import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path

# UI helpers are pure for these tests; Streamlit itself is not required.
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))

from interface.consensus_display import comparison_rows
from interface.consensus_integration import (
    consensus_ready,
    exact_request_image_inputs,
    fallback_comparison_export,
    immutable_initial_results,
)
from interface.grading_runner import EvaluationFailure, EvaluationResult


CRITERIA = {
    "knowledge_acquisition": (
        "basic_science",
        "health_system_science",
        "clinical_science",
        "patient_case_information",
        "determinants_of_health",
    ),
    "integration": (
        "prioritized_differential_diagnosis",
        "illness_scripts",
        "basic_to_foundational_science",
        "patient_data_to_clinical_information",
        "patient_data_to_basic_science",
    ),
    "application": (
        "working_diagnosis_pathophysiology",
        "patient_data_pathophysiology",
    ),
    "transfer": (
        "prior_basic_science",
        "prior_clinical_concepts",
        "deepens_understanding",
    ),
}


def grading(score: int = 3) -> dict:
    value = {
        "map_file": "map.pdf",
        "model": "test-model",
        "overall_meets_expectations": "Yes",
        "strengths": [],
        "areas_for_improvement": [],
        "grading_notes": "",
    }
    for domain, fields in CRITERIA.items():
        value[domain] = {
            field: {"score": score, "explanation": "Model-authored."}
            for field in fields
        }
        value[domain]["overall_decision"] = "Yes"
        value[domain]["if_no_explanation"] = ""
    return value


class ConsensusStreamlitIntegrationTests(unittest.TestCase):
    def _result(
        self,
        model_name: str,
        data: dict,
        image_path: Path,
    ) -> EvaluationResult:
        return EvaluationResult(
            model_name=model_name,
            model_id="model-id",
            data=data,
            output_path=image_path.with_suffix(".json"),
            source_image_path=image_path,
        )

    def test_single_model_outcomes_are_not_consensus_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "request.jpg"
            image.write_bytes(b"image")
            gemma = self._result("Gemma", grading(), image)
            llama = self._result("Llama 3.2 90B Vision", grading(), image)
            self.assertFalse(consensus_ready([gemma]))
            self.assertFalse(consensus_ready([llama]))
            self.assertTrue(consensus_ready([gemma, llama]))

    def test_one_failure_prevents_false_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "request.jpg"
            image.write_bytes(b"image")
            gemma = self._result("Gemma", grading(), image)
            failure = EvaluationFailure(
                "Llama 3.2 90B Vision",
                "model-id",
                "failed",
                Path(temp_dir) / "debug.json",
            )
            self.assertFalse(consensus_ready([gemma, failure]))

    def test_initial_outputs_are_deep_copied_and_remain_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "request.jpg"
            image.write_bytes(b"image")
            gemma_data = grading(3)
            llama_data = grading(2)
            results = [
                self._result("Gemma", gemma_data, image),
                self._result("Llama 3.2 90B Vision", llama_data, image),
            ]
            initial = immutable_initial_results(results)
            initial["gemma"]["knowledge_acquisition"]["basic_science"]["score"] = 1
            self.assertEqual(
                gemma_data["knowledge_acquisition"]["basic_science"]["score"],
                3,
            )

    def test_exact_request_images_are_passed_without_rerendering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gemma_image = Path(temp_dir) / "gemma.jpg"
            llama_image = Path(temp_dir) / "llama.jpg"
            gemma_image.write_bytes(b"exact-gemma-jpeg")
            llama_image.write_bytes(b"exact-llama-jpeg")
            inputs = exact_request_image_inputs(
                [
                    self._result("Gemma", grading(), gemma_image),
                    self._result("Llama 3.2 90B Vision", grading(), llama_image),
                ]
            )
            self.assertEqual(base64.b64decode(inputs["gemma"]), b"exact-gemma-jpeg")
            self.assertEqual(base64.b64decode(inputs["llama"]), b"exact-llama-jpeg")

    def test_comparison_table_uses_backend_values_without_averaging(self) -> None:
        gemma = grading(3)
        llama = grading(3)
        path = "integration.illness_scripts.score"
        llama["integration"]["illness_scripts"]["score"] = 2
        initial = {"gemma": gemma, "llama": llama}
        export = fallback_comparison_export("map.pdf", initial)
        export["post_review_comparison"] = {
            "resolution_status_by_path": {path: "unresolved_same_as_initial"}
        }
        export["consensus"] = {
            "consensus_grading": copy.deepcopy(gemma),
            "criterion_resolutions": [
                {
                    "path": path,
                    "consensus_value": 3,
                    "human_review_recommended": True,
                }
            ],
            "unresolved_disagreements": [{"path": path}],
        }
        row = next(item for item in comparison_rows(export) if item["Criterion"] == "Illness Scripts")
        self.assertEqual(row["Gemma initial"], 3)
        self.assertEqual(row["Llama initial"], 2)
        self.assertEqual(row["Consensus"], 3)
        self.assertEqual(row["Human review"], "Recommended")
        self.assertNotIn("Average", row)

    def test_active_app_removes_placeholder_and_learning_mode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "app.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("Coming Soon", source)
        self.assertNotIn("Learning Mode", source)
        self.assertIn("run_consensus_pipeline(", source)
        self.assertIn('model_selection == "Both"', source)
        self.assertIn('previous_file_fingerprint != uploaded_file_fingerprint', source)
        self.assertIn("clear_current_run()", source)

        display_source = (root / "interface" / "consensus_display.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('st.tabs(["Gemma", "Llama", "Consensus", "Comparison"])', display_source)
        self.assertIn("Human review is recommended", display_source)
        self.assertIn("Independent model results are still available", display_source)


if __name__ == "__main__":
    unittest.main()
