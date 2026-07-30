from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from consensus.adjudication import AdjudicationResult, ConsensusModelConfig
from consensus.comparison import (
    COMPARISON_FIELDS,
    classify_post_review,
    compare_gradings,
)
from consensus.cross_review import CrossReviewResult
from consensus.schemas import (
    ConsensusValidationError,
    validate_consensus,
    validate_cross_review,
)
from consensus.service import run_consensus_pipeline
from tests.test_multimodal_feedback import complete_result


def grading(score: int = 3, overall: str = "Yes") -> dict:
    value = complete_result(bbox=[0.1, 0.1, 0.5, 0.5])
    value["overall_meets_expectations"] = overall
    for domain in (
        "knowledge_acquisition",
        "integration",
        "application",
        "transfer",
    ):
        value[domain]["overall_decision"] = overall
        value[domain]["if_no_explanation"] = (
            "" if overall == "Yes" else "The domain is substantially incomplete."
        )
        for item in value[domain].values():
            if isinstance(item, dict) and "score" in item:
                item["score"] = score
    return value


def review_envelope(
    own: dict,
    peer: dict,
    path: str,
    *,
    reviewed_value=None,
    action: str = "defended",
) -> dict:
    from consensus.comparison import get_path, set_path

    reviewed = copy.deepcopy(own)
    own_value = get_path(own, path)
    peer_value = get_path(peer, path)
    if reviewed_value is None:
        reviewed_value = own_value
    set_path(reviewed, path, reviewed_value)
    changed = [path] if reviewed_value != own_value else []
    defended = [path] if action == "defended" else []
    return {
        "reviewer_model": "gemma",
        "reviewed_peer_model": "llama",
        "reviewed_grading": reviewed,
        "field_reviews": [
            {
                "path": path,
                "initial_own_value": own_value,
                "peer_value": peer_value,
                "reviewed_own_value": reviewed_value,
                "action": action,
                "reason": "Visible image evidence supports this rubric value.",
            }
        ],
        "changed_field_paths": changed,
        "defended_field_paths": defended,
        "review_confidence": 0.86,
    }


def consensus_envelope(
    grading_value: dict,
    path: str,
    initial_gemma,
    initial_llama,
    *,
    unresolved: bool = False,
) -> dict:
    from consensus.comparison import get_path

    resolution = {
        "path": path,
        "initial_gemma": get_path(initial_gemma, path),
        "initial_llama": get_path(initial_llama, path),
        "reviewed_gemma": get_path(initial_gemma, path),
        "reviewed_llama": get_path(initial_llama, path),
        "consensus_value": get_path(grading_value, path),
        "status": "unresolved" if unresolved else "resolved",
        "resolution_basis": "The visible relationship and rubric descriptor support this value.",
        "human_review_recommended": unresolved,
        "confidence": 0.82,
    }
    unresolved_items = (
        [
            {
                "path": path,
                "gemma_reviewed_value": get_path(initial_gemma, path),
                "llama_reviewed_value": get_path(initial_llama, path),
                "status": "unresolved",
                "reason": "The visible relationship remains ambiguous.",
                "human_review_recommended": True,
            }
        ]
        if unresolved
        else []
    )
    return {
        "consensus_status": (
            "complete_with_human_review" if unresolved else "complete"
        ),
        "consensus_grading": grading_value,
        "criterion_resolutions": [resolution],
        "unresolved_disagreements": unresolved_items,
        "consensus_confidence": 0.82,
        "human_review_recommended": unresolved,
        "consensus_notes": "Model-generated adjudication.",
    }


class ComparisonTests(unittest.TestCase):
    def test_identical_grades_have_zero_disagreements(self) -> None:
        result = compare_gradings(grading(), grading())
        self.assertEqual(result["total_compared_fields"], len(COMPARISON_FIELDS))
        self.assertEqual(result["total_compared_fields"], 20)
        self.assertEqual(result["disagreement_count"], 0)
        self.assertEqual(result["initial_agreement_rate"], 1.0)

    def test_score_domain_and_overall_differences_are_detected(self) -> None:
        gemma = grading()
        llama = grading()
        llama["integration"]["illness_scripts"]["score"] = 2
        llama["application"]["overall_decision"] = "No"
        llama["application"]["if_no_explanation"] = "Application is incomplete."
        llama["overall_meets_expectations"] = "No"
        result = compare_gradings(gemma, llama)
        types = {item["type"] for item in result["disagreements"]}
        self.assertEqual(
            types,
            {"criterion_score", "domain_decision", "overall_decision"},
        )
        self.assertEqual(result["disagreement_count"], 3)
        self.assertEqual(result["agreement_count"], 17)
        self.assertAlmostEqual(result["initial_agreement_rate"], 17 / 20)
        score_item = next(
            item
            for item in result["disagreements"]
            if item["type"] == "criterion_score"
        )
        self.assertEqual(score_item["absolute_difference"], 1)
        self.assertNotIn("consensus_value", result)

    def test_explanation_differences_do_not_count(self) -> None:
        gemma = grading()
        llama = copy.deepcopy(gemma)
        llama["integration"]["illness_scripts"]["explanation"] = "Different prose."
        self.assertEqual(compare_gradings(gemma, llama)["disagreement_count"], 0)


class CrossReviewValidationTests(unittest.TestCase):
    path = "integration.illness_scripts.score"

    def setUp(self) -> None:
        self.gemma = grading(3)
        self.llama = grading(3)
        self.llama["integration"]["illness_scripts"]["score"] = 2

    def validate(self, payload):
        return validate_cross_review(
            payload,
            reviewer_model="gemma",
            reviewed_peer_model="llama",
            initial_own=self.gemma,
            initial_peer=self.llama,
            initial_disagreement_paths={self.path},
        )

    def test_disputed_score_may_change_or_remain_defended(self) -> None:
        revised = self.validate(
            review_envelope(
                self.gemma,
                self.llama,
                self.path,
                reviewed_value=2,
                action="agreed_with_peer",
            )
        )
        self.assertEqual(
            revised.reviewed_grading["integration"]["illness_scripts"]["score"],
            2,
        )
        defended = self.validate(
            review_envelope(self.gemma, self.llama, self.path)
        )
        self.assertEqual(defended.changed_field_paths, ())

    def test_undisputed_score_change_is_rejected_and_original_preserved(self) -> None:
        undisputed = "application.patient_data_pathophysiology.score"
        payload = review_envelope(
            self.gemma,
            self.llama,
            self.path,
        )
        payload["reviewed_grading"]["application"]["patient_data_pathophysiology"]["score"] = 1
        payload["changed_field_paths"] = [undisputed]
        payload["field_reviews"].append(
            {
                "path": undisputed,
                "initial_own_value": 3,
                "peer_value": 3,
                "reviewed_own_value": 1,
                "action": "revised",
                "reason": "Attempted unauthorized change.",
            }
        )
        validated = self.validate(payload)
        self.assertEqual(
            validated.reviewed_grading["application"]["patient_data_pathophysiology"]["score"],
            3,
        )
        self.assertTrue(any("Unauthorized" in warning for warning in validated.warnings))

    def test_changed_field_requires_metadata(self) -> None:
        payload = review_envelope(
            self.gemma,
            self.llama,
            self.path,
            reviewed_value=2,
            action="revised",
        )
        payload["field_reviews"] = []
        with self.assertRaisesRegex(ConsensusValidationError, "requires field_reviews"):
            self.validate(payload)

    def test_incomplete_review_is_rejected_and_initial_is_immutable(self) -> None:
        before = copy.deepcopy(self.gemma)
        payload = review_envelope(self.gemma, self.llama, self.path)
        del payload["reviewed_grading"]["transfer"]
        with self.assertRaises(ConsensusValidationError):
            self.validate(payload)
        self.assertEqual(self.gemma, before)


class PostReviewClassificationTests(unittest.TestCase):
    path = "integration.illness_scripts.score"

    def initial(self):
        gemma = grading(3)
        llama = grading(3)
        llama["integration"]["illness_scripts"]["score"] = 2
        return gemma, llama, compare_gradings(gemma, llama)

    def status(self, reviewed_gemma, reviewed_llama):
        gemma, llama, comparison = self.initial()
        result = classify_post_review(
            comparison,
            gemma,
            llama,
            reviewed_gemma,
            reviewed_llama,
        )
        return result["field_resolutions"][0]["resolution_status"]

    def test_llama_revision_resolves(self) -> None:
        gemma, llama, _ = self.initial()
        llama["integration"]["illness_scripts"]["score"] = 3
        self.assertEqual(
            self.status(gemma, llama),
            "resolved_by_llama_revision",
        )

    def test_gemma_revision_resolves(self) -> None:
        gemma, llama, _ = self.initial()
        gemma["integration"]["illness_scripts"]["score"] = 2
        self.assertEqual(
            self.status(gemma, llama),
            "resolved_by_gemma_revision",
        )

    def test_both_revision_resolves(self) -> None:
        gemma, llama, _ = self.initial()
        gemma["integration"]["illness_scripts"]["score"] = 4
        llama["integration"]["illness_scripts"]["score"] = 4
        self.assertEqual(
            self.status(gemma, llama),
            "resolved_by_both_revision",
        )

    def test_defended_and_unavailable_reviews(self) -> None:
        gemma, llama, _ = self.initial()
        self.assertEqual(
            self.status(gemma, llama),
            "unresolved_same_as_initial",
        )
        self.assertEqual(
            self.status(None, llama),
            "review_unavailable",
        )
        self.assertEqual(
            self.status(None, None),
            "review_unavailable",
        )


class ConsensusValidationTests(unittest.TestCase):
    path = "integration.illness_scripts.score"

    def test_resolved_and_unresolved_consensus(self) -> None:
        gemma = grading(3)
        llama = grading(3)
        llama["integration"]["illness_scripts"]["score"] = 2
        resolved = consensus_envelope(
            grading(3),
            self.path,
            gemma,
            llama,
        )
        self.assertEqual(
            validate_consensus(resolved, initial_disagreement_paths={self.path})[
                "consensus_status"
            ],
            "complete",
        )
        unresolved = consensus_envelope(
            grading(3),
            self.path,
            gemma,
            llama,
            unresolved=True,
        )
        self.assertTrue(
            validate_consensus(unresolved, initial_disagreement_paths={self.path})[
                "human_review_recommended"
            ]
        )

    def test_unresolved_requires_human_review_metadata(self) -> None:
        gemma = grading(3)
        llama = grading(2)
        payload = consensus_envelope(
            grading(3),
            self.path,
            gemma,
            llama,
            unresolved=True,
        )
        payload["unresolved_disagreements"][0]["human_review_recommended"] = False
        with self.assertRaisesRegex(ConsensusValidationError, "Human review"):
            validate_consensus(payload, initial_disagreement_paths={self.path})

    def test_consensus_value_must_be_model_output_value(self) -> None:
        gemma = grading(3)
        llama = grading(2)
        payload = consensus_envelope(grading(3), self.path, gemma, llama)
        payload["criterion_resolutions"][0]["consensus_value"] = 2
        with self.assertRaisesRegex(ConsensusValidationError, "does not match"):
            validate_consensus(payload, initial_disagreement_paths={self.path})


class ConsensusServiceTests(unittest.TestCase):
    path = "integration.illness_scripts.score"

    def _review_result(self, reviewer, own, peer, comparison):
        peer_name = "llama" if reviewer == "gemma" else "gemma"
        payload = review_envelope(own, peer, self.path)
        payload["reviewer_model"] = reviewer
        payload["reviewed_peer_model"] = peer_name
        validated = validate_cross_review(
            payload,
            reviewer_model=reviewer,
            reviewed_peer_model=peer_name,
            initial_own=own,
            initial_peer=peer,
            initial_disagreement_paths={self.path},
        )
        return CrossReviewResult(
            validated,
            json.dumps(payload),
            {"mock": True},
            {
                "image_resent": True,
                "timeout_seconds": 1,
                "max_tokens": 1,
                "streaming_enabled": False,
                "prompt_character_count": 1,
            },
        )

    def test_orchestration_calls_each_review_once_and_preserves_initials(self) -> None:
        gemma = grading(3, "Yes")
        llama = grading(3, "Yes")
        llama["integration"]["illness_scripts"]["score"] = 2
        initial_before = copy.deepcopy({"Gemma": gemma, "Llama 3.2 90B Vision": llama})
        review_calls: list[tuple[str, str]] = []

        def review_runner(**kwargs):
            review_calls.append((kwargs["reviewer_model"], kwargs["image_base64"]))
            return self._review_result(
                kwargs["reviewer_model"],
                kwargs["initial_own"],
                kwargs["initial_peer"],
                kwargs["initial_comparison"],
            )

        consensus_payload = consensus_envelope(
            grading(3),
            self.path,
            gemma,
            llama,
            unresolved=True,
        )

        def adjudicate(**kwargs):
            return AdjudicationResult(
                consensus_payload,
                json.dumps(consensus_payload),
                {"mock": True},
                {
                    "image_resent": True,
                    "timeout_seconds": 1,
                    "max_tokens": 1,
                    "streaming_enabled": False,
                    "prompt_character_count": 1,
                },
            )

        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp) / "run"
            result = run_consensus_pipeline(
                pdf_path=Path(temp) / "map.pdf",
                map_file="map.pdf",
                initial_results=initial_before,
                consensus_config=ConsensusModelConfig("OpenRouter", "test/model"),
                debug_prefix=prefix,
                image_inputs={"gemma": "gemma-image", "llama": "llama-image"},
                cross_review_runner=review_runner,
                adjudication_runner=adjudicate,
            )
            self.assertTrue(result.output_path.exists())
            debug = json.loads(result.debug_path.read_text(encoding="utf-8"))
        self.assertEqual(
            review_calls,
            [("gemma", "gemma-image"), ("llama", "llama-image")],
        )
        self.assertEqual(initial_before["Gemma"], gemma)
        self.assertEqual(initial_before["Llama 3.2 90B Vision"], llama)
        self.assertEqual(
            result.export["consensus"]["consensus_status"],
            "complete_with_human_review",
        )
        self.assertTrue(debug["gemma_cross_review_image_resent"])
        self.assertTrue(debug["llama_cross_review_image_resent"])
        self.assertTrue(debug["consensus_image_resent"])

    def test_provider_failure_never_creates_python_consensus(self) -> None:
        gemma = grading(3, "Yes")
        llama = grading(1, "No")
        before_gemma, before_llama = copy.deepcopy(gemma), copy.deepcopy(llama)

        def failed_review(**_kwargs):
            raise TimeoutError("review failed")

        def failed_consensus(**_kwargs):
            raise TimeoutError("consensus failed")

        with tempfile.TemporaryDirectory() as temp:
            result = run_consensus_pipeline(
                pdf_path=Path(temp) / "map.pdf",
                map_file="map.pdf",
                initial_results={
                    "Gemma": gemma,
                    "Llama 3.2 90B Vision": llama,
                },
                debug_prefix=Path(temp) / "run",
                image_inputs={"gemma": "g", "llama": "l"},
                cross_review_runner=failed_review,
                adjudication_runner=failed_consensus,
            )
        self.assertEqual(result.export["consensus"]["consensus_status"], "unavailable")
        self.assertIsNone(result.export["consensus"]["consensus_grading"])
        self.assertEqual(result.export["initial_results"]["gemma"], before_gemma)
        self.assertEqual(result.export["initial_results"]["llama"], before_llama)

    def test_unanimous_initial_grades_skip_cross_reviews_but_confirm_consensus(self) -> None:
        same_gemma = grading(3, "Yes")
        same_llama = copy.deepcopy(same_gemma)
        review = Mock()
        consensus_payload = {
            "consensus_status": "complete",
            "consensus_grading": copy.deepcopy(same_gemma),
            "criterion_resolutions": [],
            "unresolved_disagreements": [],
            "consensus_confidence": 0.95,
            "human_review_recommended": False,
            "consensus_notes": "The independently generated grading values agree.",
        }
        adjudicate = Mock(
            return_value=AdjudicationResult(
                consensus_payload,
                json.dumps(consensus_payload),
                {"mock": True},
                {
                    "image_resent": True,
                    "timeout_seconds": 1,
                    "max_tokens": 1,
                    "streaming_enabled": False,
                    "prompt_character_count": 1,
                },
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            result = run_consensus_pipeline(
                pdf_path=Path(temp) / "map.pdf",
                map_file="ConceptMap1.pdf",
                initial_results={
                    "Gemma": same_gemma,
                    "Llama 3.2 90B Vision": same_llama,
                },
                debug_prefix=Path(temp) / "run",
                image_inputs={"gemma": "g", "llama": "l"},
                cross_review_runner=review,
                adjudication_runner=adjudicate,
            )
        review.assert_not_called()
        adjudicate.assert_called_once()
        self.assertEqual(result.export["initial_comparison"]["disagreement_count"], 0)
        self.assertEqual(result.export["consensus"]["consensus_status"], "complete")

    def test_one_missing_initial_skips_all_deliberation(self) -> None:
        review = Mock()
        adjudicate = Mock()
        with tempfile.TemporaryDirectory() as temp:
            result = run_consensus_pipeline(
                pdf_path=Path(temp) / "map.pdf",
                map_file="map.pdf",
                initial_results={"Gemma": grading()},
                debug_prefix=Path(temp) / "run",
                image_inputs={"gemma": "g", "llama": "l"},
                cross_review_runner=review,
                adjudication_runner=adjudicate,
            )
        review.assert_not_called()
        adjudicate.assert_not_called()
        self.assertEqual(result.export["consensus"]["consensus_status"], "unavailable")

    def test_strong_and_weak_initial_calibration_remains_immutable(self) -> None:
        strong = grading(3, "Yes")
        weak = grading(1, "No")
        self.assertEqual(strong["overall_meets_expectations"], "Yes")
        self.assertEqual(weak["overall_meets_expectations"], "No")
        before_strong, before_weak = copy.deepcopy(strong), copy.deepcopy(weak)
        compare_gradings(strong, weak)
        self.assertEqual(strong, before_strong)
        self.assertEqual(weak, before_weak)


if __name__ == "__main__":
    unittest.main()
