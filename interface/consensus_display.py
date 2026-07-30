"""Presentation helpers for the model-generated consensus pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from consensus.comparison import COMPARISON_FIELDS, get_path
from consensus.service import ConsensusPipelineResult
from interface.grading_runner import EvaluationFailure, EvaluationResult
from interface.result_display import display_failure, display_result


RESOLUTION_LABELS = {
    "resolved_same_value": "Resolved: same value",
    "resolved_by_gemma_revision": "Resolved by Gemma revision",
    "resolved_by_llama_revision": "Resolved by Llama revision",
    "resolved_by_both_revision": "Resolved by both revisions",
    "unresolved_same_as_initial": "Unresolved: unchanged",
    "unresolved_after_revision": "Unresolved after revision",
    "review_unavailable": "Review unavailable",
}


def _reviewed_grading(export: Mapping[str, Any], model: str) -> dict[str, Any] | None:
    review = export.get("cross_reviews", {}).get(model)
    if not isinstance(review, Mapping):
        return None
    grading = review.get("reviewed_grading")
    return dict(grading) if isinstance(grading, Mapping) else None


def comparison_rows(export: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the UI table without resolving or averaging any model values."""
    initial = export.get("initial_results", {})
    gemma = initial.get("gemma") if isinstance(initial, Mapping) else None
    llama = initial.get("llama") if isinstance(initial, Mapping) else None
    reviewed_gemma = _reviewed_grading(export, "gemma")
    reviewed_llama = _reviewed_grading(export, "llama")
    post_review = export.get("post_review_comparison")
    status_by_path = (
        post_review.get("resolution_status_by_path", {})
        if isinstance(post_review, Mapping)
        else {}
    )
    consensus = export.get("consensus", {})
    consensus_grading = (
        consensus.get("consensus_grading")
        if isinstance(consensus, Mapping)
        else None
    )
    resolution_items = (
        consensus.get("criterion_resolutions", [])
        if isinstance(consensus, Mapping)
        else []
    )
    resolutions = {
        str(item.get("path")): item
        for item in resolution_items
        if isinstance(item, Mapping)
    }
    unresolved_items = (
        consensus.get("unresolved_disagreements", [])
        if isinstance(consensus, Mapping)
        else []
    )
    unresolved_paths = {
        str(item.get("path"))
        for item in unresolved_items
        if isinstance(item, Mapping)
    }

    rows: list[dict[str, Any]] = []
    for field in COMPARISON_FIELDS:
        components = field.path.split(".")
        domain = (
            components[0].replace("_", " ").title()
            if len(components) > 1
            else "Final Overall"
        )
        disputed = field.path in status_by_path
        resolution = resolutions.get(field.path, {})
        rows.append(
            {
                "Domain": domain,
                "Criterion": field.label,
                "Gemma initial": get_path(gemma, field.path)
                if isinstance(gemma, Mapping)
                else None,
                "Llama initial": get_path(llama, field.path)
                if isinstance(llama, Mapping)
                else None,
                "Gemma reviewed": (
                    get_path(reviewed_gemma, field.path)
                    if disputed and reviewed_gemma is not None
                    else ("Review unavailable" if disputed else "Not reviewed")
                ),
                "Llama reviewed": (
                    get_path(reviewed_llama, field.path)
                    if disputed and reviewed_llama is not None
                    else ("Review unavailable" if disputed else "Not reviewed")
                ),
                "Consensus": (
                    get_path(consensus_grading, field.path)
                    if isinstance(consensus_grading, Mapping)
                    else None
                ),
                "Resolution status": (
                    RESOLUTION_LABELS.get(
                        str(status_by_path.get(field.path)),
                        str(status_by_path.get(field.path, "Not disputed")).replace(
                            "_", " "
                        ).title(),
                    )
                    if disputed
                    else "Not disputed"
                ),
                "Human review": (
                    "Recommended"
                    if field.path in unresolved_paths
                    or resolution.get("human_review_recommended") is True
                    else "No"
                ),
            }
        )
    return rows


def display_consensus_unavailable(reason: str) -> None:
    st.warning("Consensus unavailable. Independent model results are still available.")
    if reason:
        st.caption(reason)


def _display_consensus_tab(
    pipeline: ConsensusPipelineResult | None,
    error_message: str | None,
    source_image_path: Path | None,
) -> None:
    if pipeline is None:
        display_consensus_unavailable(error_message or "")
        return
    consensus = pipeline.export.get("consensus", {})
    status = consensus.get("consensus_status", "unavailable")
    if status == "complete":
        st.success("Consensus reached")
    elif status == "complete_with_human_review":
        st.warning("Consensus generated with unresolved items")
    else:
        display_consensus_unavailable(str(consensus.get("consensus_notes", "")))
        st.download_button(
            "Download full consensus pipeline JSON",
            data=json.dumps(pipeline.export, indent=2),
            file_name=pipeline.output_path.name,
            mime="application/json",
            key=f"consensus-export-{pipeline.output_path.name}",
        )
        return

    confidence = consensus.get("consensus_confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        st.metric("Consensus confidence", f"{float(confidence):.0%}")
    if consensus.get("human_review_recommended"):
        st.warning("Human review is recommended for unresolved consensus items.")

    grading = consensus.get("consensus_grading")
    if isinstance(grading, dict):
        display_result(
            {
                "model_name": "Consensus",
                "model_id": "Model-generated adjudication",
                "data": grading,
                "output_path": pipeline.output_path,
                "source_image_path": source_image_path,
                "multimodal_available": bool(grading.get("multimodal_feedback")),
            }
        )

    unresolved = consensus.get("unresolved_disagreements", [])
    if unresolved:
        st.subheader("Unresolved disagreements")
        resolutions = {
            str(item.get("path")): item
            for item in consensus.get("criterion_resolutions", [])
            if isinstance(item, Mapping)
        }
        for item in unresolved:
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path", "Unspecified field"))
            resolution = resolutions.get(path, {})
            with st.expander(path.replace("_", " ").title(), expanded=True):
                st.write(f"Gemma reviewed value: {item.get('gemma_reviewed_value', 'Unavailable')}")
                st.write(f"Llama reviewed value: {item.get('llama_reviewed_value', 'Unavailable')}")
                st.write(f"Consensus adjudication: {resolution.get('consensus_value', 'Unavailable')}")
                st.write(f"Reason: {item.get('reason', 'Not provided')}")
                st.warning("Human review recommended")

    if consensus.get("consensus_notes"):
        with st.expander("Consensus notes"):
            st.write(consensus["consensus_notes"])
    st.download_button(
        "Download full consensus pipeline JSON",
        data=json.dumps(pipeline.export, indent=2),
        file_name=pipeline.output_path.name,
        mime="application/json",
        key=f"consensus-export-{pipeline.output_path.name}",
    )


def _display_comparison_tab(
    pipeline: ConsensusPipelineResult | None,
    fallback_export: Mapping[str, Any] | None,
) -> None:
    export = pipeline.export if pipeline is not None else fallback_export
    if not export:
        st.info("Comparison unavailable.")
        return
    initial_comparison = export.get("initial_comparison")
    if isinstance(initial_comparison, Mapping):
        left, right = st.columns(2)
        left.metric("Initial agreements", initial_comparison.get("agreement_count", 0))
        right.metric(
            "Initial disagreements", initial_comparison.get("disagreement_count", 0)
        )
        if initial_comparison.get("disagreement_count", 0):
            reviews = export.get("cross_reviews", {})
            if not isinstance(reviews, Mapping) or reviews.get("gemma") is None:
                st.warning("Gemma cross-review was unavailable.")
            if not isinstance(reviews, Mapping) or reviews.get("llama") is None:
                st.warning("Llama cross-review was unavailable.")
    st.dataframe(comparison_rows(export), hide_index=True, use_container_width=True)


def display_both_with_consensus(
    *,
    results: list[EvaluationResult | EvaluationFailure],
    pipeline: ConsensusPipelineResult | None,
    consensus_error: str | None = None,
    fallback_export: Mapping[str, Any] | None = None,
) -> None:
    """Render immutable independent grades alongside consensus information."""
    by_name = {result.model_name: result for result in results}
    gemma = by_name.get("Gemma")
    llama = by_name.get("Llama 3.2 90B Vision")
    tabs = st.tabs(["Gemma", "Llama", "Consensus", "Comparison"])

    with tabs[0]:
        display_failure(gemma) if isinstance(gemma, EvaluationFailure) or gemma is None else display_result(gemma)
    with tabs[1]:
        display_failure(llama) if isinstance(llama, EvaluationFailure) or llama is None else display_result(llama)
    with tabs[2]:
        source_path = gemma.source_image_path if isinstance(gemma, EvaluationResult) else None
        _display_consensus_tab(pipeline, consensus_error, source_path)
    with tabs[3]:
        _display_comparison_tab(pipeline, fallback_export)
