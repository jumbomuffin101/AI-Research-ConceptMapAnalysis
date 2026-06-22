"""Streamlit presentation helpers for structured grading results."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from interface.grading_runner import CATEGORY_FIELDS, EvaluationResult


GROUP_LABELS = {
    "knowledge_acquisition": "Knowledge Acquisition",
    "integration": "Integration",
    "application": "Application",
    "transfer": "Transfer",
}


def _label(value: str) -> str:
    return value.replace("_", " ").title()


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _display_category(group_key: str, section: dict[str, Any]) -> None:
    rows = []
    for field in CATEGORY_FIELDS[group_key]:
        item = section.get(field, {})
        rows.append({"Category": _label(field), "Score": item.get("score", "-")})

    st.dataframe(rows, hide_index=True, use_container_width=True)

    overall = section.get("overall", {})
    if isinstance(overall, dict):
        status = overall.get("meets_expectations")
        if status:
            st.markdown(f"**Meets expectations:** {status}")
        if overall.get("reasoning"):
            st.write(overall["reasoning"])

    for field in CATEGORY_FIELDS[group_key]:
        item = section.get(field, {})
        with st.expander(f"{_label(field)} - Score {item.get('score', '-')}"):
            st.markdown("**Reasoning**")
            st.write(item.get("reasoning") or "No reasoning provided.")
            st.markdown("**Evidence from map**")
            evidence = _as_list(item.get("evidence_from_map"))
            if evidence:
                for entry in evidence:
                    st.markdown(f"- {entry}")
            else:
                st.write("No evidence provided.")


def _display_summary_items(title: str, items: Any, evidence_key: str) -> None:
    st.subheader(title)
    if not isinstance(items, list) or not items:
        st.write("None provided.")
        return

    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            description = item.get("description") or f"Item {index}"
            evidence = _as_list(item.get(evidence_key))
        else:
            description = str(item)
            evidence = []
        with st.expander(description):
            if evidence:
                for entry in evidence:
                    st.markdown(f"- {entry}")
            else:
                st.write("No supporting details provided.")


def display_result(result: EvaluationResult) -> None:
    """Render one model's complete result."""
    data = result.data
    st.header(result.model_name)
    st.caption(result.model_id)
    st.metric(
        "Overall meets expectations",
        data.get("overall_map_meets_expectations", "Not reported"),
    )

    tabs = st.tabs([GROUP_LABELS[key] for key in CATEGORY_FIELDS])
    for tab, group_key in zip(tabs, CATEGORY_FIELDS):
        with tab:
            section = data.get(group_key, {})
            _display_category(group_key, section if isinstance(section, dict) else {})

    left, right = st.columns(2)
    with left:
        _display_summary_items("Strengths", data.get("strengths"), "evidence_from_map")
    with right:
        _display_summary_items(
            "Areas for improvement",
            data.get("areas_for_improvement"),
            "missing_or_weak_evidence",
        )

    if data.get("grading_notes"):
        with st.expander("Grading notes"):
            st.write(data["grading_notes"])

    st.download_button(
        "Download JSON result",
        data=json.dumps(data, indent=2),
        file_name=result.output_path.name,
        mime="application/json",
        key=f"download-{result.output_path.name}",
    )


def display_results(results: list[EvaluationResult]) -> None:
    """Render all selected model results."""
    for index, result in enumerate(results):
        if index:
            st.divider()
        display_result(result)
