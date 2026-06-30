"""Streamlit presentation helpers for structured grading results."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from interface.grading_runner import (
    CATEGORY_FIELDS,
    EvaluationFailure,
    EvaluationOutcome,
)


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


def get_result_field(result: Any, field: str, default: Any = None) -> Any:
    """Safely read a field from a dataclass-like result or dictionary."""
    if result is None:
        return default
    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


def get_result_data(result: Any) -> dict[str, Any] | None:
    """Safely get the grading data payload from inconsistent result shapes."""
    if result is None:
        return None
    if hasattr(result, "data"):
        data = getattr(result, "data", None)
        return data if isinstance(data, dict) else None
    if isinstance(result, dict):
        data = result.get("data", result)
        return data if isinstance(data, dict) else None
    return None


def _model_name(result: Any) -> str:
    return str(
        get_result_field(
            result,
            "model_name",
            get_result_field(result, "model", "Model"),
        )
    )


def _model_id(result: Any, data: dict[str, Any] | None = None) -> str:
    value = get_result_field(result, "model_id", None)
    if value is None and isinstance(data, dict):
        value = data.get("model")
    return str(value or "")


def _failure_reason(result: Any, default: str = "Result data is missing or invalid.") -> str:
    for field in ("error_message", "error", "failure_reason", "reason"):
        value = get_result_field(result, field, None)
        if value:
            return str(value)
    return default


def _display_category(group_key: str, section: dict[str, Any]) -> None:
    rows = []
    for field in CATEGORY_FIELDS[group_key]:
        item = section.get(field, {})
        rows.append({"Category": _label(field), "Score": item.get("score", "-")})

    st.dataframe(rows, hide_index=True, use_container_width=True)

    domain_decision = section.get("overall_decision")
    if domain_decision:
        st.markdown(f"**Domain overall decision:** {domain_decision}")
    if section.get("if_no_explanation"):
        st.write(section["if_no_explanation"])

    for field in CATEGORY_FIELDS[group_key]:
        item = section.get(field, {})
        with st.expander(f"{_label(field)} - Score {item.get('score', '-')}"):
            st.markdown("**Explanation**")
            st.write(item.get("explanation") or "No explanation provided.")
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


def display_result(result: Any) -> None:
    """Render one model's complete result."""
    data = get_result_data(result)
    if not data:
        display_failure(result)
        return

    model_name = _model_name(result)
    model_id = _model_id(result, data)
    output_path = get_result_field(result, "output_path", None)

    st.success(f"{model_name} completed successfully.")
    st.header(model_name)
    if model_id:
        st.caption(model_id)
    st.metric(
        "Final Overall: This map meets expectations",
        data.get("overall_meets_expectations", "Not reported"),
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
        file_name=getattr(output_path, "name", f"{model_name.lower()}_result.json"),
        mime="application/json",
        key=f"download-{model_name}-{id(result)}",
    )


def display_failure(result: Any) -> None:
    """Render one model's failed result without hiding other model results."""
    model_name = _model_name(result)
    model_id = _model_id(result, get_result_data(result))
    error_message = _failure_reason(result)
    debug_path = get_result_field(result, "debug_path", None)

    if "implausible all-4 evaluation" in error_message:
        st.warning(
            "Nemotron returned an implausible all-4 evaluation. "
            "Raw output saved for debugging."
        )
    elif "Input is too large for the current model limit" in error_message:
        st.warning(
            "Input is too large for the current model limit. "
            "Try a smaller PDF/image or use the local CLI pipeline. "
            "Raw response saved for debugging."
        )
    else:
        st.warning(
            f"{model_name} did not return usable content. "
            "Raw response saved for debugging. "
            f"You can retry {model_name} only."
        )
    st.header(model_name)
    if model_id:
        st.caption(model_id)
    with st.expander("Failure details", expanded=True):
        st.write(error_message)
        if debug_path:
            st.caption(f"Debug file: {debug_path}")
        st.info(
            f"To retry only this model, choose '{model_name}' in the Model "
            "selector and click Run Evaluation again."
        )


def display_results(results: list[EvaluationOutcome] | Any) -> None:
    """Render successful model results and failed model warnings together."""
    if results is None:
        display_failure(None)
        return
    if not isinstance(results, list):
        results = [results]

    for index, result in enumerate(results):
        if index:
            st.divider()
        if isinstance(result, EvaluationFailure) or get_result_data(result) is None:
            display_failure(result)
        else:
            display_result(result)
