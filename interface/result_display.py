"""Streamlit presentation helpers for structured grading results."""

from __future__ import annotations

import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st

from interface.evidence_renderer import (
    OverlayItem,
    criterion_overlays,
    render_evidence_overlay,
)
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


def category_rows(
    group_key: str,
    section: dict[str, Any],
    *,
    learning_mode: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in CATEGORY_FIELDS[group_key]:
        item = section.get(field, {})
        row = {"Category": _label(field)}
        if not learning_mode:
            row["Score"] = item.get("score", "-")
        rows.append(row)
    return rows


def _display_category(
    group_key: str,
    section: dict[str, Any],
    *,
    learning_mode: bool = False,
) -> None:
    st.dataframe(
        category_rows(group_key, section, learning_mode=learning_mode),
        hide_index=True,
        use_container_width=True,
    )

    if learning_mode:
        return

    if not learning_mode:
        domain_decision = section.get("overall_decision")
        if domain_decision:
            st.markdown(f"**Domain overall decision:** {domain_decision}")
        if section.get("if_no_explanation"):
            st.write(section["if_no_explanation"])

    for field in CATEGORY_FIELDS[group_key]:
        item = section.get(field, {})
        title = _label(field)
        if not learning_mode:
            title += f" - Score {item.get('score', '-')}"
        with st.expander(title):
            st.markdown("**Explanation**")
            st.write(item.get("explanation") or "No explanation provided.")
            supporting = item.get("supporting_evidence")
            if isinstance(supporting, list) and supporting:
                st.markdown("**Visible supporting evidence**")
                for evidence in supporting:
                    if isinstance(evidence, dict):
                        st.markdown(
                            f"- {evidence.get('evidence_text', 'Visible evidence')} "
                            f"({evidence.get('location_description', 'location not provided')})"
                        )
    visual_summary = section.get("visual_summary")
    if isinstance(visual_summary, dict):
        with st.expander("Domain visual summary"):
            for evidence in visual_summary.get("strongest_visible_evidence", []):
                st.markdown(f"- {evidence}")
            if visual_summary.get("most_important_missing_connection"):
                st.write(
                    "Most important missing connection: "
                    + str(visual_summary["most_important_missing_connection"])
                )
            confidence = visual_summary.get("domain_confidence")
            if isinstance(confidence, (int, float)):
                st.caption(f"Domain confidence: {confidence:.0%}")
            if visual_summary.get("human_review_recommended"):
                st.warning("Human review is recommended for this domain.")


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


def _criterion_options(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (group, field, f"{GROUP_LABELS[group]} — {_label(field)}")
        for group, fields in CATEGORY_FIELDS.items()
        for field in fields
        if isinstance(data.get(group), dict)
    ]


def _learning_feedback(data: dict[str, Any]) -> None:
    st.subheader("Guided Learning Questions")
    items = data.get("learning_feedback")
    if not isinstance(items, list) or not items:
        st.info("No grounded learning questions were returned.")
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        with st.expander(_label(str(item.get("criterion", "Guided question")))):
            st.write(item.get("guiding_question", ""))
            if item.get("observed_evidence"):
                st.caption(f"Observed evidence: {item['observed_evidence']}")
            if item.get("hint"):
                st.info(f"Hint: {item['hint']}")
            if isinstance(item.get("confidence"), (int, float)):
                st.caption(f"Model confidence: {item['confidence']:.0%}")


def _visual_report_html(
    model_name: str,
    data: dict[str, Any],
    image_path: Path,
) -> bytes:
    overlays: list[OverlayItem] = []
    label = 1
    feedback = data.get("multimodal_feedback", {})
    if isinstance(feedback, dict):
        for region in feedback.get("strongest_regions", []):
            if isinstance(region, dict):
                confidence = region.get("confidence")
                overlays.append(
                    OverlayItem(
                        region.get("bbox"),
                        "support",
                        label,
                        isinstance(confidence, (int, float)) and confidence < 0.60,
                    )
                )
                label += 1
        for improvement in feedback.get("highest_priority_improvements", []):
            if isinstance(improvement, dict):
                overlays.append(
                    OverlayItem(improvement.get("bbox"), "improvement", label)
                )
                label += 1
    rendered, _ = render_evidence_overlay(image_path, overlays)
    buffer = BytesIO()
    rendered.save(buffer, format="JPEG", quality=88)
    image_uri = base64.b64encode(buffer.getvalue()).decode("ascii")
    strongest = feedback.get("strongest_regions", []) if isinstance(feedback, dict) else []
    improvements = (
        feedback.get("highest_priority_improvements", [])
        if isinstance(feedback, dict)
        else []
    )
    strongest_html = "".join(
        f"<li>{item.get('description', '')}</li>"
        for item in strongest
        if isinstance(item, dict)
    )
    improvement_html = "".join(
        f"<li>{item.get('missing_bridge', '')}: {item.get('suggested_revision', '')}</li>"
        for item in improvements
        if isinstance(item, dict)
    )
    review = (
        "Human review recommended."
        if isinstance(feedback, dict) and feedback.get("human_review_recommended")
        else "No overall human-review flag."
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{model_name} visual feedback</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;line-height:1.45}}
img{{max-width:100%;border:1px solid #bbb}} h1,h2{{color:#17324d}}</style></head><body>
<h1>{model_name} — Concept Map Visual Feedback</h1>
<p><strong>Overall decision:</strong> {data.get("overall_meets_expectations", "Not reported")}</p>
<img src="data:image/jpeg;base64,{image_uri}" alt="Annotated concept map">
<h2>Strongest visible evidence</h2><ul>{strongest_html or "<li>None localized.</li>"}</ul>
<h2>Highest-priority missing relationships</h2><ul>{improvement_html or "<li>None returned.</li>"}</ul>
<h2>Confidence warning</h2><p>{review}</p></body></html>"""
    return html.encode("utf-8")


def _display_visual_evidence(
    result: Any,
    data: dict[str, Any],
    *,
    learning_mode: bool,
) -> None:
    st.subheader("Visual Evidence & Feedback")
    available = bool(get_result_field(result, "multimodal_available", False))
    warnings = get_result_field(result, "multimodal_warnings", ())
    if not available:
        st.info(
            "The rubric grading is available, but grounded multimodal evidence "
            "was incomplete or invalid and has been withheld."
        )
        if warnings:
            with st.expander("Evidence validation details"):
                for warning in warnings:
                    st.caption(str(warning))
        return

    image_path_value = get_result_field(result, "source_image_path", None)
    image_path = Path(image_path_value) if image_path_value else None
    if image_path is None or not image_path.exists():
        st.info("The source concept-map image is unavailable for overlay rendering.")
        return

    model_name = _model_name(result)
    view = st.radio(
        "Evidence view",
        ["Original", f"{model_name} evidence"],
        horizontal=True,
        key=f"evidence-view-{model_name}-{id(result)}",
    )
    options = _criterion_options(data)
    selected_label = st.selectbox(
        "Criterion",
        [label for _, _, label in options],
        key=f"criterion-{model_name}-{id(result)}",
    )
    selected_group, selected_field, _ = next(
        option for option in options if option[2] == selected_label
    )
    criterion = data[selected_group][selected_field]

    show_supporting = st.toggle(
        "Show supporting evidence",
        value=True,
        key=f"support-{model_name}-{id(result)}",
    )
    show_improvements = st.toggle(
        "Show suggested improvements",
        value=not learning_mode,
        disabled=learning_mode,
        key=f"improve-{model_name}-{id(result)}",
    )
    show_low_confidence = st.toggle(
        "Show low-confidence regions",
        value=True,
        key=f"confidence-{model_name}-{id(result)}",
    )
    if view == "Original":
        st.image(str(image_path), use_container_width=True)
        overlay_warnings: list[str] = []
    else:
        overlays = criterion_overlays(
            criterion,
            show_supporting=show_supporting,
            show_improvements=show_improvements,
            show_low_confidence=show_low_confidence,
        )
        feedback = data.get("multimodal_feedback", {})
        if show_improvements and isinstance(feedback, dict):
            label = len(overlays) + 1
            for improvement in feedback.get("highest_priority_improvements", []):
                if isinstance(improvement, dict) and improvement.get("bbox") is not None:
                    overlays.append(
                        OverlayItem(improvement.get("bbox"), "improvement", label)
                    )
                    label += 1
        rendered, overlay_warnings = render_evidence_overlay(image_path, overlays)
        st.image(rendered, use_container_width=True)

    if overlay_warnings:
        for warning in overlay_warnings:
            st.caption(warning)

    confidence = criterion.get("criterion_confidence")
    if isinstance(confidence, (int, float)):
        st.progress(float(confidence), text=f"Criterion confidence: {confidence:.0%}")
    if criterion.get("human_review_recommended"):
        st.warning("Human review is recommended for this criterion.")

    for index, evidence in enumerate(criterion.get("supporting_evidence", []), start=1):
        if not isinstance(evidence, dict):
            continue
        with st.expander(f"Supporting evidence {index}"):
            st.write(evidence.get("evidence_text", ""))
            st.caption(evidence.get("location_description", ""))
            st.caption(
                f"Relationship: {evidence.get('relationship_type', '')} · "
                f"Confidence: {evidence.get('confidence', 0):.0%}"
            )

    if not learning_mode:
        for index, missing in enumerate(criterion.get("missing_evidence", []), start=1):
            if not isinstance(missing, dict):
                continue
            with st.expander(f"Missing connection {index}"):
                st.write(missing.get("missing_relationship", ""))
                st.info(missing.get("suggested_connection", ""))
                st.caption(f"Importance: {missing.get('importance', '')}")

        feedback = data.get("multimodal_feedback")
        if isinstance(feedback, dict):
            st.markdown("**Overall visual feedback**")
            for region in feedback.get("strongest_regions", []):
                if isinstance(region, dict):
                    st.success(region.get("description", "Visible strength"))
            for improvement in feedback.get("highest_priority_improvements", []):
                if isinstance(improvement, dict):
                    st.warning(
                        f"{improvement.get('missing_bridge', 'Missing connection')} "
                        f"Suggested revision: {improvement.get('suggested_revision', '')}"
                    )
            overall_confidence = feedback.get("overall_visual_confidence")
            if isinstance(overall_confidence, (int, float)):
                st.caption(f"Overall visual confidence: {overall_confidence:.0%}")
            if feedback.get("human_review_recommended"):
                st.warning("Human review is recommended for the overall visual interpretation.")

    st.download_button(
        "Download Visual Report",
        data=_visual_report_html(model_name, data, image_path),
        file_name=f"{model_name.lower().replace(' ', '_')}_visual_report.html",
        mime="text/html",
        key=f"visual-report-{model_name}-{id(result)}",
    )


def display_result(result: Any, *, learning_mode: bool = False) -> None:
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
    if not learning_mode:
        st.metric(
            "Final Overall: This map meets expectations",
            data.get("overall_meets_expectations", "Not reported"),
        )
    else:
        st.info("Learning Mode hides rubric scores and decisions by default.")

    tabs = st.tabs([GROUP_LABELS[key] for key in CATEGORY_FIELDS])
    for tab, group_key in zip(tabs, CATEGORY_FIELDS):
        with tab:
            section = data.get(group_key, {})
            _display_category(
                group_key,
                section if isinstance(section, dict) else {},
                learning_mode=learning_mode,
            )

    if learning_mode:
        _learning_feedback(data)
    else:
        left, right = st.columns(2)
        with left:
            _display_summary_items("Strengths", data.get("strengths"), "evidence_from_map")
        with right:
            _display_summary_items(
                "Areas for improvement",
                data.get("areas_for_improvement"),
                "missing_or_weak_evidence",
            )

    if data.get("grading_notes") and not learning_mode:
        with st.expander("Grading notes"):
            st.write(data["grading_notes"])

    st.download_button(
        "Download JSON result",
        data=json.dumps(data, indent=2),
        file_name=getattr(output_path, "name", f"{model_name.lower()}_result.json"),
        mime="application/json",
        key=f"download-{model_name}-{id(result)}",
    )
    _display_visual_evidence(result, data, learning_mode=learning_mode)


def display_failure(result: Any) -> None:
    """Render one model's failed result without hiding other model results."""
    model_name = _model_name(result)
    model_id = _model_id(result, get_result_data(result))
    error_message = _failure_reason(result)
    debug_path = get_result_field(result, "debug_path", None)

    if "implausible all-4 evaluation" in error_message:
        st.warning(
            f"{model_name} returned an implausible all-4 evaluation. "
            "Raw output saved for debugging."
        )
    elif "Input is too large for the current model limit" in error_message:
        st.warning(
            "Input is too large for the current model limit. "
            "Try a smaller PDF/image or use the local CLI pipeline. "
            "Raw response saved for debugging."
        )
    elif "could not be converted into valid grading JSON" in error_message:
        st.warning(
            f"{model_name} completed the grading, but its response could not be "
            "converted into valid grading JSON."
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
            try:
                debug_file = Path(debug_path)
                debug_contents = debug_file.read_bytes()
                if not debug_contents:
                    raise OSError("Debug file is empty.")
            except (OSError, TypeError, ValueError):
                st.caption("Debug file is unavailable for download.")
            else:
                st.download_button(
                    "Download Debug File",
                    data=debug_contents,
                    file_name=debug_file.name,
                    mime="application/json",
                    key=f"download-debug-{model_name}-{debug_file.name}-{id(result)}",
                )
        st.info(
            f"To retry only this model, choose '{model_name}' in the Model "
            "selector and click Run Evaluation again."
        )


def display_results(
    results: list[EvaluationOutcome] | Any,
    *,
    learning_mode: bool = False,
) -> None:
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
            display_result(result, learning_mode=learning_mode)
