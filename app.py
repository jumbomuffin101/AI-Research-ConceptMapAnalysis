"""Streamlit entry point for the AI concept map grading demo."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import streamlit as st

from interface.grading_runner import (
    GradingError,
    run_evaluation,
    save_evaluation_results,
    selected_model_names,
)
from interface.result_display import display_results
from reference_materials import load_julia_parker_reference
from scripts.generate_evaluation_report import generate_report


NO_REFERENCE_WARNING = (
    "Scores involving unit coverage, patient-case completeness, or prior-course "
    "knowledge are provisional because no reference materials were supplied."
)

st.set_page_config(page_title="AI Concept Map Grading Demo", layout="wide")

st.title("AI Concept Map Grading Demo")
st.write(
    "Upload a medical concept map PDF and generate evidence-grounded "
    "rubric-based evaluations using multimodal AI models."
)

uploaded_file = st.file_uploader("Concept map PDF", type=["pdf"])
uploaded_file_fingerprint = (
    hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if uploaded_file is not None
    else None
)
model_selection = st.radio(
    "Model",
    options=["Gemma", "Llama 4 Scout", "Both"],
    horizontal=True,
)

st.subheader("Reference Materials (Optional)")
st.caption("Reference set: Julia Parker Week 6")
use_reference_materials = st.toggle("Use reference materials", value=True)
st.caption(
    "Reference materials define what students were expected to use. They are not "
    "treated as evidence from the concept map."
)
with st.expander("Add patient case, unit content, DDx, prior concepts, or instructor notes"):
    additional_reference_material = {
        "patient_case": st.text_area("Patient case", height=120),
        "unit_content": st.text_area(
            "Unit learning objectives or session content", height=120
        ),
        "expected_differential_diagnoses": st.text_area(
            "Expected/key differential diagnoses", height=90
        ),
        "prior_concepts": st.text_area(
            "Relevant previously learned concepts", height=90
        ),
        "instructor_notes": st.text_area(
            "Instructor notes or expected content", height=120
        ),
    }

if use_reference_materials:
    reference_material: dict[str, object] | None = load_julia_parker_reference()
    supplemental = {
        key: value.strip()
        for key, value in additional_reference_material.items()
        if value.strip()
    }
    if supplemental:
        reference_material["additional_reference_material"] = supplemental
else:
    reference_material = None

if reference_material is None:
    st.warning(NO_REFERENCE_WARNING)

reference_fingerprint = hashlib.sha256(
    json.dumps(reference_material, sort_keys=True, ensure_ascii=False).encode("utf-8")
).hexdigest()
previous_model_selection = st.session_state.get("previous_model_selection")
previous_file_fingerprint = st.session_state.get("previous_file_fingerprint")
previous_reference_fingerprint = st.session_state.get("previous_reference_fingerprint")
if previous_model_selection is None:
    st.session_state["previous_model_selection"] = model_selection
elif model_selection != previous_model_selection:
    st.session_state.pop("evaluation_results", None)
    st.session_state.pop("evaluation_debug", None)
    st.session_state.pop("evaluation_error", None)
    st.session_state.pop("saved_model_results", None)
    st.session_state["previous_model_selection"] = model_selection

if previous_file_fingerprint != uploaded_file_fingerprint:
    st.session_state.pop("evaluation_results", None)
    st.session_state.pop("evaluation_debug", None)
    st.session_state.pop("evaluation_error", None)
    st.session_state.pop("saved_model_results", None)
    st.session_state["previous_file_fingerprint"] = uploaded_file_fingerprint

if previous_reference_fingerprint is None:
    st.session_state["previous_reference_fingerprint"] = reference_fingerprint
elif previous_reference_fingerprint != reference_fingerprint:
    st.session_state.pop("evaluation_results", None)
    st.session_state.pop("evaluation_debug", None)
    st.session_state.pop("evaluation_error", None)
    st.session_state.pop("saved_model_results", None)
    st.session_state["previous_reference_fingerprint"] = reference_fingerprint

st.button("Multi-AI Consensus Grading - Coming Soon", disabled=True)

if st.button("Run Evaluation", type="primary"):
    if uploaded_file is None:
        st.error("Upload a PDF before running the evaluation.")
    else:
        st.session_state.pop("evaluation_results", None)
        st.session_state.pop("saved_model_results", None)
        try:
            status_placeholder = st.empty()

            def show_progress(message: str) -> None:
                status_placeholder.info(message)

            with st.spinner("Running evaluation..."):
                with tempfile.TemporaryDirectory(prefix="concept-map-") as temp_dir:
                    pdf_path = Path(temp_dir) / "uploaded_concept_map.pdf"
                    pdf_path.write_bytes(uploaded_file.getvalue())
                    results = run_evaluation(
                        pdf_path=pdf_path,
                        model_names=selected_model_names(model_selection),
                        original_filename=uploaded_file.name,
                        progress_callback=show_progress,
                        reference_material=reference_material,
                    )
                show_progress("Rendering results")
                st.session_state["evaluation_results"] = results
        except GradingError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Evaluation failed unexpectedly: {exc}")

if st.session_state.get("evaluation_results"):
    display_results(st.session_state["evaluation_results"])

    if st.button("Save Results"):
        saved_models = save_evaluation_results(
            st.session_state["evaluation_results"],
            uploaded_file.name if uploaded_file is not None else "concept_map.pdf",
        )
        if saved_models:
            st.session_state["saved_model_results"] = saved_models
        else:
            st.warning("No successful model results are available to save.")

    saved_models = st.session_state.get("saved_model_results", [])
    if saved_models:
        st.success("Results saved successfully.")
        st.caption("Saved model results: " + ", ".join(saved_models))

    if st.button("Generate Evaluation Report"):
        try:
            report_path, csv_path, json_path = generate_report()
            st.session_state["evaluation_report_files"] = {
                "markdown": str(report_path),
                "csv": str(csv_path),
                "json": str(json_path),
            }
            st.success("Evaluation report generated.")
        except Exception as exc:
            st.error(f"Could not generate the evaluation report: {exc}")

    report_files = st.session_state.get("evaluation_report_files", {})
    if report_files:
        download_specs = [
            ("markdown", "Download Markdown Report", "concept_map_evaluation_report.md", "text/markdown"),
            ("csv", "Download CSV Summary", "concept_map_evaluation_summary.csv", "text/csv"),
            ("json", "Download JSON Summary", "concept_map_evaluation_summary.json", "application/json"),
        ]
        for key, label, filename, mime in download_specs:
            path = Path(report_files[key])
            if path.exists():
                st.download_button(label, data=path.read_bytes(), file_name=filename, mime=mime)
