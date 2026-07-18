"""Streamlit entry point for the AI concept map grading demo."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from interface.grading_runner import (
    GradingError,
    run_evaluation,
    save_evaluation_results,
    selected_model_names,
)
from interface.reference_materials import (
    ReferenceMaterialError,
    extract_reference_materials,
)
from interface.result_display import display_results
from scripts.generate_evaluation_report import generate_report

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
    options=["Gemma", "Nemotron", "Both"],
    horizontal=True,
)

reference_uploads = st.file_uploader(
    "Reference Materials (Optional)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload the patient case and relevant session slides for this evaluation only.",
)
if reference_uploads:
    st.caption("Reference materials loaded:")
    st.markdown("\n".join(f"- {file.name}" for file in reference_uploads))

reference_fingerprint = hashlib.sha256(
    b"".join(
        file.name.encode("utf-8") + b"\0" + file.getvalue()
        for file in (reference_uploads or [])
    )
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
            reference_materials = extract_reference_materials(reference_uploads)
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
                        reference_materials=reference_materials,
                    )
                show_progress("Rendering results")
                st.session_state["evaluation_results"] = results
        except (GradingError, ReferenceMaterialError) as exc:
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
