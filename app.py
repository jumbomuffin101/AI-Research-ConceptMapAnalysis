"""Streamlit entry point for the AI concept map grading demo."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st

from consensus import run_consensus_pipeline
from interface.consensus_display import display_both_with_consensus
from interface.consensus_integration import (
    consensus_ready,
    exact_request_image_inputs,
    fallback_comparison_export,
    immutable_initial_results,
    successful_results,
)
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
st.caption("Build: llama32-90b-nvidia-singlepass")
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
    options=["Gemma", "Llama 3.2 90B Vision", "Both"],
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
RUN_STATE_KEYS = (
    "evaluation_results",
    "evaluation_debug",
    "evaluation_error",
    "saved_model_results",
    "initial_gemma_result",
    "initial_llama_result",
    "consensus_pipeline_result",
    "consensus_error",
    "consensus_fallback_export",
    "current_run_id",
)


def clear_current_run() -> None:
    for key in RUN_STATE_KEYS:
        st.session_state.pop(key, None)


if previous_model_selection is None:
    st.session_state["previous_model_selection"] = model_selection
elif model_selection != previous_model_selection:
    clear_current_run()
    st.session_state["previous_model_selection"] = model_selection

if previous_file_fingerprint != uploaded_file_fingerprint:
    clear_current_run()
    st.session_state["previous_file_fingerprint"] = uploaded_file_fingerprint
    st.session_state["uploaded_map_identity"] = uploaded_file_fingerprint

if previous_reference_fingerprint is None:
    st.session_state["previous_reference_fingerprint"] = reference_fingerprint
elif previous_reference_fingerprint != reference_fingerprint:
    clear_current_run()
    st.session_state["previous_reference_fingerprint"] = reference_fingerprint

st.session_state["selected_model_mode"] = model_selection

if st.button("Run Evaluation", type="primary"):
    if uploaded_file is None:
        st.error("Upload a PDF before running the evaluation.")
    else:
        clear_current_run()
        st.session_state["current_run_id"] = uuid4().hex
        try:
            reference_materials = extract_reference_materials(reference_uploads)
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            progress_stage = {"value": 0}

            def show_progress(message: str) -> None:
                stage_labels = {
                    "Running Gemma grading": ("Grading with Gemma", 1),
                    "Running Llama 3.2 90B Vision grading": (
                        "Grading with Llama",
                        2,
                    ),
                    "Comparing model outputs": ("Comparing model outputs", 3),
                    "No disagreements detected": ("No disagreements detected", 4),
                    "Confirming consensus": ("Confirming consensus", 6),
                    "Gemma is independently reviewing disputed fields": (
                        "Gemma reviewing disagreements",
                        4,
                    ),
                    "Llama is independently reviewing disputed fields": (
                        "Llama reviewing disagreements",
                        5,
                    ),
                    "Generating model-authored consensus": (
                        "Producing consensus",
                        6,
                    ),
                }
                label, stage = stage_labels.get(message, (message, progress_stage["value"]))
                progress_stage["value"] = max(progress_stage["value"], stage)
                progress_bar.progress(min(progress_stage["value"] / 7, 1.0))
                status_placeholder.info(label)

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
                    st.session_state["evaluation_results"] = results
                    successes = successful_results(results)
                    gemma = successes.get("Gemma")
                    llama = successes.get("Llama 3.2 90B Vision")
                    immutable_initial = immutable_initial_results(results)
                    st.session_state["initial_gemma_result"] = (
                        immutable_initial.get("gemma")
                    )
                    st.session_state["initial_llama_result"] = (
                        immutable_initial.get("llama")
                    )

                    if model_selection == "Both":
                        if not consensus_ready(results):
                            st.session_state["consensus_error"] = (
                                "Consensus unavailable because both independent "
                                "graders are required."
                            )
                        else:
                            try:
                                show_progress("Comparing model outputs")
                                fallback_export = fallback_comparison_export(
                                    uploaded_file.name,
                                    immutable_initial,
                                )
                                initial_comparison = fallback_export["initial_comparison"]
                                st.session_state["consensus_fallback_export"] = (
                                    fallback_export
                                )
                                if initial_comparison["disagreement_count"] == 0:
                                    show_progress("No disagreements detected")
                                    show_progress("Confirming consensus")

                                image_inputs = exact_request_image_inputs(results)
                                pipeline = run_consensus_pipeline(
                                    pdf_path=pdf_path,
                                    map_file=uploaded_file.name,
                                    initial_results=immutable_initial,
                                    progress_callback=show_progress,
                                    image_inputs=image_inputs,
                                )
                                st.session_state["consensus_pipeline_result"] = pipeline
                            except Exception as consensus_exc:
                                st.session_state["consensus_error"] = (
                                    "Consensus unavailable. Independent model results "
                                    f"are still available. {consensus_exc}"
                                )
                progress_stage["value"] = 7
                progress_bar.progress(1.0)
                status_placeholder.success("Complete")
        except (GradingError, ReferenceMaterialError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Evaluation failed unexpectedly: {exc}")

if st.session_state.get("evaluation_results"):
    if st.session_state.get("selected_model_mode") == "Both":
        display_both_with_consensus(
            results=st.session_state["evaluation_results"],
            pipeline=st.session_state.get("consensus_pipeline_result"),
            consensus_error=st.session_state.get("consensus_error"),
            fallback_export=st.session_state.get("consensus_fallback_export"),
        )
    else:
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
