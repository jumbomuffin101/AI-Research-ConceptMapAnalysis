"""Streamlit entry point for the AI concept map grading demo."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from interface.grading_runner import (
    GradingError,
    run_evaluation,
    selected_model_names,
)
from interface.result_display import display_results


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
    options=["Gemma", "Phi-4", "Both"],
    horizontal=True,
)

previous_model_selection = st.session_state.get("previous_model_selection")
previous_file_fingerprint = st.session_state.get("previous_file_fingerprint")
if previous_model_selection is None:
    st.session_state["previous_model_selection"] = model_selection
elif model_selection != previous_model_selection:
    st.session_state.pop("evaluation_results", None)
    st.session_state.pop("evaluation_debug", None)
    st.session_state.pop("evaluation_error", None)
    st.session_state["previous_model_selection"] = model_selection

if previous_file_fingerprint != uploaded_file_fingerprint:
    st.session_state.pop("evaluation_results", None)
    st.session_state.pop("evaluation_debug", None)
    st.session_state.pop("evaluation_error", None)
    st.session_state["previous_file_fingerprint"] = uploaded_file_fingerprint

st.button("Multi-AI Consensus Grading - Coming Soon", disabled=True)

if st.button("Run Evaluation", type="primary"):
    if uploaded_file is None:
        st.error("Upload a PDF before running the evaluation.")
    else:
        st.session_state.pop("evaluation_results", None)
        try:
            with st.spinner("Rendering the PDF and running the selected model(s)..."):
                with tempfile.TemporaryDirectory(prefix="concept-map-") as temp_dir:
                    pdf_path = Path(temp_dir) / "uploaded_concept_map.pdf"
                    pdf_path.write_bytes(uploaded_file.getvalue())
                    results = run_evaluation(
                        pdf_path=pdf_path,
                        model_names=selected_model_names(model_selection),
                        original_filename=uploaded_file.name,
                    )
                st.session_state["evaluation_results"] = results
        except GradingError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Evaluation failed unexpectedly: {exc}")

if st.session_state.get("evaluation_results"):
    display_results(st.session_state["evaluation_results"])
