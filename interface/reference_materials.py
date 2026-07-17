"""In-memory extraction of optional per-evaluation reference materials."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


SUPPORTED_REFERENCE_EXTENSIONS = {".pdf", ".txt"}


class ReferenceMaterialError(ValueError):
    """A reference upload could not be read for the current evaluation."""


def _extract_pdf_text(file_bytes: bytes, filename: str) -> str:
    try:
        import fitz

        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            text = "\n".join(page.get_text("text") for page in document)
    except Exception as exc:
        raise ReferenceMaterialError(f"Could not read reference PDF '{filename}': {exc}") from exc
    return text.strip()


def _extract_txt_text(file_bytes: bytes, filename: str) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return file_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise ReferenceMaterialError(f"Could not decode reference text file '{filename}'.")


def extract_reference_materials(files: Iterable[Any]) -> list[dict[str, str]]:
    """Return transient filename/text records without writing uploaded files to disk."""
    materials: list[dict[str, str]] = []
    for uploaded_file in files:
        filename = Path(str(uploaded_file.name)).name
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_REFERENCE_EXTENSIONS:
            raise ReferenceMaterialError(
                f"Unsupported reference file '{filename}'. Upload a PDF or TXT file."
            )
        file_bytes = uploaded_file.getvalue()
        text = (
            _extract_pdf_text(file_bytes, filename)
            if extension == ".pdf"
            else _extract_txt_text(file_bytes, filename)
        )
        if not text:
            raise ReferenceMaterialError(
                f"Reference file '{filename}' does not contain extractable text."
            )
        materials.append({"filename": filename, "text": text})
    return materials


def format_reference_context(materials: Iterable[dict[str, str]]) -> str:
    """Format filename-preserving context for a model prompt."""
    sections = [
        f"[File: {material['filename']}]\n{material['text']}"
        for material in materials
        if material.get("filename") and material.get("text")
    ]
    return "\n\n".join(sections)
