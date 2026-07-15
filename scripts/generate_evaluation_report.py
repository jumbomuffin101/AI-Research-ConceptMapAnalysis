"""Generate a Markdown report from saved concept map evaluation summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = PROJECT_ROOT / "outputs" / "evaluation_summary"
REPORT_PATH = SUMMARY_DIR / "concept_map_evaluation_report.md"

DOMAIN_FIELDS = {
    "knowledge_acquisition": [
        "basic_science",
        "health_system_science",
        "clinical_science",
        "patient_case_information",
        "determinants_of_health",
    ],
    "integration": [
        "prioritized_differential_diagnosis",
        "illness_scripts",
        "basic_to_foundational_science",
        "patient_data_to_clinical_information",
        "patient_data_to_basic_science",
    ],
    "application": [
        "working_diagnosis_pathophysiology",
        "patient_data_pathophysiology",
    ],
    "transfer": [
        "prior_basic_science",
        "prior_clinical_concepts",
        "deepens_understanding",
    ],
}

DOMAIN_TITLES = {
    "knowledge_acquisition": "Knowledge Acquisition",
    "integration": "Integration",
    "application": "Application",
    "transfer": "Transfer",
}

MODEL_TITLES = {
    "gemma": "Gemma",
    "llama_4_scout": "Llama 4 Scout",
}


def load_results() -> list[tuple[int, dict[str, Any]]]:
    results: list[tuple[int, dict[str, Any]]] = []
    for path in sorted(SUMMARY_DIR.glob("map_*_results.json"), key=map_file_sort_key):
        match = re.search(r"map_(\d+)_results\.json$", path.name)
        if not match:
            continue
        results.append((int(match.group(1)), json.loads(path.read_text(encoding="utf-8"))))
    return results


def map_file_sort_key(path: Path) -> int:
    match = re.search(r"map_(\d+)_results\.json$", path.name)
    return int(match.group(1)) if match else 999


def criterion_label(field_name: str) -> str:
    return field_name.replace("_", " ").title()


def model_score(model_payload: dict[str, Any] | None, domain: str, criterion: str) -> str:
    if not isinstance(model_payload, dict):
        return "N/A"
    section = model_payload.get(domain)
    if not isinstance(section, dict):
        return "N/A"
    item = section.get(criterion)
    if not isinstance(item, dict):
        return "N/A"
    score = item.get("score")
    return str(score) if score is not None else "N/A"


def model_overall(model_payload: dict[str, Any] | None) -> str:
    if not isinstance(model_payload, dict):
        return "N/A"
    return str(model_payload.get("overall_meets_expectations", "N/A"))


def model_items(model_payload: dict[str, Any] | None, field: str) -> list[str]:
    if not isinstance(model_payload, dict):
        return []
    value = model_payload.get(field, [])
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def all_scores(model_payload: dict[str, Any] | None) -> list[int]:
    if not isinstance(model_payload, dict):
        return []
    scores: list[int] = []
    for domain, criteria in DOMAIN_FIELDS.items():
        section = model_payload.get(domain)
        if not isinstance(section, dict):
            continue
        for criterion in criteria:
            item = section.get(criterion)
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            if isinstance(score, int) and not isinstance(score, bool):
                scores.append(score)
    return scores


def average_score(model_payload: dict[str, Any] | None) -> str:
    scores = all_scores(model_payload)
    if not scores:
        return "N/A"
    return f"{sum(scores) / len(scores):.2f}"


def add_bullets(lines: list[str], items: list[str]) -> None:
    if not items:
        lines.append("- N/A")
        return
    for item in items:
        lines.append(f"- {item}")


def render_map_section(lines: list[str], map_number: int, result: dict[str, Any]) -> None:
    gemma = result.get("gemma")
    llama = result.get("llama_4_scout")

    lines.append(f"## Map {map_number}")
    lines.append("")
    lines.append(f"Source file: `{result.get('map_file', 'N/A')}`")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append(f"Gemma: {model_overall(gemma)}")
    lines.append(f"Llama 4 Scout: {model_overall(llama)}")
    lines.append("")

    for domain, criteria in DOMAIN_FIELDS.items():
        lines.append(f"### {DOMAIN_TITLES[domain]}")
        lines.append("")
        lines.append("| Criterion | Gemma | Llama 4 Scout |")
        lines.append("|---|---:|---:|")
        for criterion in criteria:
            lines.append(
                f"| {criterion_label(criterion)} | "
                f"{model_score(gemma, domain, criterion)} | "
                f"{model_score(llama, domain, criterion)} |"
            )
        lines.append("")

    lines.append("### Strengths")
    lines.append("")
    lines.append("Gemma:")
    add_bullets(lines, model_items(gemma, "strengths"))
    lines.append("")
    lines.append("Llama 4 Scout:")
    add_bullets(lines, model_items(llama, "strengths"))
    lines.append("")

    lines.append("### Areas for Improvement")
    lines.append("")
    lines.append("Gemma:")
    add_bullets(lines, model_items(gemma, "areas_for_improvement"))
    lines.append("")
    lines.append("Llama 4 Scout:")
    add_bullets(lines, model_items(llama, "areas_for_improvement"))
    lines.append("")


def render_cross_model_summary(lines: list[str], results: list[tuple[int, dict[str, Any]]]) -> None:
    agreed = 0
    disagreed = 0

    lines.append("## Cross-Model Summary")
    lines.append("")
    lines.append("| Map | Gemma Overall | Llama 4 Scout Overall | Agreed | Gemma Avg Score | Llama 4 Scout Avg Score |")
    lines.append("|---|---|---|---|---:|---:|")

    for map_number, result in results:
        gemma = result.get("gemma")
        llama = result.get("llama_4_scout")
        gemma_overall = model_overall(gemma)
        llama_overall = model_overall(llama)
        agreement = gemma_overall == llama_overall and gemma_overall != "N/A"
        if agreement:
            agreed += 1
        else:
            disagreed += 1
        lines.append(
            f"| Map {map_number} | {gemma_overall} | {llama_overall} | "
            f"{'Yes' if agreement else 'No'} | {average_score(gemma)} | "
            f"{average_score(llama)} |"
        )

    lines.append("")
    lines.append(f"- Maps where both models agreed: {agreed}")
    lines.append(f"- Maps where models disagreed: {disagreed}")
    lines.append("")


def generate_report() -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    lines: list[str] = ["# Concept Map Evaluation Results", ""]
    if not results:
        lines.append("No saved map result files were found.")
        lines.append("")
    else:
        for map_number, result in results:
            render_map_section(lines, map_number, result)
        render_cross_model_summary(lines, results)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> None:
    report_path = generate_report()
    print(f"Generated report: {report_path}")


if __name__ == "__main__":
    main()
