"""Generate consolidated reports from automatically saved successful evaluations."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = PROJECT_ROOT / "outputs" / "evaluation_summary"
RAW_DIR = SUMMARY_DIR / "raw"
FAILURE_DIR = SUMMARY_DIR / "failures"
REPORT_PATH = SUMMARY_DIR / "concept_map_evaluation_report.md"
CSV_PATH = SUMMARY_DIR / "concept_map_evaluation_summary.csv"
JSON_PATH = SUMMARY_DIR / "concept_map_evaluation_summary.json"

MODEL_KEYS = {"Gemma": "gemma", "Llama 4 Scout": "llama_4_scout"}
MODEL_TITLES = {"gemma": "Gemma", "llama_4_scout": "Llama 4 Scout"}
DOMAIN_FIELDS = {
    "knowledge_acquisition": [
        "basic_science", "health_system_science", "clinical_science",
        "patient_case_information", "determinants_of_health",
    ],
    "integration": [
        "prioritized_differential_diagnosis", "illness_scripts",
        "basic_to_foundational_science", "patient_data_to_clinical_information",
        "patient_data_to_basic_science",
    ],
    "application": [
        "working_diagnosis_pathophysiology", "patient_data_pathophysiology",
    ],
    "transfer": [
        "prior_basic_science", "prior_clinical_concepts", "deepens_understanding",
    ],
}
DOMAIN_TITLES = {
    "knowledge_acquisition": "Knowledge Acquisition",
    "integration": "Integration",
    "application": "Application",
    "transfer": "Transfer",
}


def _map_key(filename: str) -> str:
    return re.sub(r"\s+", " ", Path(filename).name).strip().lower()


def _sort_key(filename: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", filename)]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_evaluations() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], str]]:
    """Return newest success for each map/model plus latest failure for missing pairs."""
    maps: dict[str, dict[str, Any]] = {}
    successes: dict[tuple[str, str], dict[str, Any]] = {}

    for path in RAW_DIR.glob("*.json") if RAW_DIR.exists() else []:
        record = _read_json(path)
        if not record:
            continue
        filename = record.get("map_file")
        model_key = MODEL_KEYS.get(record.get("model_label"))
        result = record.get("result")
        if not isinstance(filename, str) or model_key is None or not isinstance(result, dict):
            continue
        key = (_map_key(filename), model_key)
        current = successes.get(key)
        if current is None or str(record.get("evaluated_at", "")) >= str(current.get("evaluated_at", "")):
            successes[key] = record

    for (map_key, model_key), record in successes.items():
        maps.setdefault(map_key, {"map_file": record["map_file"], "models": {}, "failures": {}})
        maps[map_key]["models"][model_key] = record

    failure_messages: dict[tuple[str, str], tuple[str, str]] = {}
    for path in FAILURE_DIR.glob("*.json") if FAILURE_DIR.exists() else []:
        record = _read_json(path)
        if not record:
            continue
        filename = record.get("map_file")
        model_key = MODEL_KEYS.get(record.get("model_label"))
        if not isinstance(filename, str) or model_key is None:
            continue
        key = (_map_key(filename), model_key)
        current = failure_messages.get(key)
        timestamp = str(record.get("evaluated_at", ""))
        if current is None or timestamp >= current[0]:
            failure_messages[key] = (timestamp, str(record.get("error", "Unknown model failure.")))
        maps.setdefault(_map_key(filename), {"map_file": filename, "models": {}, "failures": {}})

    missing_failures: dict[tuple[str, str], str] = {}
    for map_key, entry in maps.items():
        for model_key in MODEL_TITLES:
            if model_key not in entry["models"]:
                message = failure_messages.get((map_key, model_key), ("", "No successful result was saved."))[1]
                entry["failures"][model_key] = message
                missing_failures[(map_key, model_key)] = message
    return maps, missing_failures


def _result(entry: dict[str, Any], model_key: str) -> dict[str, Any] | None:
    record = entry["models"].get(model_key)
    return record.get("result") if isinstance(record, dict) and isinstance(record.get("result"), dict) else None


def _overall(result: dict[str, Any] | None) -> str:
    return str(result.get("overall_meets_expectations", "N/A")) if result else "N/A"


def _score(result: dict[str, Any] | None, domain: str, criterion: str) -> str:
    if not result or not isinstance(result.get(domain), dict):
        return "N/A"
    item = result[domain].get(criterion)
    return str(item.get("score", "N/A")) if isinstance(item, dict) else "N/A"


def _domain_decision(result: dict[str, Any] | None, domain: str) -> str:
    if not result or not isinstance(result.get(domain), dict):
        return "N/A"
    return str(result[domain].get("overall_decision", "N/A"))


def _evidence(result: dict[str, Any] | None, domain: str, criterion: str) -> list[str]:
    if not result or not isinstance(result.get(domain), dict):
        return []
    item = result[domain].get(criterion)
    if not isinstance(item, dict):
        return []
    evidence = item.get("evidence_from_map", [])
    if isinstance(evidence, list):
        return [str(value) for value in evidence if str(value).strip()]
    return [str(evidence)] if evidence is not None and str(evidence).strip() else []


def _markdown_evidence(result: dict[str, Any] | None, domain: str, criterion: str) -> str:
    items = _evidence(result, domain, criterion)
    if not items:
        return "None visible"
    return "<br>".join(item.replace("|", "\\|") for item in items)


def _average(result: dict[str, Any] | None, domain: str | None = None) -> float | None:
    if not result:
        return None
    domains = [domain] if domain else DOMAIN_FIELDS.keys()
    scores: list[int] = []
    for domain_name in domains:
        section = result.get(domain_name)
        if not isinstance(section, dict):
            continue
        for criterion in DOMAIN_FIELDS[domain_name]:
            value = section.get(criterion)
            score = value.get("score") if isinstance(value, dict) else None
            if isinstance(score, int) and not isinstance(score, bool):
                scores.append(score)
    return sum(scores) / len(scores) if scores else None


def _items(result: dict[str, Any] | None, field: str) -> list[str]:
    value = result.get(field, []) if result else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _bullets(lines: list[str], values: list[str]) -> None:
    for value in values or ["N/A"]:
        lines.append(f"- {value}")


def render_report(maps: dict[str, dict[str, Any]]) -> str:
    lines = ["# Concept Map Evaluation Results", ""]
    entries = sorted(maps.values(), key=lambda entry: _sort_key(entry["map_file"]))
    if not entries:
        lines += ["No successful evaluation results were found in `outputs/evaluation_summary/raw/`.", ""]

    for number, entry in enumerate(entries, start=1):
        gemma, llama = _result(entry, "gemma"), _result(entry, "llama_4_scout")
        lines += [f"## Map {number}: {entry['map_file']}", "", "### Overall", "",
                  f"Gemma: {_overall(gemma)}", f"Llama 4 Scout: {_overall(llama)}", ""]
        for domain, criteria in DOMAIN_FIELDS.items():
            lines += [
                f"### {DOMAIN_TITLES[domain]}", "",
                "| Criterion | Gemma Score | Gemma Evidence From Map | Llama 4 Scout Score | Llama 4 Scout Evidence From Map |",
                "|---|---:|---|---:|---|",
            ]
            for criterion in criteria:
                lines.append(
                    f"| {criterion.replace('_', ' ').title()} | {_score(gemma, domain, criterion)} | "
                    f"{_markdown_evidence(gemma, domain, criterion)} | {_score(llama, domain, criterion)} | "
                    f"{_markdown_evidence(llama, domain, criterion)} |"
                )
            lines += ["", f"Domain decision — Gemma: {_domain_decision(gemma, domain)}; Llama 4 Scout: {_domain_decision(llama, domain)}", ""]
        for heading, field in [("Strengths", "strengths"), ("Areas for Improvement", "areas_for_improvement"), ("Grading Notes", "grading_notes")]:
            lines += [f"### {heading}", "", "Gemma:"]
            _bullets(lines, _items(gemma, field))
            lines += ["", "Llama 4 Scout:"]
            _bullets(lines, _items(llama, field))
            lines.append("")
        if entry["failures"]:
            lines += ["### Missing Model Results", ""]
            for model_key, message in entry["failures"].items():
                lines.append(f"- {MODEL_TITLES[model_key]}: {message}")
            lines.append("")

    comparable = [entry for entry in entries if _result(entry, "gemma") and _result(entry, "llama_4_scout")]
    agreed = sum(_overall(_result(entry, "gemma")) == _overall(_result(entry, "llama_4_scout")) for entry in comparable)
    lines += ["## Cross-Model Summary", "", "| Map | Gemma Overall | Llama 4 Scout Overall | Agreed | Gemma Avg Score | Llama 4 Scout Avg Score |", "|---|---|---|---|---:|---:|"]
    for index, entry in enumerate(entries, start=1):
        gemma, llama = _result(entry, "gemma"), _result(entry, "llama_4_scout")
        agreement = "Yes" if gemma and llama and _overall(gemma) == _overall(llama) else "No"
        g_avg, l_avg = _average(gemma), _average(llama)
        lines.append(f"| Map {index} | {_overall(gemma)} | {_overall(llama)} | {agreement} | {g_avg:.2f} | {l_avg:.2f} |" if g_avg is not None and l_avg is not None else f"| Map {index} | {_overall(gemma)} | {_overall(llama)} | {agreement} | {f'{g_avg:.2f}' if g_avg is not None else 'N/A'} | {f'{l_avg:.2f}' if l_avg is not None else 'N/A'} |")
    percent = (agreed / len(comparable) * 100) if comparable else 0.0
    lines += ["", f"- Agreement count: {agreed}", f"- Agreement percentage: {percent:.1f}% ({len(comparable)} comparable maps)"]
    for model_key in MODEL_TITLES:
        successful = sum(_result(entry, model_key) is not None for entry in entries)
        lines.append(f"- {MODEL_TITLES[model_key]} successful runs: {successful}; failed/missing runs: {len(entries) - successful}")
    lines += ["", "### Average Domain Scores", "", "| Model | Knowledge Acquisition | Integration | Application | Transfer |"]
    lines.append("|---|---:|---:|---:|---:|")
    for model_key, title in MODEL_TITLES.items():
        averages: list[str] = []
        for domain in DOMAIN_FIELDS:
            values = [
                value
                for entry in entries
                if (value := _average(_result(entry, model_key), domain)) is not None
            ]
            averages.append(f"{sum(values) / len(values):.2f}" if values else "N/A")
        lines.append(f"| {title} | " + " | ".join(averages) + " |")
    lines.append("")
    return "\n".join(lines)


def write_machine_summaries(maps: dict[str, dict[str, Any]]) -> None:
    entries = sorted(maps.values(), key=lambda entry: _sort_key(entry["map_file"]))
    rows: list[dict[str, Any]] = []
    for entry in entries:
        row: dict[str, Any] = {"map_file": entry["map_file"]}
        for model_key in MODEL_TITLES:
            result = _result(entry, model_key)
            prefix = model_key
            row[f"{prefix}_overall"] = _overall(result)
            row[f"{prefix}_successful"] = result is not None
            row[f"{prefix}_failure"] = entry["failures"].get(model_key, "")
            row[f"{prefix}_average_score"] = _average(result)
            for domain in DOMAIN_FIELDS:
                row[f"{prefix}_{domain}_average"] = _average(result, domain)
                row[f"{prefix}_{domain}_overall_decision"] = _domain_decision(
                    result, domain
                )
                for criterion in DOMAIN_FIELDS[domain]:
                    row[f"{prefix}_{domain}_{criterion}_score"] = _score(
                        result, domain, criterion
                    )
                    row[f"{prefix}_{domain}_{criterion}_evidence_from_map"] = json.dumps(
                        _evidence(result, domain, criterion), ensure_ascii=False
                    )
        row["overall_agreement"] = bool(_result(entry, "gemma") and _result(entry, "llama_4_scout") and _overall(_result(entry, "gemma")) == _overall(_result(entry, "llama_4_scout")))
        rows.append(row)
    fieldnames = sorted({field for row in rows for field in row}) or ["map_file"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    JSON_PATH.write_text(json.dumps({"maps": entries, "summary_rows": rows}, indent=2), encoding="utf-8")


def generate_report() -> tuple[Path, Path, Path]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    maps, _ = load_evaluations()
    REPORT_PATH.write_text(render_report(maps), encoding="utf-8")
    write_machine_summaries(maps)
    return REPORT_PATH, CSV_PATH, JSON_PATH


def main() -> None:
    for path in generate_report():
        print(path)


if __name__ == "__main__":
    main()
