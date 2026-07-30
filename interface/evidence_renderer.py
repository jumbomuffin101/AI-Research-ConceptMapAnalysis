"""Render model-provided normalized evidence boxes on an image copy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from grading.multimodal_feedback import valid_bbox


@dataclass(frozen=True)
class OverlayItem:
    bbox: list[float] | None
    kind: str
    label: int
    low_confidence: bool = False


def normalized_to_pixels(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if not valid_bbox(bbox):
        raise ValueError("Invalid normalized bounding box.")
    x_min, y_min, x_max, y_max = (float(value) for value in bbox)
    return (
        max(0, min(width - 1, round(x_min * width))),
        max(0, min(height - 1, round(y_min * height))),
        max(1, min(width, round(x_max * width))),
        max(1, min(height, round(y_max * height))),
    )


def render_evidence_overlay(
    image_path: Path,
    overlays: Iterable[OverlayItem],
) -> tuple[Any, list[str]]:
    """Return a PIL image copy plus warnings; never mutate the source file."""
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as source:
        rendered = source.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    warnings: list[str] = []

    for overlay in overlays:
        if overlay.bbox is None:
            continue
        if not valid_bbox(overlay.bbox):
            warnings.append(f"Overlay {overlay.label} omitted because its bbox is invalid.")
            continue
        try:
            box = normalized_to_pixels(overlay.bbox, rendered.width, rendered.height)
        except ValueError:
            warnings.append(f"Overlay {overlay.label} omitted because its bbox is invalid.")
            continue
        color = "#f59e0b" if overlay.low_confidence else (
            "#dc2626" if overlay.kind == "improvement" else "#16a34a"
        )
        width = 5 if overlay.low_confidence else 4
        if overlay.kind == "improvement":
            _draw_dashed_rectangle(draw, box, color, width)
        else:
            draw.rectangle(box, outline=color, width=width)
        marker_x, marker_y = box[0], max(0, box[1] - 18)
        marker = str(overlay.label)
        text_box = draw.textbbox((marker_x, marker_y), marker, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((marker_x, marker_y), marker, fill="white", font=font)
    return rendered, warnings


def _draw_dashed_rectangle(
    draw: Any,
    box: tuple[int, int, int, int],
    color: str,
    width: int,
    dash: int = 14,
) -> None:
    x1, y1, x2, y2 = box
    for start in range(x1, x2, dash * 2):
        draw.line((start, y1, min(start + dash, x2), y1), fill=color, width=width)
        draw.line((start, y2, min(start + dash, x2), y2), fill=color, width=width)
    for start in range(y1, y2, dash * 2):
        draw.line((x1, start, x1, min(start + dash, y2)), fill=color, width=width)
        draw.line((x2, start, x2, min(start + dash, y2)), fill=color, width=width)


def criterion_overlays(
    criterion: Mapping[str, Any],
    *,
    show_supporting: bool,
    show_improvements: bool,
    show_low_confidence: bool,
) -> list[OverlayItem]:
    overlays: list[OverlayItem] = []
    label = 1
    if show_supporting:
        for evidence in criterion.get("supporting_evidence", []):
            if not isinstance(evidence, Mapping):
                continue
            confidence = evidence.get("confidence")
            low = isinstance(confidence, (int, float)) and confidence < 0.60
            if low and not show_low_confidence:
                continue
            overlays.append(OverlayItem(evidence.get("bbox"), "support", label, low))
            label += 1
    if show_improvements:
        for improvement in criterion.get("missing_evidence", []):
            if not isinstance(improvement, Mapping):
                continue
            bbox = improvement.get("bbox")
            if bbox is not None:
                overlays.append(OverlayItem(bbox, "improvement", label, False))
                label += 1
    return overlays
