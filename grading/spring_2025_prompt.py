"""Shared authoritative Spring 2025 concept-map grading prompt."""

from __future__ import annotations

import json
from typing import Any

from interface.reference_materials import format_reference_context


# Keep this wording aligned with the Spring 2025 Concept Map Feedback Tool for
# SUMMATIVE Activities. Both active graders use this same rubric and rules.
SPRING_2025_RUBRIC = """SPRING 2025 CONCEPT MAP FEEDBACK TOOL FOR SUMMATIVE ACTIVITIES

KNOWLEDGE ACQUISITION

Identifies key knowledge from basic sciences learned this unit

1 = little or irrelevant knowledge
2 = partly relevant knowledge and/or too general
3 = relevant and mostly synthesized knowledge
4 = synthesized and detailed knowledge from each session

Identifies key knowledge from health system science learned this unit

1 = little or irrelevant knowledge
2 = partly relevant knowledge and/or too general
3 = relevant and mostly synthesized knowledge
4 = synthesized and detailed knowledge from multiple sessions

Identifies key knowledge from clinical sciences learned this unit

1 = little or irrelevant knowledge
2 = partly relevant knowledge and/or too general
3 = relevant and mostly synthesized knowledge
4 = relevant, synthesized, and detailed knowledge from each session

Extracts key information from the patient case

1 = little or irrelevant information
2 = partly relevant knowledge and/or with limited synthesis
3 = relevant and mostly synthesized information from the patient's data
4 = synthesized, relevant, and comprehensive patient data

Identifies key determinants of health (DoH)

1 = DoH are not included
2 = at least one DoH is included but not specific to this patient and/or not clinically relevant
3 = multiple DoH included across multiple areas of the concept map, and it is clear how they impact this patient's condition and/or plan of care
4 = DoH are comprehensively represented across different areas of the concept map, and it is clear how they impact this patient's condition, plan of care, and/or prognosis

Overall:
Does the student's map include key knowledge from the case and content learned during this unit?

Allowed values:
Yes
No

INTEGRATION

Includes a prioritized differential diagnosis (DDx) that contains common, must not miss, and other possible diagnoses based on patient's unique characteristics

1 = DDx not explicitly included and/or mostly incorrect
2 = DDx too narrow or not accurately connected to patient data
3 = DDx is appropriately focused and relevant to patient
4 = DDx is appropriately focused and relevant to patient, and correctly prioritizes them

Connects patient data to reflect illness script(s)

1 = insufficient data included to create illness script(s)
2 = illness script(s) is/are incorrect and/or incomplete
3 = illness script(s) is/are reflected in accurate connections between patient data
4 = illness scripts are reflected in accurate and prioritized connections between patient data for multiple diagnoses

Connects basic science knowledge learned in the unit to other relevant foundational science information

1 = inaccurate and/or illogical connections
2 = mostly accurate but simplistic with some errors
3 = connections are accurate and logically flow from basic science learned to molecular and cellular basis of disease
4 = connections are accurate, logically flow from basic science learned to molecular and cellular basis of disease, and are comprehensive (including a combination of anatomy, histology, biochemistry, genetics, physiology, and/or pharmacology, etc.)

Connects patient data to other relevant clinical information

1 = inaccurate and/or illogical connections
2 = mostly accurate but simplistic with some errors
3 = connections are accurate and logically flow from patient data to other relevant clinical information
4 = connections are accurate, logically flow from patient data to other relevant clinical information, and are comprehensive (including a combination of epidemiology, symptoms, signs, diagnostics, treatments and patient specific risk factors)

Connects patient data to relevant basic science knowledge

1 = inaccurate and/or illogical connections
2 = mostly accurate but simplistic with some errors
3 = connections are accurate and logically flow from patient data to molecular and cellular basis of disease
4 = connections are accurate, logically flow from patient data to molecular and cellular basis of disease, and are comprehensive (including a combination of anatomy, histology, biochemistry, genetics, physiology, and/or pharmacology, etc.)

Overall:
Did the learner connect key knowledge accurately & comprehensively?

Allowed values:
Yes
No

APPLICATION

Concept map explains the underlying pathophysiology of the working diagnosis

1 = connections to explain the pathophysiology are absent or unclear
2 = connections are present but inaccurate and/or too simplistic to explain pathophysiology
3 = connections form a flow of concepts that explains the pathophysiology
4 = connections form a flow of concepts that explains the pathophysiology and includes basic, clinical, and health system sciences

Connections explain the pathophysiology underlying the key patient data

1 = connections to explain the pathophysiology are absent or unclear
2 = connections are present but inaccurate and/or too simplistic to explain pathophysiology
3 = connections form a flow of concepts that explains the pathophysiology
4 = connections form a flow of concepts that explains the pathophysiology of multiple components of patient data (symptoms, signs, findings) and/or plan of care

Overall:
Did the learner explain key clinical data with relevant basic science?

Allowed values:
Yes
No

TRANSFER

Identifies relevant basic science concepts learned in previous courses

1 = little or irrelevant knowledge
2 = partly relevant knowledge and/or too general
3 = relevant and mostly synthesized knowledge
4 = synthesized and detailed knowledge

Identifies relevant clinical concepts learned in previous courses

1 = little or irrelevant knowledge
2 = partly relevant knowledge and/or too general
3 = relevant and mostly synthesized knowledge
4 = synthesized and detailed knowledge

Uses previously learned knowledge to deepen understanding of the pathophysiology of the condition (the “So what?”)

1 = little or irrelevant knowledge
2 = partly relevant knowledge and/or too general
3 = relevant and mostly synthesized knowledge
4 = synthesized and detailed knowledge that connects to both patient data and basic science knowledge

Overall:
Did the learner use previously learned content to deepen understanding?

Allowed values:
Yes
No

OVERALL

This map meets expectations.

Allowed values:
Yes
No

GRADING INSTRUCTIONS

1. Use the rubric above as the sole scoring authority.

2. For every criterion:
   - compare the concept map directly against the exact 1, 2, 3, and 4 descriptors
   - assign the single score whose descriptor best matches the visible map
   - do not invent additional scoring thresholds
   - do not average scores to determine criterion ratings
   - do not use hidden weights

3. Do not automatically choose the lower score when uncertain.
Choose the descriptor that best matches the evidence.

4. Do not automatically fail a domain because one criterion is low.

5. Do not automatically pass a domain because most criterion scores are high.

6. Answer each domain overall question directly using the rubric question itself.

7. Answer the final overall question directly:
"This map meets expectations."

8. Do not overwrite or recalculate domain or final Yes/No decisions in Python based on score thresholds.

9. Python may only:
   - normalize obvious Yes/No formatting variants
   - fill missing if_no_explanation text
   - repair JSON/schema formatting

10. Python must not:
   - change any numeric score
   - force final overall to No because one domain is No
   - force domain decisions from criterion averages
   - apply model-specific grading rules

REFERENCE MATERIALS

When reference materials are uploaded:
- use them only to understand the patient case and content students were expected to learn
- grade only content actually present in the concept map
- do not treat reference material as student evidence
- do not assume every reference detail must appear unless the rubric requires it

When no reference materials are uploaded:
- grade normally from the concept map and rubric
- do not penalize solely because references are absent
"""


def build_grading_prompt(
    map_file: str,
    output_schema: dict[str, Any],
    reference_materials: list[dict[str, str]] | None = None,
) -> str:
    """Build the identical rubric instruction set for each active grader."""
    reference_context = format_reference_context(reference_materials or [])
    reference_section = (
        "\nREFERENCE MATERIAL\n"
        "The following files define the patient case and/or course content the student was expected to use.\n\n"
        f"{reference_context}\n\n"
        "STUDENT CONCEPT MAP\n"
        "The concept map image is the only source of evidence for what the student actually included.\n"
        if reference_context
        else ""
    )
    return (
        f"{SPRING_2025_RUBRIC}\n"
        f"{reference_section}\n"
        "Return ONLY raw valid JSON using this exact schema. No Markdown or prose outside the JSON. "
        "Use a brief explanation for each criterion and brief strengths, areas_for_improvement, and grading_notes.\n"
        f"{json.dumps(output_schema, separators=(',', ':'))}\n"
    )
