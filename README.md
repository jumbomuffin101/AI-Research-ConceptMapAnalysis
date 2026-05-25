# AI Research Concept Map Analysis

This project investigates how AI-generated concept map evaluations differ from human-generated grading and how collaborative grading strategies compare across multiple AI and human evaluators. The current implementation uses multimodal AI models to analyze concept maps through concept extraction, relationship detection, and hierarchy identification as an initial step toward automated rubric-based evaluation.

Current models:

- Qwen-VL (`qwen/qwen2.5-vl-72b-instruct`)
- Nemotron (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`)

Current tasks:

- Extract concepts from medical concept maps
- Detect relationships between concepts
- Identify hierarchical structure
- Compare model outputs
- Support future rubric-based grading experiments

Repository structure:

maps/
- Example strong and weak concept maps

outputs/
- Generated model outputs

Scripts:
- `detect_content_qwen.py`
- `detect_content_nemotron.py`
