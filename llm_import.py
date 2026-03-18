"""
Smart Task Import — LLM-powered extraction of tasks from unstructured text.

Uses a two-pass
LLM chain:

  Pass 1 (Extract):  The LLM receives raw unstructured text (syllabus,
      email, project brief, etc.) and extracts structured task data as JSON.

  Pass 2 (Validate): A second LLM call reviews the extracted tasks for
      reasonableness — checking whether hour estimates are realistic,
      deadlines make sense, and dependencies are correctly identified.
      It can adjust estimates and add warnings.

The validated output is then post-processed in Python to conform to
Cadence's task schema and can be directly added to the user's workload.

Architecture notes:
  - Both passes use structured JSON output with careful prompt engineering.
  - Python post-processing validates types, clamps ranges, deduplicates,
    and converts to the internal task format.
  - Uses a multi-call chain with complex
    post-processing of LLM output.
"""

import json
import cohere
import streamlit as st


def _get_client() -> cohere.ClientV2:
    """Return a Cohere V2 client. Reads from Streamlit secrets or CO_API_KEY env var."""
    api_key = ""
    try:
        api_key = st.secrets["CO_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    if not api_key:
        import os
        api_key = os.environ.get("CO_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Cohere API key not configured. "
            "Set it as CO_API_KEY in .streamlit/secrets.toml or as an environment variable."
        )
    return cohere.ClientV2(api_key)


# ----------------------------------------------
# Pass 1: Extract tasks from unstructured text
# ----------------------------------------------

EXTRACT_SYSTEM_PROMPT = """You are a task extraction engine. Given unstructured text (a syllabus, email, project brief, meeting notes, etc.), extract all actionable tasks.

For each task, output:
- name: a concise task name (max 6 words)
- est_hours: estimated hours to complete (be realistic)
- days_left: days until deadline from today (infer from context; if unclear, use 7)
- dependency_risk: true if this task depends on others or blocks others, false otherwise
- strategic_importance: 1-5 (how strategically important, default 3)
- business_impact: 1-5 (consequence of failure, default 3)
- reasoning: brief explanation of your estimates

RESPOND WITH ONLY valid JSON — no markdown, no backticks, no preamble. Use this exact format:
{"tasks": [{"name": "...", "est_hours": N, "days_left": N, "dependency_risk": bool, "strategic_importance": N, "business_impact": N, "reasoning": "..."}]}

If no actionable tasks can be found, return: {"tasks": [], "note": "No actionable tasks found in the provided text."}"""


VALIDATE_SYSTEM_PROMPT = """You are a task validation engine. Review the following extracted tasks and check for issues:

1. Are the hour estimates realistic? (e.g., "Write a 10-page paper" should be 15-25h, not 2h)
2. Are the deadlines reasonable given the context?
3. Are dependencies correctly identified?
4. Are there any duplicate or overlapping tasks that should be merged?

For each task, either confirm it or adjust it. Add a "validation_note" explaining any changes.

RESPOND WITH ONLY valid JSON — no markdown, no backticks, no preamble. Use this exact format:
{"tasks": [{"name": "...", "est_hours": N, "days_left": N, "dependency_risk": bool, "strategic_importance": N, "business_impact": N, "validation_note": "..."}]}"""


def _clean_json_response(text: str) -> str:
    """Strip markdown fences, whitespace, and fix common LLM JSON issues."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Fix: sometimes the LLM outputs non-ASCII key names (e.g. Chinese)
    # Replace any non-ASCII key that looks like "dependency_risk" variants
    import re
    # Generic fix: find the JSON object boundaries
    # Replace known problematic patterns
    text = re.sub(r'"[^"]*(?:依赖|depend)[^"]*"', '"dependency_risk"', text)

    return text


def _safe_parse_tasks(text: str) -> list[dict]:
    """
    Parse tasks JSON from LLM response, handling various malformed outputs.
    Returns a list of task dicts with normalized keys.
    """
    import re

    text = _clean_json_response(text)

    # Try direct parse first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("tasks", [])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array or object from the text
    # Look for the outermost { ... } or [ ... ]
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        match = re.search(pattern, text)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return data.get("tasks", [])
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                continue

    return []


def _normalize_task(raw: dict) -> dict:
    """Normalize a task dict from LLM output to our expected schema."""
    # Map possible key variations to our standard keys
    def _get(keys, default=None):
        for k in keys:
            if k in raw:
                return raw[k]
        return default

    return {
        "name": str(_get(["name", "task_name", "task"], "Unnamed Task")),
        "est_hours": _get(["est_hours", "hours", "estimated_hours", "time"], 4),
        "days_left": _get(["days_left", "days", "deadline_days", "days_until_deadline"], 7),
        "dependency_risk": _get(["dependency_risk", "dependency", "depends_on", "has_dependency"], False),
        "strategic_importance": _get(["strategic_importance", "importance", "priority"], 3),
        "business_impact": _get(["business_impact", "impact"], 3),
        "reasoning": str(_get(["reasoning", "reason", "validation_note", "note"], "")),
        "validation_note": str(_get(["validation_note", "note", "reasoning"], "")),
    }


def extract_tasks_from_text(raw_text: str, mode: str = "Academic") -> dict:
    """
    Two-pass LLM chain to extract and validate tasks from unstructured text.

    Returns:
        {
            "tasks": [list of validated task dicts ready for Cadence],
            "raw_extraction": [list from pass 1],
            "validation_notes": [list of notes from pass 2],
            "error": None or error string
        }
    """
    co = _get_client()

    # ── Pass 1: Extract ──
    try:
        extract_response = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract tasks from the following text. Respond ONLY in English with valid JSON.\n\n{raw_text}"},
            ],
        )
        extract_text = extract_response.message.content[0].text
        raw_tasks = _safe_parse_tasks(extract_text)
    except Exception as e:
        return {"tasks": [], "raw_extraction": [], "validation_notes": [], "error": f"Extraction failed: {e}"}

    if not raw_tasks:
        return {
            "tasks": [],
            "raw_extraction": [],
            "validation_notes": [],
            "error": "No tasks found in the provided text. Try pasting more detailed text with deadlines and descriptions.",
        }

    # Normalize raw tasks
    raw_tasks = [_normalize_task(t) for t in raw_tasks]

    # ── Pass 2: Validate ──
    try:
        validate_response = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Validate these extracted tasks. Respond ONLY in English with valid JSON.\n\n{json.dumps(raw_tasks, indent=2)}\n\nOriginal text context:\n{raw_text[:1000]}"},
            ],
        )
        validate_text = validate_response.message.content[0].text
        validated_tasks = _safe_parse_tasks(validate_text)
        if validated_tasks:
            validated_tasks = [_normalize_task(t) for t in validated_tasks]
        else:
            validated_tasks = raw_tasks
    except Exception:
        validated_tasks = raw_tasks

    # ── Post-processing: conform to Cadence schema ──
    cadence_tasks = []
    validation_notes = []

    for t in validated_tasks:
        # Clamp and validate types
        task = {
            "name": str(t.get("name", "Unnamed Task"))[:50],
            "days_left": max(1, int(t.get("days_left", 7))),
            "est_hours": max(0.5, round(float(t.get("est_hours", 4.0)), 1)),
            "strategic_importance": max(1, min(5, int(t.get("strategic_importance", 3)))),
            "business_impact": max(1, min(5, int(t.get("business_impact", 3)))),
            "dependency_risk": bool(t.get("dependency_risk", False)),
        }

        # In Academic mode, reset strategic/impact to defaults
        if mode == "Academic":
            task["strategic_importance"] = 1
            task["business_impact"] = 1

        cadence_tasks.append(task)
        validation_notes.append(t.get("validation_note", t.get("reasoning", "")))

    # Deduplicate by name (keep first occurrence)
    seen = set()
    unique_tasks = []
    unique_notes = []
    for task, note in zip(cadence_tasks, validation_notes):
        if task["name"].lower() not in seen:
            seen.add(task["name"].lower())
            unique_tasks.append(task)
            unique_notes.append(note)

    return {
        "tasks": unique_tasks,
        "raw_extraction": raw_tasks,
        "validation_notes": unique_notes,
        "error": None,
    }