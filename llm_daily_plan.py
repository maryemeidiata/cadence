"""
Daily Plan — LLM-generated hour-by-hour schedule.
Generates an hour-by-hour schedule as structured JSON.
"""

import json
import cohere
import streamlit as st
import pandas as pd

from scoring import compute_priority_scores
from risk_model import compute_task_risk, compute_weighted_stress


def _get_client() -> cohere.ClientV2:
    api_key = ""
    try:
        api_key = st.secrets["CO_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    if not api_key:
        import os
        api_key = os.environ.get("CO_API_KEY", "")
    if not api_key:
        raise ValueError("Cohere API key not configured.")
    return cohere.ClientV2(api_key)


DAILY_PLAN_PROMPT = """You are a productivity planner creating an hour-by-hour schedule.

TASKS (sorted by priority — highest risk first):
{task_data}

CONTEXT:
- Stress index: {stress}/100
- Mode: {mode}

=== CRITICAL AVAILABILITY CONSTRAINT ===
{availability}

You MUST ONLY schedule tasks within the available time windows listed above.
If the user is available 13:00–17:00, your FIRST block must start at 13:00 or later.
If the user is available 08:00–12:00, your LAST block in that window must end by 12:00.
NEVER place any task or block outside the available windows. This is mandatory.
==========================================

Respond with ONLY valid JSON. No markdown, no backticks, no explanation.
{{
  "summary": "1-2 sentence strategy overview",
  "blocks": [
    {{
      "start": "HH:MM",
      "end": "HH:MM",
      "task": "Exact task name from list, or Break",
      "note": "Specific sub-activity"
    }}
  ],
  "total_productive_hours": N,
  "advice": "One actionable tip"
}}

Rules:
- Hardest/highest-risk tasks go in the earliest available slot
- 10-15 min breaks between blocks over 1.5 hours
- Blocks must be 30 min to 2.5 hours
- Use EXACT task names from the list
- Skip completed tasks (0h remaining)
- English only"""


def generate_daily_plan(tasks, mode, available_hours, planning_horizon, availability_text):
    if not tasks:
        return None

    co = _get_client()

    df = pd.DataFrame(tasks)
    df_scored = compute_priority_scores(df, mode=mode)
    df_risk = compute_task_risk(df_scored, available_hours, planning_horizon)
    stress = compute_weighted_stress(df_risk)

    progress = st.session_state.get("progress", {})

    task_lines = []
    for _, row in df_risk.iterrows():
        name = row["name"]
        est = float(row["est_hours"])
        done = progress.get(name, 0.0)
        remaining = max(0, est - done)
        if remaining <= 0:
            continue
        fp = round(float(row["failure_probability"]) * 100, 1)
        task_lines.append(
            f"- {name}: {remaining}h remaining, "
            f"{int(row['days_left'])} days left, {fp}% risk"
        )

    if not task_lines:
        return {"summary": "All tasks completed!", "blocks": [],
                "total_productive_hours": 0, "advice": "Great work!"}

    prompt = DAILY_PLAN_PROMPT.format(
        task_data="\n".join(task_lines),
        stress=stress, mode=mode,
        availability=availability_text,
    )

    try:
        response = co.chat(
            model="command-a-03-2025",
            messages=[
                {"role": "system", "content": "You are a productivity planner. Respond ONLY with valid JSON in English. STRICTLY respect the user's availability windows."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.message.content[0].text.strip()
        if raw.startswith("```json"): raw = raw[7:]
        elif raw.startswith("```"): raw = raw[3:]
        if raw.endswith("```"): raw = raw[:-3]

        plan = json.loads(raw.strip())

        if "blocks" in plan:
            cleaned = []
            for b in plan["blocks"]:
                cleaned.append({
                    "start": str(b.get("start", "00:00")),
                    "end": str(b.get("end", "00:00")),
                    "task": str(b.get("task", "Unknown")),
                    "note": str(b.get("note", b.get("activity", ""))),
                })
            plan["blocks"] = cleaned

        return plan
    except Exception:
        return None