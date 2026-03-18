"""
AI Coach — Cohere-powered strategic advisor with tool use.

The coach has access to the user's computed risk data, schedule, and
forecasts through callable tools.  It can:
  • Summarise current workload risk
  • Explain why a specific task is high/low risk
  • Run what-if scenarios (drop a task, change capacity, move deadline)
  • Compare multiple scenarios and narrate tradeoffs

Architecture
────────────
1.  User sends a message via the Streamlit chat UI.
2.  The message + conversation history + tool schemas are sent to Cohere.
3.  If Cohere decides to call tools, we execute them locally (they run
    your existing risk model / scheduler) and return results to Cohere.
4.  Cohere generates a final natural-language response grounded in tool
    outputs (with citations).
5.  The response is streamed back to the user.

This is a multi-turn, multi-call pattern with tool use — clearly
non-straightforward LLM usage.
"""

import json
import copy
import cohere
import pandas as pd
import streamlit as st

from scoring import compute_priority_scores
from risk_model import compute_task_risk, compute_weighted_stress, forecast_system_risk
from scheduler import build_schedule


# ──────────────────────────────────────────────
# Cohere client (reads CO_API_KEY env var)
# ──────────────────────────────────────────────

def _get_client() -> cohere.ClientV2:
    """Return a Cohere V2 client. Reads from Streamlit secrets or CO_API_KEY env var."""
    api_key = ""
    # 1. Streamlit secrets (for Streamlit Cloud deployment)
    try:
        api_key = st.secrets["CO_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # 2. Environment variable (for local development)
    if not api_key:
        import os
        api_key = os.environ.get("CO_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Cohere API key not configured. "
            "Set it as CO_API_KEY in .streamlit/secrets.toml or as an environment variable."
        )
    return cohere.ClientV2(api_key)


# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are Sync, the AI strategy coach inside Cadence — a workload planning tool.

Your job is to help users make smart decisions about their task workload. You have tools that give you access to their live data: risk scores, schedules, forecasts, and the ability to simulate changes.

How to respond:
- ALWAYS call a tool first before answering questions about the user's workload. Never guess — use the data.
- When the user asks "what's my biggest risk?", call get_current_analysis, then identify the task with the HIGHEST failure_probability and explain WHY it's risky (deadline pressure, hours needed vs available, competition with other tasks).
- When explaining risk, be specific: name the task, state its failure probability, and say what's causing it (e.g., "You need 8 hours for Essay but only have 6 hours of capacity before its deadline in 3 days").
- Give actionable advice: "Start with X today", "Consider extending Y's deadline", "Drop Z to free up capacity".
- Keep responses concise — 2-4 short paragraphs max. No bullet-point dumps.
- When running what-if scenarios, always state the before AND after numbers so the user can see the impact.
- Speak in plain language like a helpful advisor, not a robot reading a spreadsheet.
- Always respond in English."""


# ──────────────────────────────────────────────
# Tool schemas (Cohere V2 format)
# ──────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_analysis",
            "description": (
                "Returns a full summary of the user's current workload: "
                "stress index, all tasks with their risk scores, schedule status, "
                "missed deadlines, hours at risk, and capacity forecast. "
                "Call this when the user asks about their current situation, "
                "biggest risks, or what to focus on."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_task_risk",
            "description": (
                "Explains why a specific task has its current risk level. "
                "Returns the task's risk breakdown: failure probability, "
                "overload severity, competition pressure, urgency, slack hours, "
                "and impact weight. Use when the user asks 'why is X risky?' "
                "or 'tell me about task X'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "The name of the task to explain. Must match an existing task name.",
                    }
                },
                "required": ["task_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_what_if_scenario",
            "description": (
                "Runs a what-if scenario by temporarily modifying the workload "
                "and returning the new stress index, risk scores, and schedule. "
                "Can drop tasks, change available hours, or move deadlines. "
                "Use this when the user asks 'what if I drop X?', "
                "'what if I only have N hours?', or 'what if I extend the deadline?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drop_tasks": {
                        "type": "string",
                        "description": "Comma-separated list of task names to remove from the workload. Leave empty string if not dropping any.",
                    },
                    "new_capacity": {
                        "type": "number",
                        "description": "New available hours per day. Use 0 to keep current capacity.",
                    },
                    "deadline_changes": {
                        "type": "string",
                        "description": "Comma-separated list of 'task_name:new_days_left' pairs to change deadlines. Leave empty string if not changing any. Example: 'Essay:10,Lab Report:5'",
                    },
                },
                "required": ["drop_tasks", "new_capacity", "deadline_changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_scenarios",
            "description": (
                "Compares two what-if scenarios side by side and returns both results "
                "plus the delta between them. Use when the user asks to compare options, "
                "e.g. 'should I drop the essay or extend the deadline?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scenario_a_label": {
                        "type": "string",
                        "description": "Short label for scenario A, e.g. 'Drop Essay'",
                    },
                    "scenario_a_drop_tasks": {"type": "string", "description": "Tasks to drop in scenario A (comma-separated, or empty string)"},
                    "scenario_a_capacity": {"type": "number", "description": "Capacity for scenario A (0 = keep current)"},
                    "scenario_a_deadline_changes": {"type": "string", "description": "Deadline changes for scenario A (or empty string)"},
                    "scenario_b_label": {
                        "type": "string",
                        "description": "Short label for scenario B, e.g. 'Extend Deadline'",
                    },
                    "scenario_b_drop_tasks": {"type": "string", "description": "Tasks to drop in scenario B (comma-separated, or empty string)"},
                    "scenario_b_capacity": {"type": "number", "description": "Capacity for scenario B (0 = keep current)"},
                    "scenario_b_deadline_changes": {"type": "string", "description": "Deadline changes for scenario B (or empty string)"},
                },
                "required": [
                    "scenario_a_label", "scenario_a_drop_tasks", "scenario_a_capacity", "scenario_a_deadline_changes",
                    "scenario_b_label", "scenario_b_drop_tasks", "scenario_b_capacity", "scenario_b_deadline_changes",
                ],
            },
        },
    },
]


# ──────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────

def _get_workload_context() -> dict:
    """Pull current computed state from session_state."""
    tasks = st.session_state.get("tasks", [])
    mode = st.session_state.get("coach_mode", "Academic")
    available_hours = st.session_state.get("coach_available_hours", 6.0)
    planning_horizon = st.session_state.get("coach_planning_horizon", 7)
    return {
        "tasks": tasks,
        "mode": mode,
        "available_hours": available_hours,
        "planning_horizon": planning_horizon,
    }


def _run_pipeline(tasks: list[dict], mode: str, available_hours: float, horizon: int) -> dict:
    """Run the full scoring → risk → schedule → forecast pipeline on a task list."""
    if not tasks:
        return {"error": "No tasks in workload. Ask the user to add tasks in the sidebar."}

    df = pd.DataFrame(tasks)
    df_scored = compute_priority_scores(df, mode=mode)
    df_risk = compute_task_risk(df_scored, available_hours, horizon)
    schedule_df = build_schedule(df_risk, available_hours, horizon, mode=mode)
    forecast_df = forecast_system_risk(df_scored, available_hours, horizon)
    stress = compute_weighted_stress(df_risk)

    deadline_risks = schedule_df.attrs.get("deadline_risks", [])

    task_summaries = []
    for _, row in df_risk.iterrows():
        task_summaries.append({
            "name": row["name"],
            "est_hours": float(row["est_hours"]),
            "days_left": int(row["days_left"]),
            "failure_probability": round(float(row["failure_probability"]) * 100, 1),
            "priority_score": round(float(row["priority_score"]), 3),
            "expected_loss_hours": round(float(row["expected_loss_hours"]), 1),
            "impact_weight": round(float(row["impact_weight"]), 2),
        })

    return {
        "stress_index": stress,
        "total_hours_at_risk": round(float(df_risk["expected_loss_hours"].sum()), 1),
        "high_risk_count": int((df_risk["failure_probability"] >= 0.7).sum()),
        "missed_deadlines": len(deadline_risks),
        "missed_deadline_details": deadline_risks,
        "tasks": task_summaries,
        "available_hours_per_day": available_hours,
        "planning_horizon_days": horizon,
        "mode": mode,
        "capacity_forecast": forecast_df[["capacity_hours_per_day", "stress_index", "expected_loss_hours"]].to_dict(orient="records"),
    }


def tool_get_current_analysis() -> list[dict]:
    """Tool: get_current_analysis — returns full workload summary."""
    ctx = _get_workload_context()
    result = _run_pipeline(ctx["tasks"], ctx["mode"], ctx["available_hours"], ctx["planning_horizon"])
    return [result]


def tool_explain_task_risk(task_name: str) -> list[dict]:
    """Tool: explain_task_risk — returns detailed risk breakdown for one task."""
    ctx = _get_workload_context()
    if not ctx["tasks"]:
        return [{"error": "No tasks in workload."}]

    df = pd.DataFrame(ctx["tasks"])
    df_scored = compute_priority_scores(df, mode=ctx["mode"])
    df_risk = compute_task_risk(df_scored, ctx["available_hours"], ctx["planning_horizon"])

    match = df_risk[df_risk["name"].str.lower() == task_name.lower()]
    if match.empty:
        available = df_risk["name"].tolist()
        return [{"error": f"Task '{task_name}' not found. Available tasks: {available}"}]

    row = match.iloc[0]
    return [{
        "name": row["name"],
        "failure_probability_pct": round(float(row["failure_probability"]) * 100, 1),
        "est_hours": float(row["est_hours"]),
        "days_left": int(row["days_left"]),
        "slack_hours": round(float(row["slack_hours"]), 1),
        "overload_severity": round(float(row["overload_severity"]), 1),
        "competition_pressure": round(float(row["competition_pressure"]), 2),
        "urgency": round(float(row["urgency"]), 3),
        "dependency_risk": bool(row["dependency_risk"]),
        "impact_weight": round(float(row["impact_weight"]), 2),
        "priority_score": round(float(row["priority_score"]), 3),
        "expected_loss_hours": round(float(row["expected_loss_hours"]), 1),
        "capacity_before_deadline": round(float(row["capacity_before_deadline"]), 1),
    }]


def _apply_scenario(drop_tasks_str: str, new_capacity: float, deadline_changes_str: str) -> dict:
    """Apply scenario modifications and run the pipeline."""
    ctx = _get_workload_context()
    if not ctx["tasks"]:
        return {"error": "No tasks in workload."}

    tasks = copy.deepcopy(ctx["tasks"])
    capacity = ctx["available_hours"] if (new_capacity == 0) else new_capacity

    # Drop tasks
    if drop_tasks_str.strip():
        drop_names = [n.strip().lower() for n in drop_tasks_str.split(",")]
        tasks = [t for t in tasks if t["name"].lower() not in drop_names]

    # Deadline changes
    if deadline_changes_str.strip():
        for change in deadline_changes_str.split(","):
            if ":" in change:
                tname, new_days = change.rsplit(":", 1)
                tname = tname.strip()
                try:
                    new_days = int(new_days.strip())
                except ValueError:
                    continue
                for t in tasks:
                    if t["name"].lower() == tname.lower():
                        t["days_left"] = new_days

    result = _run_pipeline(tasks, ctx["mode"], capacity, ctx["planning_horizon"])
    return result


def tool_run_what_if_scenario(drop_tasks: str, new_capacity: float, deadline_changes: str) -> list[dict]:
    """Tool: run_what_if_scenario — modifies workload and returns new analysis."""
    # Also get current state for comparison
    ctx = _get_workload_context()
    current = _run_pipeline(ctx["tasks"], ctx["mode"], ctx["available_hours"], ctx["planning_horizon"])
    scenario = _apply_scenario(drop_tasks, new_capacity, deadline_changes)

    if "error" in scenario:
        return [scenario]

    delta_stress = round(scenario["stress_index"] - current["stress_index"], 1)
    delta_loss = round(scenario["total_hours_at_risk"] - current["total_hours_at_risk"], 1)

    return [{
        "current_stress_index": current["stress_index"],
        "scenario_stress_index": scenario["stress_index"],
        "stress_change": delta_stress,
        "current_hours_at_risk": current["total_hours_at_risk"],
        "scenario_hours_at_risk": scenario["total_hours_at_risk"],
        "hours_at_risk_change": delta_loss,
        "current_missed_deadlines": current["missed_deadlines"],
        "scenario_missed_deadlines": scenario["missed_deadlines"],
        "scenario_tasks": scenario["tasks"],
        "scenario_missed_deadline_details": scenario.get("missed_deadline_details", []),
        "modifications_applied": {
            "dropped_tasks": drop_tasks,
            "new_capacity": new_capacity if new_capacity > 0 else "unchanged",
            "deadline_changes": deadline_changes,
        },
    }]


def tool_compare_scenarios(
    scenario_a_label: str, scenario_a_drop_tasks: str, scenario_a_capacity: float, scenario_a_deadline_changes: str,
    scenario_b_label: str, scenario_b_drop_tasks: str, scenario_b_capacity: float, scenario_b_deadline_changes: str,
) -> list[dict]:
    """Tool: compare_scenarios — runs two scenarios and returns both + delta."""
    result_a = _apply_scenario(scenario_a_drop_tasks, scenario_a_capacity, scenario_a_deadline_changes)
    result_b = _apply_scenario(scenario_b_drop_tasks, scenario_b_capacity, scenario_b_deadline_changes)

    if "error" in result_a or "error" in result_b:
        return [{"error": result_a.get("error") or result_b.get("error")}]

    return [{
        "scenario_a": {
            "label": scenario_a_label,
            "stress_index": result_a["stress_index"],
            "hours_at_risk": result_a["total_hours_at_risk"],
            "missed_deadlines": result_a["missed_deadlines"],
            "high_risk_count": result_a["high_risk_count"],
            "tasks": result_a["tasks"],
        },
        "scenario_b": {
            "label": scenario_b_label,
            "stress_index": result_b["stress_index"],
            "hours_at_risk": result_b["total_hours_at_risk"],
            "missed_deadlines": result_b["missed_deadlines"],
            "high_risk_count": result_b["high_risk_count"],
            "tasks": result_b["tasks"],
        },
        "comparison": {
            "stress_delta": round(result_a["stress_index"] - result_b["stress_index"], 1),
            "hours_at_risk_delta": round(result_a["total_hours_at_risk"] - result_b["total_hours_at_risk"], 1),
            "missed_deadline_delta": result_a["missed_deadlines"] - result_b["missed_deadlines"],
            "recommendation": (
                f"{scenario_a_label} is better"
                if result_a["stress_index"] < result_b["stress_index"]
                else f"{scenario_b_label} is better"
            ) + " on stress index.",
        },
    }]


# ──────────────────────────────────────────────
# Tool dispatcher
# ──────────────────────────────────────────────

FUNCTIONS_MAP = {
    "get_current_analysis": lambda **kwargs: tool_get_current_analysis(),
    "explain_task_risk": lambda **kwargs: tool_explain_task_risk(**kwargs),
    "run_what_if_scenario": lambda **kwargs: tool_run_what_if_scenario(**kwargs),
    "compare_scenarios": lambda **kwargs: tool_compare_scenarios(**kwargs),
}


# ──────────────────────────────────────────────
# Main chat function (multi-turn with tool loop)
# ──────────────────────────────────────────────

def chat_with_coach(user_message: str, conversation_history: list[dict]) -> tuple[str, list[dict]]:
    """
    Send a message to the AI Coach, handle tool calls, return final response.

    Parameters
    ----------
    user_message : str
        The user's latest message.
    conversation_history : list[dict]
        The conversation history in Cohere message format.

    Returns
    -------
    (response_text, updated_history)
    """
    co = _get_client()

    # Build messages: system + history + new user message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Tool-use loop: keep calling until Cohere returns a text response
    max_iterations = 5  # safety limit
    for _ in range(max_iterations):
        response = co.chat(
            model="command-a-03-2025",
            messages=messages,
            tools=TOOLS,
        )

        # If there are tool calls, execute them and feed results back
        if response.message.tool_calls:
            # Append the assistant's tool-call message
            messages.append(response.message)

            for tc in response.message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                # Execute the tool
                if func_name in FUNCTIONS_MAP:
                    tool_result = FUNCTIONS_MAP[func_name](**func_args)
                else:
                    tool_result = [{"error": f"Unknown tool: {func_name}"}]

                # Format tool results as Cohere expects
                tool_content = []
                for data in tool_result:
                    tool_content.append({
                        "type": "document",
                        "document": {"data": json.dumps(data)},
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                })

            # Continue the loop — Cohere will now see the tool results
            continue

        # No tool calls — we have the final text response
        response_text = ""
        if response.message.content:
            response_text = response.message.content[0].text
        break
    else:
        response_text = "I hit the maximum number of tool calls. Please try rephrasing your question."

    # Build updated history (without system prompt — we add that fresh each time)
    updated_history = [m for m in messages[1:]]  # skip system prompt
    # Add the final assistant response
    updated_history.append({"role": "assistant", "content": response_text})

    return response_text, updated_history