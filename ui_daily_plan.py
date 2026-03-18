"""
UI for the Daily Plan tab — colorful timeline with time pickers.
"""

import streamlit as st
from datetime import time, datetime, timedelta
from llm_daily_plan import generate_daily_plan


TASK_PALETTE = [
    ("#0F3D3E", "#E1F5EE"),   # teal
    ("#185FA5", "#E6F1FB"),   # blue
    ("#854F0B", "#FAEEDA"),   # amber
    ("#534AB7", "#EEEDFE"),   # purple
    ("#993C1D", "#FAECE7"),   # coral
    ("#0F6E56", "#D1FAE5"),   # green
]

# Track color assignments per session
def _get_colors(task_name):
    if task_name.lower() == "break":
        return ("#9CA3AF", "#F3F4F6")
    if "task_color_map" not in st.session_state:
        st.session_state["task_color_map"] = {}
    cmap = st.session_state["task_color_map"]
    if task_name not in cmap:
        idx = len(cmap) % len(TASK_PALETTE)
        cmap[task_name] = TASK_PALETTE[idx]
    return cmap[task_name]


def _generate_daily_ics(blocks, date):
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Cadence//Daily Plan//EN"]
    for block in blocks:
        task = block.get("task", "")
        if task.lower() == "break":
            continue
        start, end = block.get("start", "00:00"), block.get("end", "00:00")
        note = block.get("note", "")
        try:
            sh, sm = int(start.split(":")[0]), int(start.split(":")[1])
            eh, em = int(end.split(":")[0]), int(end.split(":")[1])
        except (ValueError, IndexError):
            continue
        dt_s = datetime(date.year, date.month, date.day, sh, sm)
        dt_e = datetime(date.year, date.month, date.day, eh, em)
        lines.extend([
            "BEGIN:VEVENT",
            f"DTSTART:{dt_s.strftime('%Y%m%dT%H%M%S')}",
            f"DTEND:{dt_e.strftime('%Y%m%dT%H%M%S')}",
            f"SUMMARY:{task}", f"DESCRIPTION:{note}",
            f"UID:{task.replace(' ','-')}-{start}@cadence",
            "END:VEVENT",
        ])
    lines.append("END:VCALENDAR")
    return "\n".join(lines)


def render_daily_plan(mode, available_hours, planning_horizon):

    st.markdown("## Daily plan")
    st.caption("AI-generated hour-by-hour schedule based on your tasks, risk levels, and availability.")

    # ── Availability ──
    st.markdown("**When are you available today?**")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        avail_start_1 = st.time_input("From", value=time(8, 0), key="avail_s1")
    with col_a2:
        avail_end_1 = st.time_input("To", value=time(17, 0), key="avail_e1")

    has_second = st.checkbox("I have a second availability window", key="has_second_window")

    availability_text = (
        f"Available ONLY from {avail_start_1.strftime('%H:%M')} to {avail_end_1.strftime('%H:%M')}. "
    )

    if has_second:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            avail_start_2 = st.time_input("From (2nd)", value=time(19, 0), key="avail_s2")
        with col_b2:
            avail_end_2 = st.time_input("To (2nd)", value=time(22, 0), key="avail_e2")
        availability_text += (
            f"Also available from {avail_start_2.strftime('%H:%M')} to {avail_end_2.strftime('%H:%M')}. "
        )

    availability_text += "DO NOT schedule anything outside these windows."

    col1, col2 = st.columns([1, 4])
    with col1:
        generate_btn = st.button("Generate plan", type="primary")

    if generate_btn:
        api_key = ""
        try:
            api_key = st.secrets["CO_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass
        if not api_key:
            import os
            api_key = os.environ.get("CO_API_KEY", "")
        if not api_key:
            st.error("Cohere API key not configured.")
            return
        if not st.session_state.get("tasks"):
            st.warning("Add some tasks first.")
            return

        with st.spinner("Planning your day..."):
            plan = generate_daily_plan(
                st.session_state.tasks, mode, available_hours,
                planning_horizon, availability_text,
            )
        if plan is None:
            st.error("Failed to generate a plan. Please try again.")
            return
        st.session_state["daily_plan"] = plan
        st.rerun()

    # ── Display ──
    if "daily_plan" not in st.session_state:
        return

    plan = st.session_state["daily_plan"]

    summary = plan.get("summary", "")
    if summary:
        st.success(summary)

    blocks = plan.get("blocks", [])
    for block in blocks:
        task = block.get("task", "")
        start = block.get("start", "")
        end = block.get("end", "")
        note = block.get("note", "")
        text_col, bg_col = _get_colors(task)
        is_break = task.lower() == "break"

        if is_break:
            st.caption(f"{start} – {end}  ·  Break")
        else:
            c1, c2 = st.columns([1, 5])
            with c1:
                st.markdown(f"**{start} – {end}**")
            with c2:
                st.markdown(
                    f'<div style="background:{bg_col};border-left:3px solid {text_col};'
                    f'border-radius:0 10px 10px 0;padding:12px 16px;">'
                    f'<div style="font-weight:500;font-size:14px;color:{text_col};">{task}</div>'
                    f'<div style="font-size:13px;color:#6B7280;margin-top:2px;">{note}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    advice = plan.get("advice", "")
    if advice:
        st.info(advice)

    total = plan.get("total_productive_hours", "")
    if total:
        st.metric("Total productive hours", f"{total}h")

    # ── Downloads ──
    st.markdown("---")
    col_dl1, col_dl2 = st.columns([1, 1])
    with col_dl1:
        if blocks:
            ics = _generate_daily_ics(blocks, st.session_state.start_date)
            st.download_button("Download today's plan (.ics)", ics,
                "cadence_daily_plan.ics", "text/calendar")
    with col_dl2:
        if st.button("Regenerate plan", key="regen_plan"):
            del st.session_state["daily_plan"]
            st.rerun()