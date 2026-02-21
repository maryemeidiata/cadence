import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# ------------------------------------------------------------------
# Colour helpers
# ------------------------------------------------------------------

def _risk_colour(fp: float) -> str:
    """
    Maps failure_probability [0,1] to a hex colour on a
    green → amber → red gradient.
    """
    if fp < 0.35:
        return "#22C55E"   # green
    elif fp < 0.60:
        return "#F59E0B"   # amber
    elif fp < 0.80:
        return "#EF4444"   # red
    else:
        return "#7F1D1D"   # deep red — critical


def _risk_label(fp: float) -> str:
    if fp < 0.35:  return "Low risk"
    if fp < 0.60:  return "Moderate risk"
    if fp < 0.80:  return "High risk"
    return "Critical"


# ------------------------------------------------------------------
# Main render function
# ------------------------------------------------------------------

def render_calendar(schedule_df: pd.DataFrame, available_hours: float) -> None:

    st.markdown("## Execution Timeline")

    allocation_log: list[dict] = schedule_df.attrs.get("allocation_log", [])
    deadline_risks: list[dict] = schedule_df.attrs.get("deadline_risks", [])

    if not allocation_log:
        st.info("No allocations to display. Add tasks with estimated hours.")
        return

    log_df = pd.DataFrame(allocation_log)

    # ------------------------------------------------------------------
    # Build Gantt data
    #
    # Strategy: collapse consecutive days for the same task into a single
    # bar (start_day → end_day).  Non-consecutive segments become separate
    # bars (task interrupted and resumed).
    # ------------------------------------------------------------------

    # Sort so we can detect consecutive days easily
    log_df = log_df.sort_values(["task", "day"]).reset_index(drop=True)

    gantt_rows: list[dict] = []
    tasks_ordered: list[str] = log_df["task"].unique().tolist()

    for task in tasks_ordered:
        t_df = log_df[log_df["task"] == task].reset_index(drop=True)
        fp            = float(t_df["failure_probability"].iloc[0])
        deadline_day  = int(t_df["deadline_day"].iloc[0])

        # Group consecutive days into segments
        segment_start = int(t_df["day"].iloc[0])
        segment_end   = segment_start
        segment_hours = float(t_df["hours"].iloc[0])

        for i in range(1, len(t_df)):
            this_day = int(t_df["day"].iloc[i])
            if this_day == segment_end + 1:
                # Consecutive — extend segment
                segment_end   = this_day
                segment_hours += float(t_df["hours"].iloc[i])
            else:
                # Gap — close current segment, open new one
                gantt_rows.append({
                    "task":         task,
                    "start":        segment_start,
                    "end":          segment_end,
                    "hours":        round(segment_hours, 2),
                    "fp":           fp,
                    "deadline_day": deadline_day,
                })
                segment_start = this_day
                segment_end   = this_day
                segment_hours = float(t_df["hours"].iloc[i])

        # Close final segment
        gantt_rows.append({
            "task":         task,
            "start":        segment_start,
            "end":          segment_end,
            "hours":        round(segment_hours, 2),
            "fp":           fp,
            "deadline_day": deadline_day,
        })

    gantt_df = pd.DataFrame(gantt_rows)

    # Y-axis: tasks in priority order (as scheduled)
    y_labels = list(reversed(tasks_ordered))   # Plotly draws bottom-up
    y_index  = {task: i for i, task in enumerate(y_labels)}

    # ------------------------------------------------------------------
    # Build Plotly figure
    # ------------------------------------------------------------------

    fig = go.Figure()

    bar_height = 0.55   # fractional height of each row

    for _, seg in gantt_df.iterrows():
        task         = seg["task"]
        y_pos        = y_index[task]
        x_start      = seg["start"] - 0.45        # centre on day integer
        x_end        = seg["end"]   + 0.45
        fp           = seg["fp"]
        colour       = _risk_colour(fp)
        risk_lbl     = _risk_label(fp)
        hours        = seg["hours"]
        deadline_day = seg["deadline_day"]

        hover_text = (
            f"<b>{task}</b><br>"
            f"Days {seg['start']}–{seg['end']}<br>"
            f"Allocated: {hours}h<br>"
            f"Deadline: Day {deadline_day}<br>"
            f"Risk: {round(fp * 100, 1)}% ({risk_lbl})"
        )

        # Task bar
        fig.add_shape(
            type="rect",
            x0=x_start, x1=x_end,
            y0=y_pos - bar_height / 2,
            y1=y_pos + bar_height / 2,
            fillcolor=colour,
            opacity=0.85,
            line=dict(color="white", width=1.5),
            layer="below",
        )

        # Invisible scatter point for hover tooltip
        fig.add_trace(go.Scatter(
            x=[(x_start + x_end) / 2],
            y=[y_pos],
            mode="markers",
            marker=dict(size=0, color="rgba(0,0,0,0)"),
            text=[hover_text],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        # Task label inside bar (if wide enough)
        bar_width_days = seg["end"] - seg["start"] + 1
        if bar_width_days >= 2:
            fig.add_annotation(
                x=(x_start + x_end) / 2,
                y=y_pos,
                text=f"<b>{task}</b> · {hours}h",
                showarrow=False,
                font=dict(color="white", size=11),
                xanchor="center",
                yanchor="middle",
            )

    # ------------------------------------------------------------------
    # Deadline markers (vertical dashed line per task at its deadline day)
    # ------------------------------------------------------------------
    seen_deadlines: set = set()
    for _, seg in gantt_df.iterrows():
        task         = seg["task"]
        deadline_day = seg["deadline_day"]
        y_pos        = y_index[task]
        key          = (task, deadline_day)

        if key in seen_deadlines:
            continue
        seen_deadlines.add(key)

        # Vertical tick mark at the task's y-row
        fig.add_shape(
            type="line",
            x0=deadline_day + 0.45,
            x1=deadline_day + 0.45,
            y0=y_pos - bar_height / 2 - 0.05,
            y1=y_pos + bar_height / 2 + 0.05,
            line=dict(color="#1E293B", width=2, dash="dot"),
        )
        fig.add_annotation(
            x=deadline_day + 0.45,
            y=y_pos + bar_height / 2 + 0.12,
            text="DL",
            showarrow=False,
            font=dict(size=9, color="#1E293B"),
            xanchor="center",
        )

    # ------------------------------------------------------------------
    # Daily capacity utilisation — secondary bar at the bottom
    # ------------------------------------------------------------------
    util_y = -1.0   # row below all tasks

    for _, day_row in schedule_df.iterrows():
        day       = int(day_row["day"])
        used      = float(day_row["allocated_hours"])
        util_pct  = min(used / available_hours, 1.0)
        overloaded = day_row["overload_hours"] > 0

        bar_colour = "#EF4444" if overloaded else "#94A3B8"

        fig.add_shape(
            type="rect",
            x0=day - 0.45, x1=day - 0.45 + 0.9 * util_pct,
            y0=util_y - 0.2,
            y1=util_y + 0.2,
            fillcolor=bar_colour,
            opacity=0.7,
            line=dict(color="white", width=0.5),
            layer="below",
        )

        fig.add_annotation(
            x=day,
            y=util_y,
            text=f"{used}h",
            showarrow=False,
            font=dict(size=9, color="#374151"),
            xanchor="center",
            yanchor="middle",
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    all_days    = list(range(1, int(schedule_df["day"].max()) + 1))
    n_tasks     = len(tasks_ordered)
    fig_height  = max(350, 100 + n_tasks * 60)

    fig.update_layout(
        height=fig_height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        xaxis=dict(
            title="Planning Day",
            tickmode="array",
            tickvals=all_days,
            ticktext=[f"Day {d}" for d in all_days],
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(-1, n_tasks)),
            ticktext=["Capacity"] + y_labels,
            showgrid=False,
            zeroline=False,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            bordercolor="#E2E8F0",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    st.markdown("""
    <div style="display:flex; gap:24px; margin-top:-8px; margin-bottom:16px; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:14px;height:14px;border-radius:3px;background:#22C55E;"></div>
            <span style="font-size:13px;color:#374151;">Low risk</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:14px;height:14px;border-radius:3px;background:#F59E0B;"></div>
            <span style="font-size:13px;color:#374151;">Moderate risk</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:14px;height:14px;border-radius:3px;background:#EF4444;"></div>
            <span style="font-size:13px;color:#374151;">High risk</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:14px;height:14px;border-radius:3px;background:#7F1D1D;"></div>
            <span style="font-size:13px;color:#374151;">Critical</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:2px;height:14px;border-left:2px dotted #1E293B;"></div>
            <span style="font-size:13px;color:#374151;">Deadline (DL)</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
            <div style="width:14px;height:14px;border-radius:3px;background:#94A3B8;"></div>
            <span style="font-size:13px;color:#374151;">Daily capacity used</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Missed deadline callouts below chart
    # ------------------------------------------------------------------
    if deadline_risks:
        st.markdown("### ⚠️ Deadline Risks")
        for risk in deadline_risks:
            st.error(
                f"**{risk['task']}** — {risk['unfinished_hours']}h will remain "
                f"unfinished by Day {risk['deadline_day']} under current capacity."
            )
    else:
        st.success("All tasks complete within their deadlines.")