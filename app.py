import streamlit as st
import pandas as pd
from datetime import datetime

from scoring import compute_priority_scores
from scheduler import build_schedule
from risk_model import compute_task_risk, forecast_system_risk, compute_weighted_stress

from ui_overview import render_overview
from ui_calendar import render_calendar
from ui_strategic import render_strategic
from ui_coach import render_coach
from ui_import import render_import
from ui_daily_plan import render_daily_plan


st.set_page_config(
    layout="wide",
    page_title="Cadence",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Hide default header but keep sidebar toggle */
[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}
[data-testid="stHeader"]::after {
    display: none;
}

/* Make sidebar collapse button visible on dark background */
[data-testid="stSidebar"] button {
    color: white !important;
}
[data-testid="stSidebar"] svg {
    stroke: white !important;
    fill: white !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    color: #0F3D3E !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    stroke: #0F3D3E !important;
    fill: #0F3D3E !important;
}

/* ── Force sidebar toggle button to always be visible ── */
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #0F3D3E !important;
}
/* Keep sidebar toggle button visible */
button[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    position: fixed !important;
    z-index: 999 !important;
}
.stApp {
    background: linear-gradient(160deg, #DAE5E5 0%, #E6EDEC 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A2E2F 0%, #134849 100%) !important;
    border-right: none;
}
[data-testid="stSidebar"] label { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] p { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] .sidebar-section {
    font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase;
    color: rgba(255,255,255,0.40) !important; margin-bottom: 10px; margin-top: 4px;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: #ffffff !important; border-radius: 8px !important;
    width: 100%; font-weight: 600;
}
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.20) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #1A5C5E !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #1A5C5E !important; }
.task-card {
    background: white; border: 0.5px solid #D0DADA;
    border-radius: 12px; padding: 16px 20px; margin-bottom: 8px;
}
.task-card.risk-high    { border-left: 4px solid #DC2626; border-radius: 0 12px 12px 0; }
.task-card.risk-moderate { border-left: 4px solid #D97706; border-radius: 0 12px 12px 0; }
.task-card.risk-low     { border-left: 4px solid #0F6E56; border-radius: 0 12px 12px 0; }
.risk-badge { display: inline-block; font-size: 11px; padding: 3px 10px; border-radius: 6px; font-weight: 500; }
.risk-badge.high     { background: #FEE2E2; color: #991B1B; }
.risk-badge.moderate { background: #FEF3C7; color: #92400E; }
.risk-badge.low      { background: #D1FAE5; color: #065F46; }
.strategy-card { background: white; padding: 24px 28px; border-radius: 12px; border: 0.5px solid #D0DADA; line-height: 1.8; }

/* ── Brand-aligned buttons (teal instead of red) ── */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background-color: #0F3D3E !important;
    border-color: #0F3D3E !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #1A5C5E !important;
    border-color: #1A5C5E !important;
}

/* ── Teal slider thumb and track ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #1A5C5E !important;
}
.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
    background: #1A5C5E !important;
}

/* ── Teal download buttons ── */
.stDownloadButton > button {
    background-color: #0F3D3E !important;
    border-color: #0F3D3E !important;
    color: white !important;
}
.stDownloadButton > button:hover {
    background-color: #1A5C5E !important;
    border-color: #1A5C5E !important;
}

/* ── Tab font size ── */
.stTabs [data-baseweb="tab"] {
    font-size: 15px !important;
}
</style>
""", unsafe_allow_html=True)

if "tasks" not in st.session_state:
    st.session_state.tasks = []
if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.today().date()
if "progress" not in st.session_state:
    st.session_state.progress = {}

with st.sidebar:
    st.markdown('<div class="sidebar-section">Planning Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Academic", "Operational"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="sidebar-section">Add Task</div>', unsafe_allow_html=True)
    if "task_form_key" not in st.session_state:
        st.session_state["task_form_key"] = 0
    fk = st.session_state["task_form_key"]
    name       = st.text_input("Task name", key=f"task_name_{fk}")
    days_left  = st.number_input("Days until deadline", min_value=1, value=3, key=f"task_days_{fk}")
    est_hours  = st.number_input("Estimated hours", min_value=0.5, value=4.0, step=0.5, key=f"task_hours_{fk}")
    dependency = st.checkbox("Dependency risk", key=f"task_dep_{fk}")
    if mode == "Operational":
        strategic = st.slider("Strategic importance", 1, 5, 3, key=f"task_strat_{fk}")
        impact    = st.slider("Business impact", 1, 5, 3, key=f"task_impact_{fk}")
    else:
        strategic = 1
        impact    = 1
    if st.button("Add Task"):
        if name.strip():
            st.session_state.tasks.append({
                "name": name.strip(), "days_left": days_left, "est_hours": est_hours,
                "strategic_importance": strategic, "business_impact": impact,
                "dependency_risk": dependency,
            })
            st.session_state["task_form_key"] += 1
            st.rerun()
    st.divider()
    st.markdown('<div class="sidebar-section">Planning Settings</div>', unsafe_allow_html=True)
    available_hours  = st.number_input("Available hours per day", min_value=1.0, value=6.0)
    planning_horizon = st.slider("Planning horizon (days)", 1, 30, 7)
    st.session_state.start_date = st.date_input("Start planning date", st.session_state.start_date)

st.session_state["coach_mode"] = mode
st.session_state["coach_available_hours"] = available_hours
st.session_state["coach_planning_horizon"] = planning_horizon

# ── Adjust tasks for progress before computing ──
adjusted_tasks = []
for t in st.session_state.tasks:
    adj = dict(t)
    done = st.session_state.progress.get(t["name"], 0.0)
    remaining = t["est_hours"] - done
    adj["est_hours"] = max(0.0, remaining) if remaining <= 0 else remaining
    adjusted_tasks.append(adj)

# Filter out fully completed tasks for the pipeline (but keep all for display)
active_tasks = [t for t in adjusted_tasks if t["est_hours"] > 0]

df_risk = schedule_df = forecast_df = None
stress_index = expected_loss = high_risk_count = missed_deadlines = on_track_count = 0
total_tasks = len(st.session_state.tasks)

if active_tasks:
    df          = pd.DataFrame(active_tasks)
    df_scored   = compute_priority_scores(df, mode=mode)
    df_risk     = compute_task_risk(df_scored, available_hours, planning_horizon)
    schedule_df = build_schedule(df_risk, available_hours, planning_horizon, mode=mode)
    forecast_df = forecast_system_risk(df_scored, available_hours, planning_horizon)
    stress_index     = compute_weighted_stress(df_risk)
    expected_loss    = round(float(df_risk["expected_loss_hours"].sum()), 1)
    high_risk_count  = int((df_risk["failure_probability"] >= 0.7).sum())
    deadline_risks   = schedule_df.attrs.get("deadline_risks", [])
    missed_deadlines = len(deadline_risks)
    on_track_count   = int((df_risk["failure_probability"] < 0.5).sum())
    # Restore original est_hours for display in task cards
    for i, row in df_risk.iterrows():
        for orig in st.session_state.tasks:
            if orig["name"] == row["name"]:
                df_risk.at[i, "est_hours"] = orig["est_hours"]
                break

def _kpi_color(val, thresholds):
    if val >= thresholds[1]: return "#F09595"
    if val >= thresholds[0]: return "#EF9F27"
    return "#5DCAA5"

stress_col = _kpi_color(stress_index, [40, 75])
risk_col   = "#F09595" if high_risk_count > 0 else "#5DCAA5"
track_col  = "#5DCAA5" if total_tasks == 0 or on_track_count == total_tasks else ("#EF9F27" if on_track_count > 0 else "#F09595")
loss_col   = "#EF9F27" if expected_loss > 0 else "#5DCAA5"

EQUALIZER_SVG = '<svg width="42" height="42" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg"><rect width="34" height="34" rx="9" fill="rgba(255,255,255,0.10)"/><rect x="7" y="18" width="4" rx="1.5" fill="#5DCAA5"><animate attributeName="y" values="18;10;18" dur="1.2s" repeatCount="indefinite"/><animate attributeName="height" values="9;17;9" dur="1.2s" repeatCount="indefinite"/></rect><rect x="13" y="14" width="4" rx="1.5" fill="#5DCAA5"><animate attributeName="y" values="14;8;14" dur="1.2s" repeatCount="indefinite" begin="0.2s"/><animate attributeName="height" values="13;19;13" dur="1.2s" repeatCount="indefinite" begin="0.2s"/></rect><rect x="19" y="10" width="4" rx="1.5" fill="white"><animate attributeName="y" values="10;6;10" dur="1.2s" repeatCount="indefinite" begin="0.4s"/><animate attributeName="height" values="17;21;17" dur="1.2s" repeatCount="indefinite" begin="0.4s"/></rect><rect x="25" y="15" width="4" rx="1.5" fill="#5DCAA5"><animate attributeName="y" values="15;11;15" dur="1.2s" repeatCount="indefinite" begin="0.3s"/><animate attributeName="height" values="12;16;12" dur="1.2s" repeatCount="indefinite" begin="0.3s"/></rect></svg>'

st.markdown(f"""
<div style="background:#0C2829;border-radius:14px;overflow:hidden;margin-bottom:20px;">
    <div style="padding:22px 32px;display:flex;justify-content:space-between;align-items:center;
                border-bottom:1px solid rgba(255,255,255,0.08);
                background:linear-gradient(135deg, #0A2E2F 0%, #134849 60%, #1A5C5E 100%);">
        <div style="display:flex;align-items:center;gap:14px;">
            {EQUALIZER_SVG}
            <div>
                <div style="color:white;font-size:24px;font-weight:600;letter-spacing:-0.3px;">Cadence</div>
                <div style="font-size:13px;color:rgba(255,255,255,0.50);margin-top:2px;">Workload Intelligence</div>
            </div>
        </div>
        <div style="display:flex;gap:6px;">
            <span style="font-size:12px;padding:4px 12px;border-radius:6px;
                         background:rgba(255,255,255,0.08);color:rgba(255,255,255,0.55);
                         border:1px solid rgba(255,255,255,0.12);">{mode}</span>
            <span style="font-size:12px;padding:4px 12px;border-radius:6px;
                         background:rgba(255,255,255,0.08);color:rgba(255,255,255,0.55);
                         border:1px solid rgba(255,255,255,0.12);">{available_hours}h/day · {planning_horizon} days</span>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0;">
        <div style="padding:22px 32px;border-right:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:12px;color:rgba(255,255,255,0.45);letter-spacing:0.5px;margin-bottom:6px;">STRESS INDEX</div>
            <div style="font-size:32px;font-weight:600;color:{stress_col};">{stress_index}<span style="font-size:16px;color:rgba(255,255,255,0.35);">/100</span></div>
        </div>
        <div style="padding:22px 32px;border-right:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:12px;color:rgba(255,255,255,0.45);letter-spacing:0.5px;margin-bottom:6px;">HOURS AT RISK</div>
            <div style="font-size:32px;font-weight:600;color:{loss_col};">{expected_loss}h</div>
        </div>
        <div style="padding:22px 32px;border-right:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:12px;color:rgba(255,255,255,0.45);letter-spacing:0.5px;margin-bottom:6px;">HIGH RISK</div>
            <div style="font-size:32px;font-weight:600;color:{risk_col};">{high_risk_count}</div>
        </div>
        <div style="padding:22px 32px;">
            <div style="font-size:12px;color:rgba(255,255,255,0.45);letter-spacing:0.5px;margin-bottom:6px;">ON TRACK</div>
            <div style="font-size:32px;font-weight:600;color:{track_col};">{on_track_count}<span style="font-size:16px;color:rgba(255,255,255,0.35);">/{total_tasks}</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if missed_deadlines > 0 and schedule_df is not None:
    for risk in schedule_df.attrs.get("deadline_risks", []):
        st.warning(
            f"**{risk['task']}** cannot be completed before its deadline "
            f"(Day {risk['deadline_day']}). {risk['unfinished_hours']}h will remain "
            "unfinished under current capacity settings.")

EMPTY_STATE = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 40px;text-align:center;">
    <h2 style="font-size:22px;font-weight:600;color:#1E293B;margin:0 0 10px 0;">Welcome to Cadence</h2>
    <p style="font-size:14px;color:#7A8B8D;max-width:380px;line-height:1.6;margin:0;">
        Open the sidebar to add your first task, or use Smart Import to get started.
    </p>
</div>
"""

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Situation", "Execution Plan", "Strategic Outlook", "Sync", "Smart Import", "Daily Plan"])

with tab1:
    if df_risk is not None:
        render_overview(df_risk, schedule_df, available_hours, planning_horizon, mode)
    else:
        st.markdown(EMPTY_STATE, unsafe_allow_html=True)
with tab2:
    if schedule_df is not None:
        render_calendar(schedule_df, available_hours)
    else:
        st.markdown(EMPTY_STATE, unsafe_allow_html=True)
with tab3:
    if forecast_df is not None:
        render_strategic(forecast_df, stress_index, df_risk)
    else:
        st.markdown(EMPTY_STATE, unsafe_allow_html=True)
with tab4:
    render_coach()
with tab5:
    render_import(mode)
with tab6:
    render_daily_plan(mode, available_hours, planning_horizon)