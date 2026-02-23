import streamlit as st
import pandas as pd
from datetime import datetime

from scoring import compute_priority_scores
from scheduler import build_schedule
from risk_model import compute_task_risk, forecast_system_risk, compute_weighted_stress

from ui_overview import render_overview
from ui_calendar import render_calendar
from ui_strategic import render_strategic


st.set_page_config(layout="wide", page_title="Cadence")

# -------------------------
# CSS — no input overrides, let Streamlit handle those natively
# -------------------------
st.markdown("""
<style>
header {visibility: hidden;}

/* ── Background gradient ── */
.stApp {
    background: linear-gradient(160deg, #DDE8EC 0%, #E8EDF2 100%);
    min-height: 100vh;
}

/* ── Sidebar background only — no input/text overrides ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F3D3E 0%, #1A5C5E 100%) !important;
    border-right: none;
}

/* Labels and plain text in sidebar */
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.90) !important;
}

[data-testid="stSidebar"] p {
    color: rgba(255,255,255,0.90) !important;
}

/* Section uppercase headers */
[data-testid="stSidebar"] .sidebar-section {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.50) !important;
    margin-bottom: 10px;
    margin-top: 4px;
}

/* Sidebar button */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.14) !important;
    border: 1px solid rgba(255,255,255,0.30) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    width: 100%;
    font-weight: 600;
    transition: background 0.2s;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.26) !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.18) !important;
}

/* ── Teal tab underline ── */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #1A5C5E !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #1A5C5E !important;
}

/* ── Header band ── */
.header-band {
    background: linear-gradient(135deg, #0F3D3E 0%, #1A5C5E 100%);
    padding: 56px 60px;
    border-radius: 20px;
    color: white;
    margin-bottom: -40px;
    box-shadow: 0px 16px 40px rgba(15, 61, 62, 0.25);
}

.header-tag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.70);
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* ── KPI strip ── */
.kpi-strip {
    background: white;
    padding: 30px 60px;
    border-radius: 18px;
    box-shadow: 0px 12px 30px rgba(15, 61, 62, 0.10);
    display: flex;
    justify-content: space-between;
    margin: 0 80px 50px 80px;
}

.kpi-number         { font-size: 36px; font-weight: 600; color: #1E293B; }
.kpi-number.red     { color: #DC2626; }
.kpi-number.amber   { color: #D97706; }
.kpi-number.green   { color: #16A34A; }
.kpi-label          { font-size: 13px; color: #6B7280; margin-top: 4px; }

.kpi-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-red    { background: #DC2626; }
.dot-amber  { background: #D97706; }
.dot-green  { background: #16A34A; }
</style>
""", unsafe_allow_html=True)


# -------------------------
# SESSION STATE
# -------------------------
if "tasks" not in st.session_state:
    st.session_state.tasks = []

if "start_date" not in st.session_state:
    st.session_state.start_date = datetime.today().date()


# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:

    st.markdown('<div class="sidebar-section">Planning Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Academic", "Operational"], label_visibility="collapsed")

    st.divider()

    st.markdown('<div class="sidebar-section">Add Task</div>', unsafe_allow_html=True)

    name       = st.text_input("Task name")
    days_left  = st.number_input("Days until deadline", min_value=1, value=3)
    est_hours  = st.number_input("Estimated hours", min_value=0.5, value=4.0, step=0.5)
    dependency = st.checkbox("Dependency risk")

    if mode == "Operational":
        strategic = st.slider("Strategic importance", 1, 5, 3)
        impact    = st.slider("Business impact", 1, 5, 3)
    else:
        strategic = 1
        impact    = 1

    if st.button("Add Task"):
        if name.strip():
            st.session_state.tasks.append({
                "name":                 name.strip(),
                "days_left":            days_left,
                "est_hours":            est_hours,
                "strategic_importance": strategic,
                "business_impact":      impact,
                "dependency_risk":      dependency,
            })

    st.divider()

    st.markdown('<div class="sidebar-section">Planning Settings</div>', unsafe_allow_html=True)
    available_hours  = st.number_input("Available hours per day", min_value=1.0, value=6.0)
    planning_horizon = st.slider("Planning horizon (days)", 1, 30, 7)
    st.session_state.start_date = st.date_input(
        "Start planning date", st.session_state.start_date
    )


# -------------------------
# HEADER
# -------------------------
st.markdown("""
<div class="header-band">
    <div class="header-tag">Workload Intelligence</div>
    <h1 style="margin:0; font-size:52px; font-weight:700; letter-spacing:-1.5px;">Cadence</h1>
    <p style="margin:10px 0 0 0; font-weight:400; font-size:18px; color:rgba(255,255,255,0.65);">
        Workload Prioritization &amp; Strategic Planning Engine
    </p>
</div>
""", unsafe_allow_html=True)


# -------------------------
# COMPUTE
# -------------------------
df_risk          = None
schedule_df      = None
forecast_df      = None
stress_index     = 0
expected_loss    = 0
high_risk_count  = 0
missed_deadlines = 0

if st.session_state.tasks:

    df          = pd.DataFrame(st.session_state.tasks)
    df_scored   = compute_priority_scores(df, mode=mode)
    df_risk     = compute_task_risk(df_scored, available_hours, planning_horizon)
    schedule_df = build_schedule(df_risk, available_hours, planning_horizon, mode=mode)
    forecast_df = forecast_system_risk(df_scored, available_hours, planning_horizon)

    stress_index     = compute_weighted_stress(df_risk)
    expected_loss    = round(float(df_risk["expected_loss_hours"].sum()), 1)
    high_risk_count  = int((df_risk["failure_probability"] >= 0.7).sum())
    deadline_risks   = schedule_df.attrs.get("deadline_risks", [])
    missed_deadlines = len(deadline_risks)


# -------------------------
# KPI helpers
# -------------------------
def _stress_cls(s):
    if s >= 75: return "red"
    if s >= 40: return "amber"
    return "green"

def _dot(cls):
    return f'<span class="kpi-dot dot-{cls}"></span>'

stress_cls = _stress_cls(stress_index)
hr_cls     = "red"   if high_risk_count  > 0 else "green"
missed_cls = "red"   if missed_deadlines > 0 else "green"
loss_cls   = "amber" if expected_loss    > 0 else "green"


# -------------------------
# KPI STRIP
# -------------------------
st.markdown(f"""
<div class="kpi-strip">
    <div>
        <div class="kpi-number {stress_cls}">{_dot(stress_cls)}{stress_index}/100</div>
        <div class="kpi-label">Weighted Stress Index</div>
    </div>
    <div>
        <div class="kpi-number {loss_cls}">{_dot(loss_cls)}{expected_loss}h</div>
        <div class="kpi-label">Hours at Risk</div>
    </div>
    <div>
        <div class="kpi-number {hr_cls}">{_dot(hr_cls)}{high_risk_count}</div>
        <div class="kpi-label">High Risk Tasks</div>
    </div>
    <div>
        <div class="kpi-number {missed_cls}">{_dot(missed_cls)}{missed_deadlines}</div>
        <div class="kpi-label">Missed Deadlines</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Deadline warnings
if missed_deadlines > 0 and schedule_df is not None:
    for risk in schedule_df.attrs.get("deadline_risks", []):
        st.warning(
            f"**{risk['task']}** cannot be completed before its deadline "
            f"(Day {risk['deadline_day']}). {risk['unfinished_hours']}h will remain "
            "unfinished under current capacity settings."
        )


# -------------------------
# EMPTY STATE
# -------------------------
EMPTY_STATE = """
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 40px;text-align:center;">
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:24px;opacity:0.85;">
        <rect x="8" y="8" width="48" height="48" rx="10" stroke="#1A5C5E" stroke-width="3" fill="none"/>
        <line x1="20" y1="24" x2="44" y2="24" stroke="#1A5C5E" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="20" y1="32" x2="44" y2="32" stroke="#1A5C5E" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="20" y1="40" x2="34" y2="40" stroke="#1A5C5E" stroke-width="2.5" stroke-linecap="round"/>
        <circle cx="50" cy="50" r="10" fill="#1A5C5E"/>
        <line x1="50" y1="45" x2="50" y2="55" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="45" y1="50" x2="55" y2="50" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
    </svg>
    <h2 style="font-size:24px;font-weight:700;color:#1E293B;margin:0 0 12px 0;">Welcome to Cadence</h2>
    <p style="font-size:15px;color:#6B7280;max-width:380px;line-height:1.6;margin:0 0 28px 0;">Add your first task in the sidebar to start analysing your workload, modelling risk, and building your schedule.</p>
    <div style="display:flex;align-items:center;gap:8px;background:#0F3D3E;color:white;padding:12px 24px;border-radius:10px;font-size:14px;font-weight:600;">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 8L6 4M6 12L10 8" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Add your first task in the sidebar
    </div>
</div>
"""

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs(["Situation", "Execution Plan", "Strategic Outlook"])

with tab1:
    if df_risk is not None:
        render_overview(df_risk)
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
