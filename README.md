# Cadence
# Workload Prioritization & Strategic Planning Engine

Cadence is a Streamlit prototype that helps users manage task workloads intelligently. It goes beyond a simple to-do list by modelling risk, building an optimized schedule, and providing strategic recommendations based on your available capacity and deadlines.


## What it does

- **Scores tasks** by urgency, strategic importance, business impact, and dependency risk
- **Estimates failure probability** per task using a sigmoid risk model with per-task competition pressure
- **Builds a day-by-day schedule** using mode-aware algorithms (Earliest Deadline First for Academic, Value Density for Operational)
- **Visualizes the schedule** as an interactive Gantt chart colour-coded by risk level
- **Forecasts system stress** across capacity scenarios
- **Generates strategic recommendations** based on risk distribution, worst-consequence tasks, and capacity sensitivity



## Planning Modes



Academic Mode -  Earliest Deadline First, Students, deadline-driven work
Operational Mode - Value Density (priority/hour), Teams, business planning



## Setup

```bash
pip install streamlit pandas numpy plotly
streamlit run app.py
```

---

## File Structure

cadence/
├── app.py              # Main Streamlit app
├── scoring.py          # Priority scoring engine
├── risk_model.py       # Failure probability & stress index
├── scheduler.py        # Mode-aware task scheduler
├── ai_explainer.py     # Strategic recommendation engine
├── ui_overview.py      # Situation tab
├── ui_calendar.py      # Execution Plan tab (Gantt chart)
├── ui_strategic.py     # Strategic Outlook tab
└── requirements.txt

## Built with

- [Streamlit](https://streamlit.io)
- [Plotly](https://plotly.com)
- [Pandas](https://pandas.pydata.org)
- [NumPy](https://numpy.org)

---

*Assignment 1 — Prototyping with Streamlit*