# Cadence
### Workload Prioritization & Strategic Planning Engine

Cadence helps users manage task workloads intelligently. It goes beyond a simple to-do list by modelling risk, building an optimized schedule, providing AI-powered strategic advice, and generating personalised daily plans.

---

## Features

- **Risk Scoring** — Scores tasks by urgency, strategic importance, business impact, and dependency risk using a sigmoid failure probability model
- **Execution Timeline** — Gantt chart showing a day-by-day schedule built with mode-aware algorithms (Earliest Deadline First or Value Density)
- **Risk Forecast** — Dual-axis chart showing how stress and expected loss change across capacity scenarios
- **Strategic Summary** — AI-generated analysis of risk distribution, worst-consequence tasks, and capacity sensitivity
- **Sync — AI Strategy Coach** — Conversational chatbot powered by Cohere with tool use: ask questions, run what-if scenarios, and compare options grounded in your live data
- **Smart Task Import** — Paste text or upload a PDF and the AI extracts structured tasks using a two-pass LLM chain (extract + validate)
- **Daily Plan** — AI-generated hour-by-hour schedule that respects your personal availability windows
- **Progress Tracking** — Log hours completed per task; the risk model recalculates with remaining hours
- **Exports** — Download deadlines as a calendar file (.ics) or a full workload report (.pdf); export daily plan time blocks to your calendar

---

## Planning Modes

| Mode | Strategy | Best for |
|------|----------|----------|
| Academic | Earliest Deadline First | Students, deadline-driven work |
| Operational | Value Density (priority/hour) | Teams, business planning |

---

## Setup

```bash
pip install -r requirements.txt
export CO_API_KEY="your_cohere_api_key"
streamlit run app.py
```

Get a free Cohere API key at [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).

For Streamlit Cloud deployment, add `CO_API_KEY` in Settings > Secrets.

---

## File Structure

```
cadence/
├── app.py              # Main app — layout, header, KPIs, tabs
├── scoring.py          # Priority scoring engine
├── risk_model.py       # Failure probability & stress index
├── scheduler.py        # Mode-aware task scheduler
├── ai_explainer.py     # Strategic recommendation engine
├── llm_coach.py        # Sync — AI Coach with Cohere tool use
├── llm_import.py       # Smart Import — two-pass LLM extraction
├── llm_daily_plan.py   # Daily Plan — structured LLM output
├── ui_overview.py      # Situation tab (task cards, progress, exports)
├── ui_calendar.py      # Execution Plan tab (Gantt chart)
├── ui_strategic.py     # Strategic Outlook tab (forecast + summary)
├── ui_coach.py         # Sync chat interface
├── ui_import.py        # Smart Import interface
├── ui_daily_plan.py    # Daily Plan visual timeline
├── requirements.txt    # Dependencies
└── .gitignore          # Keeps API keys out of the repo
```

---

## LLM Architecture

Three distinct non-straightforward LLM integration patterns:

### 1. Sync — AI Strategy Coach
- **Pattern:** Multi-turn conversational chatbot with tool use
- **Tools:** `get_current_analysis`, `explain_task_risk`, `run_what_if_scenario`, `compare_scenarios`
- **Flow:** User message → Cohere selects tools → tools execute locally (re-running risk model/scheduler) → results fed back → Cohere generates grounded response
- **Non-straightforward because:** Multi-turn history, automatic tool selection, tool results that re-run the data pipeline, iterative tool-calling loop

### 2. Smart Task Import
- **Pattern:** Two-pass LLM chain with post-processing
- **Pass 1:** Extract tasks from unstructured text/PDF as structured JSON
- **Pass 2:** Validate and adjust hour estimates, deadlines, dependencies
- **Post-processing:** Normalize key names, clamp ranges, deduplicate, conform to internal schema
- **Non-straightforward because:** Two sequential LLM calls where output of first feeds the second, plus robust Python post-processing handling multilingual LLM output

### 3. Daily Plan
- **Pattern:** Structured JSON output with context-aware reasoning
- **Flow:** LLM receives task data + risk scores + progress + availability windows → generates hour-by-hour time blocks as JSON → Python validates and renders as visual timeline
- **Non-straightforward because:** LLM reasons about time constraints, energy management, and personal scheduling; outputs precisely formatted JSON that drives the UI; availability parsed from structured time pickers

---

## Built with

- [Streamlit](https://streamlit.io)
- [Plotly](https://plotly.com)
- [Pandas](https://pandas.pydata.org)
- [NumPy](https://numpy.org)
- [Cohere](https://cohere.com) — Command A model with tool use
- [ReportLab](https://www.reportlab.com) — PDF generation
- [PyPDF2](https://pypdf2.readthedocs.io) — PDF text extraction

---

*Assignment 2 — Prototyping with LLMs*