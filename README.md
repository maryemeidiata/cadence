# Cadence
### Workload Prioritization & Strategic Planning Engine

Cadence is a Streamlit prototype that helps users manage task workloads intelligently. It goes beyond a simple to-do list by modelling risk, building an optimized schedule, providing AI-powered strategic advice, and generating personalised daily plans.

---

## What it does

- **Scores tasks** by urgency, strategic importance, business impact, and dependency risk
- **Estimates failure probability** per task using a sigmoid risk model with per-task competition pressure
- **Builds a day-by-day schedule** using mode-aware algorithms (Earliest Deadline First for Academic, Value Density for Operational)
- **Visualizes the schedule** as an interactive Gantt chart colour-coded by risk level
- **Forecasts system stress** across capacity scenarios
- **Generates strategic recommendations** based on risk distribution, worst-consequence tasks, and capacity sensitivity
- **Pulse — AI Strategy Coach** powered by Cohere with tool use: ask questions about your workload, run what-if scenarios, compare options, and get actionable advice grounded in your live data
- **Smart Task Import** using a two-pass LLM chain: paste text or upload a PDF (syllabus, email, meeting notes) and the AI extracts structured tasks with validation
- **Daily Plan** — AI-generated hour-by-hour schedule for today that respects your personal availability and prioritises by risk and deadline
- **Progress tracking** with visual progress bars on each task
- **Export** your schedule as a calendar file (.ics) or a PDF report

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

For Streamlit Cloud deployment, add `CO_API_KEY` in the app dashboard under Settings → Secrets.

---

## File Structure

```
cadence/
├── app.py              # Main Streamlit app — UI layout, KPIs, tabs
├── scoring.py          # Priority scoring engine
├── risk_model.py       # Failure probability & stress index
├── scheduler.py        # Mode-aware task scheduler
├── ai_explainer.py     # Strategic recommendation engine
├── llm_coach.py        # Pulse — AI Coach with Cohere tool use
├── llm_import.py       # Smart Import — two-pass LLM extraction chain
├── llm_daily_plan.py   # Daily Plan — structured LLM output generation
├── ui_overview.py      # Situation tab (task cards, progress, exports)
├── ui_calendar.py      # Execution Plan tab (Gantt chart)
├── ui_strategic.py     # Strategic Outlook tab
├── ui_coach.py         # Pulse chat interface
├── ui_import.py        # Smart Import interface
├── ui_daily_plan.py    # Daily Plan visual timeline
├── requirements.txt    # Python dependencies
└── .gitignore          # Keeps API keys out of the repo
```

---

## LLM Architecture

Cadence uses three distinct, non-straightforward LLM integration patterns:

### 1. Pulse — AI Strategy Coach
- **Pattern:** Multi-turn conversational chatbot with tool use
- **Tools:** `get_current_analysis`, `explain_task_risk`, `run_what_if_scenario`, `compare_scenarios`
- **How it works:** User sends a message → Cohere decides which tools to call → tools execute locally (running the risk model and scheduler with modified parameters) → results are fed back to Cohere → Cohere generates a grounded natural-language response
- **Why non-straightforward:** Multi-turn conversation history, automatic tool selection by the LLM, tool results that re-run the entire data pipeline, iterative tool-calling loop (up to 5 rounds)

### 2. Smart Task Import
- **Pattern:** Two-pass LLM chain with post-processing
- **Pass 1 (Extract):** LLM parses unstructured text (or PDF content) into structured task JSON
- **Pass 2 (Validate):** A second LLM call reviews the extracted tasks for reasonableness — checking hour estimates, deadlines, and dependencies
- **Post-processing:** Python normalizes key names (handling multilingual LLM output), clamps value ranges, deduplicates by name, and conforms to the internal task schema
- **Why non-straightforward:** Two sequential LLM calls where the output of the first feeds the second, plus substantial Python post-processing of LLM output

### 3. Daily Plan
- **Pattern:** Structured JSON output generation with context-aware reasoning
- **How it works:** The LLM receives the full computed risk data, the user's progress, AND their free-form availability description (e.g. "busy 9-12, meeting at 3"). It generates a structured JSON array of time blocks with task assignments, sub-activities, and reasoning
- **Post-processing:** Python validates the JSON structure, normalizes time formats, and renders the blocks as a visual timeline with color-coded task categories
- **Why non-straightforward:** The LLM must reason about time constraints, task priorities, energy management, and personal scheduling — then output precisely formatted JSON that drives the visual UI. The free-form availability input showcases the LLM's ability to parse natural language constraints.

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