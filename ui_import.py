"""
UI for the Smart Import feature — paste unstructured text or upload a PDF,
and the LLM extracts structured tasks with a two-pass chain.
"""

import streamlit as st
from llm_import import extract_tasks_from_text


def _extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF file."""
    try:
        import PyPDF2
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except ImportError:
        st.error("PyPDF2 is not installed. Run: pip install PyPDF2")
        return ""
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


def render_import(mode: str) -> None:
    """Render the Smart Import interface."""

    st.markdown("## Smart Task Import")
    st.caption(
        "Paste text or upload a PDF (syllabus, project brief, email, meeting notes). "
        "The AI will extract tasks, estimate effort, and validate the results."
    )

    # ── Input mode: text or PDF ──
    input_mode = st.radio(
        "Input method",
        ["Paste text", "Upload PDF"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_text = ""

    if input_mode == "Paste text":
        raw_text = st.text_area(
            "Paste your text here",
            height=180,
            placeholder="e.g., 'I have a research paper due next Friday (8-10 pages), "
                        "a group presentation on Wednesday about market analysis, "
                        "and I need to submit my lab report by tomorrow...'",
        )
    else:
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_pdf:
            with st.spinner("Reading PDF..."):
                raw_text = _extract_pdf_text(uploaded_pdf)
            if raw_text:
                with st.expander("Extracted text from PDF", expanded=False):
                    st.text(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))

    col1, col2 = st.columns([1, 4])
    with col1:
        extract_btn = st.button("Extract Tasks", type="primary")

    # ── Run extraction ──
    if extract_btn:
        api_key = ""
        try:
            api_key = st.secrets["CO_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass
        if not api_key:
            import os
            api_key = os.environ.get("CO_API_KEY", "")
        if not api_key:
            st.error("Cohere API key not configured. Set CO_API_KEY in your environment or Streamlit secrets.")
            return

        if not raw_text.strip():
            st.warning("Please paste some text or upload a PDF first.")
            return

        with st.spinner("Analysing your text..."):
            result = extract_tasks_from_text(raw_text, mode=mode)

        if result["error"]:
            st.error(result["error"])
            return

        st.session_state["import_results"] = result
        st.rerun()

    # ── Display extracted tasks for review ──
    if "import_results" in st.session_state:
        result = st.session_state["import_results"]
        tasks = result["tasks"]
        notes = result["validation_notes"]

        if not tasks:
            st.info("No tasks were extracted. Try pasting more detailed text.")
            return

        st.markdown("### Extracted Tasks")
        st.caption("Review the AI's extraction below. Edit any values before adding to your workload.")

        # Track which tasks the user wants to add
        if "import_selections" not in st.session_state:
            st.session_state["import_selections"] = [True] * len(tasks)
        if len(st.session_state["import_selections"]) != len(tasks):
            st.session_state["import_selections"] = [True] * len(tasks)

        # ── Column headers ──
        col_check, col_name, col_hours, col_days, col_dep = st.columns([0.5, 3, 1.5, 1.5, 1])
        with col_check:
            st.markdown("**Add**")
        with col_name:
            st.markdown("**Task Name**")
        with col_hours:
            st.markdown("**Est. Hours**")
        with col_days:
            st.markdown("**Days Left**")
        with col_dep:
            st.markdown("**Dep.**")

        # ── Task rows ──
        for i, (task, note) in enumerate(zip(tasks, notes)):
            with st.container():
                col_check, col_name, col_hours, col_days, col_dep = st.columns([0.5, 3, 1.5, 1.5, 1])

                with col_check:
                    selected = st.checkbox("", value=True, key=f"import_sel_{i}", label_visibility="collapsed")
                    st.session_state["import_selections"][i] = selected

                with col_name:
                    task["name"] = st.text_input("Task", value=task["name"], key=f"import_name_{i}", label_visibility="collapsed")

                with col_hours:
                    task["est_hours"] = st.number_input("Hours", value=task["est_hours"], min_value=0.5, step=0.5, key=f"import_hours_{i}", label_visibility="collapsed")

                with col_days:
                    task["days_left"] = st.number_input("Days", value=task["days_left"], min_value=1, key=f"import_days_{i}", label_visibility="collapsed")

                with col_dep:
                    task["dependency_risk"] = st.checkbox("Dep", value=task["dependency_risk"], key=f"import_dep_{i}", label_visibility="collapsed")

                # Show validation note (skip generic/useless ones)
                if note and note.lower() not in ["dependency_risk", "false", "true", ""]:
                    st.caption(f"Note: {note}")

                st.markdown("---")

        # ── Action buttons ──
        col_add, col_clear = st.columns([1, 1])

        with col_add:
            if st.button("Add selected tasks to workload", type="primary"):
                added = 0
                for i, task in enumerate(tasks):
                    if st.session_state["import_selections"][i]:
                        st.session_state.tasks.append(task)
                        added += 1

                if added > 0:
                    st.success(f"Added {added} task{'s' if added > 1 else ''} to your workload.")
                    del st.session_state["import_results"]
                    if "import_selections" in st.session_state:
                        del st.session_state["import_selections"]
                    st.rerun()
                else:
                    st.warning("No tasks selected.")

        with col_clear:
            if st.button("Discard all"):
                del st.session_state["import_results"]
                if "import_selections" in st.session_state:
                    del st.session_state["import_selections"]
                st.rerun()