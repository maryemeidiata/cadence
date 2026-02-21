import streamlit as st


def render_overview(df_risk):

    st.markdown("## Task Risk Analysis")

    for i, row in df_risk.iterrows():

        col1, col2 = st.columns([6,1])

        with col1:
            prob = row["failure_probability"] * 100
            st.markdown(f"**{row['name']}** — {row['est_hours']}h | {round(prob,1)}% risk")

        with col2:
            if st.button("❌", key=f"delete_{i}"):
                st.session_state.tasks.pop(i)
                st.rerun()

    st.markdown("---")