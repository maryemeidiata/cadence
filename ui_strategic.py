import streamlit as st
from ai_explainer import generate_strategic_advice


def render_strategic(forecast_df, stress_index, df_risk):

    st.markdown("## Risk Forecast")
    st.caption(
        "Shows how the weighted stress index changes as daily available hours increase or decrease. "
        "Use this to quantify the value of adding capacity."
    )

    chart_df = forecast_df.set_index("capacity_hours_per_day")[["stress_index", "expected_loss_hours"]]
    chart_df = chart_df.rename(columns={
        "stress_index": "Stress Index",
        "expected_loss_hours": "Expected Loss (hours)"
    })
    st.line_chart(chart_df, height=320)

    st.markdown("## Strategic Summary")

    # Pass forecast_df so the explainer can read capacity sensitivity
    advice = generate_strategic_advice(stress_index, df_risk, forecast_df=forecast_df)

    st.markdown(f"""
    <div style="
        background: white;
        padding: 28px 32px;
        border-radius: 16px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.06);
        line-height: 1.8;
    ">
        {advice}
    </div>
    """, unsafe_allow_html=True)