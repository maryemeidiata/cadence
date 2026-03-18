import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ai_explainer import generate_strategic_advice


def render_strategic(forecast_df, stress_index, df_risk):

    st.markdown("## Risk forecast")
    st.caption("How your stress index and expected loss change as you add or reduce daily capacity.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Stress Index — left Y — dark teal
    fig.add_trace(
        go.Scatter(
            x=forecast_df["capacity_hours_per_day"],
            y=forecast_df["stress_index"],
            mode="lines+markers",
            line=dict(color="#0F3D3E", width=3),
            marker=dict(size=7, color="#0F3D3E"),
            name="Stress Index",
            hovertemplate="<b>%{x}h/day</b><br>Stress: %{y:.1f}/100<extra></extra>",
        ),
        secondary_y=False,
    )

    # Expected Loss — right Y — amber dotted
    fig.add_trace(
        go.Scatter(
            x=forecast_df["capacity_hours_per_day"],
            y=forecast_df["expected_loss_hours"],
            mode="lines+markers",
            line=dict(color="#D97706", width=2, dash="dot"),
            marker=dict(size=6, color="#D97706"),
            name="Expected Loss (h)",
            hovertemplate="<b>%{x}h/day</b><br>Loss: %{y:.1f}h<extra></extra>",
        ),
        secondary_y=True,
    )

    # Current capacity marker
    current_cap = forecast_df[forecast_df["capacity_delta"] == 0]
    if len(current_cap) > 0:
        cap_val = float(current_cap["capacity_hours_per_day"].iloc[0])
        stress_val = float(current_cap["stress_index"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=[cap_val], y=[stress_val],
                mode="markers+text",
                marker=dict(size=14, color="#0F3D3E", symbol="circle"),
                text=["Current"],
                textposition="top center",
                textfont=dict(size=12, color="#0F3D3E"),
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

    fig.update_xaxes(title_text="Daily capacity (hours)", showgrid=True, gridcolor="#EEF1F2", dtick=1)
    fig.update_yaxes(title_text="Stress index", showgrid=True, gridcolor="#EEF1F2", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="Expected loss (hours)", showgrid=False, secondary_y=True)

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#D0DADA"),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Strategic summary")

    advice = generate_strategic_advice(stress_index, df_risk, forecast_df=forecast_df)

    st.markdown(f'<div class="strategy-card">{advice}</div>', unsafe_allow_html=True)