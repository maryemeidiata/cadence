import pandas as pd
import numpy as np


# ------------------------------------------------------------------
# HTML helpers
# ------------------------------------------------------------------

def _dot(colour: str) -> str:
    """Renders a small coloured circle as a status indicator."""
    return (
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;background:{colour};'
        f'margin-right:8px;vertical-align:middle;"></span>'
    )


STATUS_COLOURS = {
    "critical":    "#DC2626",   # red
    "elevated":    "#D97706",   # amber
    "manageable":  "#CA8A04",   # yellow-gold
    "stable":      "#16A34A",   # green
    "info":        "#2563EB",   # blue
    "warning":     "#9333EA",   # purple — used for concentration/distribution findings
}


def generate_strategic_advice(
    stress_index: float,
    df_risk: pd.DataFrame,
    forecast_df: pd.DataFrame | None = None,
) -> str:
    """
    Generates structured strategic commentary.

    Visual signals use colored dot indicators instead of emojis.
    Analysis covers:
      1. System-level stress level
      2. Risk distribution (variance / concentration)
      3. Worst-consequence task
      4. High-risk and moderate-risk task lists
      5. Drop candidates (low impact, high risk)
      6. Capacity sensitivity from forecast_df
      7. Closing strategic suggestion
    """

    blocks: list[str] = []

    # ------------------------------------------------------------------
    # 1. System-level stress
    # ------------------------------------------------------------------
    if stress_index >= 75:
        c = STATUS_COLOURS["critical"]
        blocks.append(
            f'{_dot(c)}<b>Critical Load:</b> The weighted stress index indicates that your '
            'high-impact tasks are at severe risk of failure. '
            'Immediate workload redistribution or deadline renegotiation is required.'
        )
    elif stress_index >= 50:
        c = STATUS_COLOURS["elevated"]
        blocks.append(
            f'{_dot(c)}<b>Elevated Stress:</b> Execution risk is trending upward on your '
            'most important tasks. Consider reducing non-essential work or '
            'extending your planning horizon.'
        )
    elif stress_index >= 25:
        c = STATUS_COLOURS["manageable"]
        blocks.append(
            f'{_dot(c)}<b>Manageable Pressure:</b> Current allocation is sustainable, '
            'but contingency buffer is limited. Monitor closely.'
        )
    else:
        c = STATUS_COLOURS["stable"]
        blocks.append(
            f'{_dot(c)}<b>Stable Configuration:</b> Resource allocation remains within '
            'safe limits. Maintain current pacing.'
        )

    # ------------------------------------------------------------------
    # 2. Risk distribution — concentrated vs. diffuse
    # ------------------------------------------------------------------
    probs     = df_risk["failure_probability"].values
    risk_std  = float(np.std(probs))
    risk_mean = float(np.mean(probs))

    if len(probs) > 1:
        if risk_std > 0.25:
            c = STATUS_COLOURS["warning"]
            blocks.append(
                f'{_dot(c)}<b>Risk Concentration Detected:</b> Failure probability variance is high '
                f'(σ = {round(risk_std, 2)}). Risk is not spread evenly — a small number of tasks '
                'are driving most of the exposure. Focus intervention there, not across the board.'
            )
        elif risk_std < 0.08 and risk_mean > 0.3:
            c = STATUS_COLOURS["info"]
            blocks.append(
                f'{_dot(c)}<b>Diffuse Risk Pattern:</b> Failure probabilities are uniformly elevated '
                f'(σ = {round(risk_std, 2)}, mean = {round(risk_mean * 100, 1)}%). '
                'This suggests a systemic capacity problem, not a task-specific one. '
                'Increasing daily available hours is likely more effective than re-prioritising.'
            )

    # ------------------------------------------------------------------
    # 3. Worst-consequence task
    # ------------------------------------------------------------------
    if len(df_risk) > 0:
        worst      = df_risk.loc[df_risk["expected_loss_hours"].idxmax()]
        worst_name = worst["name"]
        worst_loss = round(float(worst["expected_loss_hours"]), 1)
        worst_prob = round(float(worst["failure_probability"]) * 100, 1)

        if worst_prob >= 30:
            c = STATUS_COLOURS["elevated"]
            blocks.append(
                f'{_dot(c)}<b>Highest-Consequence Task:</b> <i>{worst_name}</i> carries '
                f'{worst_prob}% failure probability and represents {worst_loss}h of '
                'projected rework if it misses its deadline. '
                'Prioritise early execution or add buffer to this task specifically.'
            )

    # ------------------------------------------------------------------
    # 4. High-risk and moderate-risk task lists
    # ------------------------------------------------------------------
    high_risk = df_risk[df_risk["failure_probability"] >= 0.7]
    mod_risk  = df_risk[
        (df_risk["failure_probability"] >= 0.4) &
        (df_risk["failure_probability"] <  0.7)
    ]

    if len(high_risk) > 0:
        names = ", ".join(f"<i>{n}</i>" for n in high_risk["name"].tolist())
        c = STATUS_COLOURS["critical"]
        blocks.append(
            f'{_dot(c)}<b>High-Risk Tasks (≥ 70%):</b> {names}. '
            'These tasks have a high probability of missing their deadlines under '
            'current capacity. Reallocate additional hours or renegotiate scope.'
        )
    elif len(mod_risk) > 0:
        names = ", ".join(f"<i>{n}</i>" for n in mod_risk["name"].tolist())
        c = STATUS_COLOURS["manageable"]
        blocks.append(
            f'{_dot(c)}<b>Moderate-Risk Tasks (40–70%):</b> {names}. '
            'Monitor closely — these could escalate if capacity tightens.'
        )

    # ------------------------------------------------------------------
    # 5. Drop candidates (low impact, high risk)
    # ------------------------------------------------------------------
    if "impact_weight" in df_risk.columns:
        drop_candidates = df_risk[
            (df_risk["failure_probability"] >= 0.5) &
            (df_risk["impact_weight"] <= 0.3)
        ]
        if len(drop_candidates) > 0:
            names = ", ".join(f"<i>{n}</i>" for n in drop_candidates["name"].tolist())
            c = STATUS_COLOURS["info"]
            blocks.append(
                f'{_dot(c)}<b>Drop Candidates:</b> {names} have elevated failure risk but low '
                'strategic impact. Consider deferring or removing them to free '
                'capacity for higher-value work.'
            )

    # ------------------------------------------------------------------
    # 6. Capacity sensitivity
    # ------------------------------------------------------------------
    if forecast_df is not None and len(forecast_df) > 1:
        current_row  = forecast_df[forecast_df["capacity_delta"] == 0]
        plus_one_row = forecast_df[forecast_df["capacity_delta"] == 1]

        if len(current_row) > 0 and len(plus_one_row) > 0:
            current_stress  = float(current_row["stress_index"].values[0])
            plus_one_stress = float(plus_one_row["stress_index"].values[0])
            stress_drop     = round(current_stress - plus_one_stress, 1)

            if stress_drop > 0:
                c = STATUS_COLOURS["info"]
                blocks.append(
                    f'{_dot(c)}<b>Capacity Sensitivity:</b> Adding just <b>1 hour/day</b> would reduce '
                    f'your stress index by <b>{stress_drop} points</b> '
                    f'({round(current_stress, 1)} → {round(plus_one_stress, 1)}). '
                    'See the Risk Forecast chart for the full capacity curve.'
                )

    # ------------------------------------------------------------------
    # 7. Closing strategic suggestion
    # ------------------------------------------------------------------
    if stress_index >= 50:
        c = STATUS_COLOURS["elevated"]
        blocks.append(
            f'{_dot(c)}<b>Strategic Suggestion:</b> Increase daily available hours, shift '
            'low-impact tasks to later planning cycles, or renegotiate deadlines '
            'on the high-risk tasks identified above.'
        )
    else:
        c = STATUS_COLOURS["stable"]
        blocks.append(
            f'{_dot(c)}<b>Strategic Suggestion:</b> Maintain current pacing. '
            'Use remaining capacity buffer to front-load any tasks approaching '
            'moderate risk before they escalate.'
        )

    return "<br><br>".join(blocks)