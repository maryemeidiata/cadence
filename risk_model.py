import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def compute_task_risk(
    df: pd.DataFrame,
    available_hours_per_day: float,
    planning_horizon_days: int,
) -> pd.DataFrame:
    """
    Computes per-task risk metrics:

      failure_probability  – sigmoid of a risk score that combines:
                             • local overload (capacity vs. required hours
                               within the task's own deadline window)
                             • urgency (1 / days_left)
                             • dependency flag
                             • local competition pressure (how much of the
                               remaining capacity this task needs relative
                               to all other tasks competing within the same
                               deadline window — replaces the old global
                               system_pressure constant that penalised every
                               task equally regardless of deadline proximity)

      impact_weight        – normalised composite of strategic_importance
                             and business_impact  →  [0.1, 1.0]

      expected_impact      – failure_probability × impact_weight

      expected_loss_hours  – failure_probability × est_hours

    Key fix vs. previous version
    ----------------------------
    The old model applied a single global `system_over` constant to every
    task, meaning a task due in 10 days bore the same system-load penalty
    as one due tomorrow.  This version computes a *per-task* competition
    pressure: how much of the available capacity within *this task's*
    deadline window is already claimed by other tasks with equal or tighter
    deadlines.  Tasks with far-off deadlines get lower competition pressure
    than tasks fighting for the same near-term slots.
    """

    out = df.copy()

    required_cols = [
        "name", "days_left", "est_hours",
        "strategic_importance", "business_impact", "dependency_risk",
    ]
    for c in required_cols:
        if c not in out.columns:
            raise KeyError(f"Missing required column: {c}")

    out["days_left"] = out["days_left"].astype(int).clip(lower=1)
    out["est_hours"] = out["est_hours"].astype(float).clip(lower=0.0)

    # ------------------------------------------------------------------
    # Local capacity window
    # ------------------------------------------------------------------
    out["deadline_window_days"]     = out["days_left"].clip(upper=planning_horizon_days)
    out["capacity_before_deadline"] = out["deadline_window_days"] * float(available_hours_per_day)

    # Slack: positive = comfortable, negative = overloaded
    out["slack_hours"]      = out["capacity_before_deadline"] - out["est_hours"]
    out["overload_severity"] = (-out["slack_hours"]).clip(lower=0.0)

    # Normalise overload by daily capacity so large tasks don't explode the scale
    out["overload_norm"] = out["overload_severity"] / (float(available_hours_per_day) + 1e-9)

    # ------------------------------------------------------------------
    # Urgency
    # ------------------------------------------------------------------
    out["urgency"] = 1.0 / out["days_left"].astype(float)

    # ------------------------------------------------------------------
    # Per-task competition pressure  (replaces global system_pressure)
    #
    # For each task i, sum the est_hours of all tasks j whose deadline
    # is ≤ task i's deadline (they compete for the same capacity window).
    # Divide by the capacity available in that window.
    # A value > 1 means that window is oversubscribed.
    # ------------------------------------------------------------------
    def _competition_pressure(row):
        window_cap = row["capacity_before_deadline"]
        competing_hours = out.loc[
            out["days_left"] <= row["days_left"], "est_hours"
        ].sum()
        pressure = competing_hours / (window_cap + 1e-9)
        return max(0.0, pressure - 1.0)   # only penalise when oversubscribed

    out["competition_pressure"] = out.apply(_competition_pressure, axis=1)

    # ------------------------------------------------------------------
    # Dependency flag
    # ------------------------------------------------------------------
    out["dep"] = out["dependency_risk"].astype(bool).astype(int)

    # ------------------------------------------------------------------
    # Impact weight  [0.1 … 1.0]
    # ------------------------------------------------------------------
    out["impact_weight"] = (
        0.5 * out["strategic_importance"].astype(float)
        + 0.5 * out["business_impact"].astype(float)
    ) / 5.0
    out["impact_weight"] = out["impact_weight"].clip(lower=0.1)

    # ------------------------------------------------------------------
    # Risk score → failure probability
    #
    # Weights are interpretable:
    #   overload_norm         most important individual signal
    #   competition_pressure  contextual load signal (was global, now local)
    #   urgency               time pressure
    #   dep                   structural risk multiplier
    #   bias                  sets baseline so no-pressure tasks stay low-risk
    # ------------------------------------------------------------------
    w_overload     = 1.8
    w_competition  = 1.2
    w_urgency      = 1.1
    w_dep          = 0.7
    bias           = -1.2

    out["risk_score"] = (
        bias
        + w_overload    * out["overload_norm"]
        + w_competition * out["competition_pressure"]
        + w_urgency     * out["urgency"]
        + w_dep         * out["dep"]
    )

    out["failure_probability"] = out["risk_score"].apply(_sigmoid).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    out["expected_impact"]     = out["failure_probability"] * out["impact_weight"]
    out["expected_loss_hours"] = out["failure_probability"] * out["est_hours"]

    return out


def compute_weighted_stress(df_risk: pd.DataFrame) -> float:
    """
    Impact-weighted stress index  →  [0, 100].

    Uses a weighted average of failure_probability where the weights are
    impact_weight.  This means one critical task at 95% failure probability
    raises the stress index more than nine trivial tasks at 5%.

    Compare to the old unweighted mean which would report ~14 (stable) in
    that exact scenario — masking the real risk.
    """
    w = df_risk["impact_weight"]
    p = df_risk["failure_probability"]
    weighted = float((p * w).sum() / (w.sum() + 1e-9)) * 100.0
    return round(weighted, 1)


def forecast_system_risk(
    df: pd.DataFrame,
    available_hours_per_day: float,
    planning_horizon_days: int,
    capacity_scenarios: list[float] | None = None,
) -> pd.DataFrame:
    """
    Forecast system metrics under capacity delta scenarios.

    capacity_scenarios: list of hour-per-day deltas, e.g. [-2, -1, 0, 1, 2].

    Stress index is now impact-weighted (see compute_weighted_stress).
    The resulting dataframe can be used to tell the user exactly how much
    stress drops per additional hour/day of capacity — a concrete,
    actionable insight.
    """
    if capacity_scenarios is None:
        capacity_scenarios = [-2, -1, 0, 1, 2]

    rows = []
    for delta in capacity_scenarios:
        cap = max(1.0, float(available_hours_per_day) + float(delta))
        df_risk = compute_task_risk(df, cap, planning_horizon_days)

        stress        = compute_weighted_stress(df_risk)
        loss_hours    = round(float(df_risk["expected_loss_hours"].sum()), 1)
        high_risk_pct = round(float((df_risk["failure_probability"] >= 0.7).mean() * 100.0), 1)

        rows.append({
            "capacity_hours_per_day": cap,
            "capacity_delta":         delta,
            "stress_index":           stress,
            "expected_loss_hours":    loss_hours,
            "high_risk_tasks_%":      high_risk_pct,
        })

    return pd.DataFrame(rows).sort_values("capacity_hours_per_day").reset_index(drop=True)