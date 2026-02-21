import pandas as pd


# ---------------------------------------------------
# URGENCY FUNCTION
# ---------------------------------------------------
def compute_urgency_score(days_left: int) -> float:
    """
    Returns urgency in [0, 1].
    days_left=1  -> 1.0 (maximum urgency)
    days_left=10 -> 0.1
    days_left=30 -> 0.033
    Capped at 1.0 to stay on the same scale as strategic_importance / 5.
    """
    raw = 1.0 / max(days_left, 1)   # 1/days, max = 1.0 at day=1
    return min(raw, 1.0)


# ---------------------------------------------------
# PRIORITY ENGINE WITH MODE SWITCH
# ---------------------------------------------------
def compute_priority_scores(df: pd.DataFrame, mode: str = "Academic") -> pd.DataFrame:
    """
    Adds:
      - urgency_score        (0-1, normalized)
      - dependency_score     (0 or 1)
      - priority_score       (weighted composite, all inputs on comparable scales)

    All input features are normalized to [0, 1] before weighting so that
    the weight coefficients are actually meaningful.

    Mode options:
      Academic    – deadline-driven: urgency dominates
      Operational – value-driven: strategic importance + business impact dominate
    """

    out = df.copy()

    # --- Feature engineering -------------------------------------------------

    out["urgency_score"] = out["days_left"].apply(compute_urgency_score)
    out["dependency_score"] = out["dependency_risk"].astype(int).astype(float)

    # Normalize strategic_importance and business_impact from [1,5] → [0,1]
    out["strat_norm"]  = (out["strategic_importance"].astype(float) - 1.0) / 4.0
    out["impact_norm"] = (out["business_impact"].astype(float)       - 1.0) / 4.0

    # --- Mode-specific weights -----------------------------------------------

    if mode == "Academic":
        # Deadline-sensitive: urgency is primary driver
        # Strategic/impact weights are zero because Academic tasks
        # (assignments, exams) don't carry business value metadata.
        w_strat  = 0.10
        w_impact = 0.05
        w_urg    = 0.70
        w_dep    = 0.15

    elif mode == "Operational":
        # Value/stability-driven: impact and strategy dominate,
        # urgency is still present but not dominant.
        w_strat  = 0.35
        w_impact = 0.35
        w_urg    = 0.20
        w_dep    = 0.10

    else:
        # Fallback: balanced
        w_strat  = 0.25
        w_impact = 0.25
        w_urg    = 0.35
        w_dep    = 0.15

    # --- Priority score  (all inputs now on [0,1]) ----------------------------
    out["priority_score"] = (
        out["strat_norm"]        * w_strat  +
        out["impact_norm"]       * w_impact +
        out["urgency_score"]     * w_urg    +
        out["dependency_score"]  * w_dep
    )

    # Drop intermediate normalized columns to keep df clean
    out.drop(columns=["strat_norm", "impact_norm"], inplace=True)

    return out