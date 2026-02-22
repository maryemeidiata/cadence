import pandas as pd


def build_schedule(
    df_scored: pd.DataFrame,
    daily_capacity: float,
    horizon_days: int,
    mode: str = "Academic",
) -> pd.DataFrame:
    """
    Mode-aware scheduler.

    Academic mode    → Earliest Deadline First (EDF).
    Operational mode → Highest Value Density first (priority_score / est_hours).

    Stores two metadata objects in schedule_df.attrs:
      deadline_risks : list[dict]  — tasks not completed before their deadline
      allocation_log : list[dict]  — one entry per (task, day) allocation,
                                     used by ui_calendar to render the Gantt.
    """

    tasks = df_scored.copy()

    # ------------------------------------------------------------------
    # Sort order depends on mode
    # ------------------------------------------------------------------
    if mode == "Operational":
        tasks["_vd"] = tasks["priority_score"] / tasks["est_hours"].clip(lower=0.1)
        tasks = tasks.sort_values("_vd", ascending=False).reset_index(drop=True)
        tasks.drop(columns=["_vd"], inplace=True)
    else:
        # Academic / default: Earliest Deadline First
        tasks = tasks.sort_values(
            ["days_left", "est_hours"], ascending=[True, True]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Per-task state
    # ------------------------------------------------------------------
    remaining: dict[str, float] = {
        row["name"]: float(row["est_hours"]) for _, row in tasks.iterrows()
    }

    fp_lookup: dict[str, float] = {}
    if "failure_probability" in tasks.columns:
        fp_lookup = {
            row["name"]: float(row["failure_probability"])
            for _, row in tasks.iterrows()
        }

    schedule_rows: list[dict] = []
    allocation_log: list[dict] = []   # granular data consumed by Gantt

    # ------------------------------------------------------------------
    # Day loop
    # ------------------------------------------------------------------
    for day in range(1, horizon_days + 1):
        cap_left = float(daily_capacity)
        allocations: list[tuple[str, float]] = []

        for _, row in tasks.iterrows():
            task_name     = row["name"]
            task_deadline = int(row["days_left"])

            if remaining[task_name] <= 0:
                continue
            if day > task_deadline:
                continue

            take = min(cap_left, remaining[task_name])
            if take > 0:
                allocations.append((task_name, round(take, 2)))
                remaining[task_name] -= take
                cap_left -= take

                allocation_log.append({
                    "task":                task_name,
                    "day":                 day,
                    "hours":               round(take, 2),
                    "failure_probability": fp_lookup.get(task_name, 0.0),
                    "deadline_day":        task_deadline,
                })

            if cap_left <= 0:
                break

        used = round(daily_capacity - cap_left, 2)

        schedule_rows.append({
            "day":                day,
            "allocated_hours":    used,
            "remaining_capacity": round(cap_left, 2),
            "allocations":        ", ".join(f"{t} ({h}h)" for t, h in allocations),
            "overload_hours":     round(max(0.0, used - daily_capacity), 2),
        })

    # ------------------------------------------------------------------
    # Deadline risks
    # ------------------------------------------------------------------
    deadline_risks: list[dict] = []
    for _, row in tasks.iterrows():
        name = row["name"]
        if remaining[name] > 0:
            deadline_risks.append({
                "task":             name,
                "unfinished_hours": round(remaining[name], 2),
                "deadline_day":     int(row["days_left"]),
            })

    schedule_df = pd.DataFrame(schedule_rows)
    schedule_df.attrs["deadline_risks"] = deadline_risks
    schedule_df.attrs["allocation_log"] = allocation_log

    return schedule_df