"""Phase 1 feature-set decision (training data only).

The source rows are snapshots taken the day before arrival (Antonio et al., 2019b), so some
fields may hold values recorded after booking, or even after a cancellation. Each leakage
candidate is judged by a rule written down before any model comparison:
  (a) documented timing: the dataset paper places the value at or after the outcome;
  (b) outcome-tied pattern: a category with >= MIN_ROWS training rows cancels < 1% or > 99%;
  (c) calendar field: a value (a year) that cannot recur for future bookings.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import log, save_figure, save_json, table
from src.data import LEAKAGE_CANDIDATES, OUTCOME_FIELDS, load_partition

MIN_ROWS = 500
DOCUMENTED_TIMING = {
    "country": "Antonio et al. (2019b, p. 45): hotels often do not know a guest's nationality until check-in.",
    "deposit_type": "Antonio et al. (2019b, Table 1): calculated from payments found before arrival or cancellation.",
}
CALENDAR_FIELDS = {"arrival_date_year": "Each year occurs once; a future booking always has an unseen year."}


def signal_groups(X):
    """Readable groupings of each candidate for the cancellation-rate table."""
    top_countries = X.country.value_counts().index[:5]
    return {
        "required_car_parking_spaces": np.where(X.required_car_parking_spaces > 0, "one or more", "none"),
        "deposit_type": X.deposit_type.fillna("Missing"),
        "country": X.country.where(X.country.isin(top_countries), "Other").fillna("Missing"),
        "total_of_special_requests": X.total_of_special_requests.clip(upper=3).map(
            lambda v: "3 or more" if v == 3 else str(int(v))),
        "adr": pd.qcut(X.adr, 4, labels=["lowest quarter", "second quarter", "third quarter", "highest quarter"]).astype(str),
        "arrival_date_year": X.arrival_date_year.astype(str),
    }


def leakage_signals():
    """Cancellation rate within each group of every candidate field (training rows only)."""
    X, y = load_partition("train", excluded=[])
    rows = []
    for feature, groups in signal_groups(X).items():
        frame = pd.DataFrame({"group": groups, "cancelled": y.to_numpy()})
        for group, values in frame.groupby("group").cancelled:
            rows.append({"feature": feature, "group": group, "rows": len(values), "cancel_rate": values.mean()})
    signals = table(pd.DataFrame(rows), "phase1_leakage_signals.csv")
    leakage_figure(signals, y.mean())
    return signals


SIGNAL_ORDER = {"required_car_parking_spaces": ["none", "one or more"],
                "total_of_special_requests": ["0", "1", "2", "3 or more"],
                "adr": ["lowest quarter", "second quarter", "third quarter", "highest quarter"]}


def leakage_figure(signals, overall):
    """Cancellation rate per group of each candidate, labelled with the rate (and size, for extreme groups)."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, (feature, frame) in zip(axes.flat, signals.groupby("feature", sort=False)):
        frame = frame.set_index("group")
        frame = frame.reindex(SIGNAL_ORDER.get(feature, sorted(frame.index)))
        axis.barh(frame.index.astype(str), frame.cancel_rate, color="#0072B2")
        axis.invert_yaxis()
        for position, (rate, rows) in enumerate(zip(frame.cancel_rate, frame.rows)):
            extreme = rows >= MIN_ROWS and (rate < 0.01 or rate > 0.99)
            label = f"{rate:.3f} (n = {int(rows):,})" if extreme else f"{rate:.3f}"
            axis.text(rate + 0.02, position, label, va="center", fontsize=8, zorder=4,
                      fontweight="bold" if extreme else "normal",
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        axis.axvline(overall, color="#D55E00", linestyle="--", label=f"overall {overall:.3f}")
        axis.set(title=feature, xlabel="Cancellation rate (training)", xlim=(0, 1.15))
        axis.legend(loc="lower right")
    save_figure(fig, "fig_A_02_leakage_signals.png", "Cancellation rate by candidate field (training rows)")


def decide_feature_set(signals):
    """Apply rules (a)-(c) to every candidate and save the decision with its evidence."""
    decisions = []
    for feature, reason in LEAKAGE_CANDIDATES.items():
        groups = signals[(signals.feature == feature) & (signals.rows >= MIN_ROWS)]
        extreme = groups[(groups.cancel_rate < 0.01) | (groups.cancel_rate > 0.99)]
        if feature in DOCUMENTED_TIMING:
            rule, evidence = "a", DOCUMENTED_TIMING[feature]
        elif len(extreme):
            group = extreme.iloc[0]
            rule = "b"
            evidence = (f"'{group.group}' ({int(group.rows):,} training rows) cancels at "
                        f"{group.cancel_rate:.3f}, so the field is tied to the outcome.")
        elif feature in CALENDAR_FIELDS:
            rule, evidence = "c", CALENDAR_FIELDS[feature]
        else:
            rule = ""
            evidence = "No documented timing and no outcome-tied group; kept, with a sensitivity check."
        decisions.append({"feature": feature, "concern": reason, "rule": rule, "excluded": bool(rule), "evidence": evidence})
    frame = table(pd.DataFrame(decisions), "phase1_feature_decisions.csv")
    excluded = frame.loc[frame.excluded, "feature"].tolist()
    save_json({"excluded": excluded, "kept_candidates": frame.loc[~frame.excluded, "feature"].tolist(),
               "outcome_fields_always_excluded": OUTCOME_FIELDS,
               "rules": {"a": "documented timing at or after the outcome",
                         "b": f"a group with >= {MIN_ROWS} training rows cancels < 1% or > 99%",
                         "c": "calendar value that cannot recur for future bookings"},
               "decided_on": "training rows only, before any model comparison"},
              "results/phase1_feature_set.json")
    log("Phase 1", "feature set", f"Excluded {excluded}; kept candidates {frame.loc[~frame.excluded, 'feature'].tolist()}.")
    return frame
