"""Phase 1 feature-set decision and the notebook 05 feature audit (training data only).

The source rows are snapshots taken the day before arrival (Antonio et al., 2019b), so some
fields may hold values recorded after booking, or even after a cancellation. Each leakage
candidate is judged by a rule written down before any model comparison:
  (a) documented timing: the dataset paper places the value at or after the outcome;
  (b) outcome-tied pattern: a category with >= MIN_ROWS training rows cancels < 1% or > 99%;
  (c) calendar field: a value (a year) that cannot recur for future bookings.
Training CV then measures what each exclusion costs; it does not choose the set, because the
most leaky set would always score highest.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common import MODEL_COLOURS, ROOT, figure, load_json, log, save_figure, save_json, table
from src.data import LEAKAGE_CANDIDATES, OUTCOME_FIELDS, TARGET, load_partition
from src.inference import corrected_interval, holm
from src.metrics import cv_scores, fold_frame, summarize
from src.preprocess import build_pipeline, reference_models

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


def feature_set_experiment():
    """Training-CV cost of each exclusion, for a linear and a tree reference model."""
    selected = load_json("results/phase1_feature_set.json")["excluded"]
    every = list(LEAKAGE_CANDIDATES)
    sets = {
        "all candidates kept": [],
        "minus year": ["arrival_date_year"],
        "minus year, parking": ["arrival_date_year", "required_car_parking_spaces"],
        "minus year, parking, deposit": ["arrival_date_year", "required_car_parking_spaces", "deposit_type"],
        "selected set": selected,
        "selected minus requests": selected + ["total_of_special_requests"],
        "selected minus ADR": selected + ["adr"],
        "all six candidates removed": every,
    }
    summaries, fold_tables = [], []
    for model_name, model in reference_models().items():
        for label, excluded in sets.items():
            X, y = load_partition("train", excluded=excluded)
            scores = cv_scores(build_pipeline(model), X, y)
            summaries.append({"model": model_name, "feature_set": label, "excluded": "; ".join(excluded) or "none",
                              "n_predictors": X.shape[1], **summarize(scores)})
            fold_tables.append(fold_frame(scores, label).assign(model=model_name))
            log("Phase 1", "feature-set CV", f"{model_name}, {label}: AP {summaries[-1]['pr_auc_mean']:.3f}")
    summary = table(pd.DataFrame(summaries), "phase1_feature_set_cv.csv")
    folds_frame = table(pd.concat(fold_tables, ignore_index=True), "phase1_feature_set_folds.csv")
    comparisons = []
    for model_name, frame in folds_frame.groupby("model"):
        pivot = frame.pivot(index="fold", columns="configuration", values="pr_auc")
        for label in sets:
            if label != "selected set":
                comparisons.append({"model": model_name, "option": label, "reference": "selected set",
                                    **corrected_interval(pivot[label] - pivot["selected set"])})
    comparison = table(holm(pd.DataFrame(comparisons)), "phase1_feature_set_comparisons.csv")
    feature_set_figure(summary)
    return summary, comparison


FEATURE_SET_LABELS = {"selected set": "selected: minus year, parking, deposit, country",
                      "selected minus requests": "selected, minus special requests",
                      "selected minus ADR": "selected, minus ADR",
                      "all six candidates removed": "selected, minus requests and ADR"}


def feature_set_figure(summary):
    """Training-CV AP of each feature set for both reference models, top to bottom in the order tested."""
    order = list(dict.fromkeys(summary.feature_set))
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    for offset, (model_name, frame) in zip([-0.2, 0.2], summary.groupby("model")):
        frame = frame.set_index("feature_set").loc[order]
        positions = np.arange(len(order)) + offset
        axis.barh(positions, frame.pr_auc_mean, height=0.4, xerr=frame.pr_auc_std, label=model_name,
                  color=MODEL_COLOURS[model_name])
        for position, value, sd in zip(positions, frame.pr_auc_mean, frame.pr_auc_std):
            axis.text(value + sd + 0.01, position, f"{value:.3f}", va="center", fontsize=8)
    axis.set_yticks(np.arange(len(order)), [FEATURE_SET_LABELS.get(label, label) for label in order])
    axis.invert_yaxis()
    axis.set_xlim(0, 0.9)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    axis.set(title="What each exclusion costs on training CV",
             xlabel="Average precision (5-fold group CV, error bar = fold SD)", ylabel="Feature set")
    save_figure(fig, "fig_A_03_feature_set_cv.png")


def feature_audit():
    """Notebook 05: role of every field, plus tuned-model sensitivity to the exclusions."""
    from src.models import tuned_pipelines

    decisions = pd.read_csv(ROOT / "results/phase1_feature_decisions.csv").set_index("feature")
    dictionary = pd.read_csv(ROOT / "results/phase1_feature_dictionary.csv").set_index("feature")
    rows = []
    for feature, meaning in dictionary.meaning.items():
        if feature == TARGET:
            role, decision = "target", "predicted, never an input"
        elif feature in OUTCOME_FIELDS:
            role, decision = "outcome or after-booking record", "always excluded"
        elif feature in decisions.index:
            excluded = decisions.loc[feature, "excluded"]
            role = f"leakage candidate (rule {decisions.loc[feature, 'rule']})" if excluded else "snapshot field, kept"
            decision = "excluded" if excluded else "kept; sensitivity checked"
        else:
            role, decision = "booking-time field", "kept"
        evidence = decisions.loc[feature, "evidence"] if feature in decisions.index else ""
        rows.append({"feature": feature, "meaning": meaning, "role": role, "decision": decision, "evidence": evidence})
    for feature in ["total_nights", "party_size", "lead_days_per_night", "prior_cancellation_fraction"]:
        rows.append({"feature": feature, "meaning": "engineered inside the pipeline from kept fields",
                     "role": "engineered", "decision": "kept", "evidence": "row-local; uses no outcome"})
    audit = table(pd.DataFrame(rows), "d2_1_feature_audit.csv")

    selected = load_json("results/phase1_feature_set.json")["excluded"]
    model = tuned_pipelines()["Hist Gradient Boosting"]
    variants = {"final feature set": selected}
    for feature in selected:
        variants[f"add back {feature}"] = [item for item in selected if item != feature]
    variants["remove total_of_special_requests"] = selected + ["total_of_special_requests"]
    variants["remove adr"] = selected + ["adr"]
    fold_tables = []
    for label, excluded in variants.items():
        X, y = load_partition("train", excluded=excluded)
        fold_tables.append(fold_frame(cv_scores(model, X, y), label))
        log("D2-1", "sensitivity", f"tuned HGB, {label}: AP {fold_tables[-1].pr_auc.mean():.3f}")
    folds_frame = table(pd.concat(fold_tables, ignore_index=True), "d2_1_availability_folds.csv")
    summary = table(pd.DataFrame([{"configuration": label, **corrected_interval(group.pr_auc)}
                                  for label, group in folds_frame.groupby("configuration", sort=False)]),
                    "d2_1_availability_summary.csv")
    pivot = folds_frame.pivot(index="fold", columns="configuration", values="pr_auc")
    comparison = table(holm(pd.DataFrame([{"option": label, "reference": "final feature set",
                                           **corrected_interval(pivot[label] - pivot["final feature set"])}
                                          for label in variants if label != "final feature set"])),
                       "d2_1_availability_comparison.csv")
    plt.figure(figsize=(8, 4.5))
    plt.barh(summary.configuration, summary["mean"],
             xerr=[summary["mean"] - summary.ci_low, summary.ci_high - summary["mean"]], color="#E69F00")
    figure("fig_A_01_availability.png", "Tuned gradient boosting: AP with each excluded field added back",
           "Average precision (training CV, corrected 95% CI)", "Feature set")
    return audit, summary, comparison
