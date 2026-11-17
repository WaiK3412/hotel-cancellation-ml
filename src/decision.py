"""Notebook 09: probability calibration, a cost-based decision policy, and transfer checks.

1. Calibration and the cost policy are fitted on *training* out-of-fold predictions, split into
   three disjoint roles (fit the calibration map / choose the cutoff / check the policy). The
   policy is frozen and saved before the test predictions are read.
2. The temporal check trains on earlier arrivals and validates on bookings made later, inside the
   training partition. It is the closest available stand-in for "next season".
3. Hotel transfer trains on one hotel and validates on the other.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.common import ROOT, SEED, figure, load_json, log, save_json, table
from src.data import load_audit_columns, load_partition
from src.diagnostics import (cost_loss, map_probability, paired_ap_comparisons, paired_loss_comparison,
                             prediction_intervals, probability_metrics)
from src.evaluate import ensembles, slug
from src.inference import holm
from src.models import default_pipelines, tuned_pipelines

COST_RATIOS = [1, 2, 5, 10]
CUTOFF = pd.Timestamp("2017-01-01")


def policy_model():
    """The model whose probabilities the policy maps: the recommendation (Stacking falls back to Voting)."""
    recommended = load_json("results/phase4_ensemble_selection.json")["recommended"]
    return "Voting" if recommended == "Stacking" else recommended


def training_probabilities(name):
    """Out-of-fold training probabilities for a base model or for soft voting of the pair."""
    oof = pd.read_csv(ROOT / "results/phase4_oof_predictions.csv").set_index("source_row")
    if name == "Voting":
        pair = load_json("results/phase4_ensemble_selection.json")["pair"]
        return oof[pair].mean(axis=1), oof.is_canceled
    return oof[name], oof.is_canceled


def calibration_and_policy():
    name = policy_model()
    raw, target = training_probabilities(name)
    raw, target = raw.to_numpy(), target.to_numpy()
    positions = np.arange(len(target))
    calibration, rest = train_test_split(positions, test_size=0.5, stratify=target, random_state=SEED)
    selection, validation = train_test_split(rest, test_size=0.5, stratify=target[rest], random_state=SEED)
    table(pd.DataFrame({"position": positions, "role": np.select(
        [np.isin(positions, calibration), np.isin(positions, selection)], ["calibration", "threshold_selection"],
        "policy_validation")}), "d2_5_policy_roles.csv")

    # Raw vs sigmoid mapping is chosen by 3-fold CV inside the calibration role only.
    z = logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    cv_rows = []
    for fold, (train, valid) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(z[calibration], target[calibration]), 1):
        train, valid = calibration[train], calibration[valid]
        mapper = LogisticRegression(max_iter=2000).fit(z[train], target[train])
        cv_rows.append({"option": "raw", "fold": fold, "brier": brier_score_loss(target[valid], raw[valid])})
        cv_rows.append({"option": "sigmoid", "fold": fold,
                        "brier": brier_score_loss(target[valid], mapper.predict_proba(z[valid])[:, 1])})
    method = table(pd.DataFrame(cv_rows), "d2_5_calibration_inner_cv.csv").groupby("option").brier.mean().idxmin()
    mapper = LogisticRegression(max_iter=2000).fit(z[calibration], target[calibration])
    policy = {"model": name, "calibration": method, "slope": float(mapper.coef_[0, 0]),
              "intercept": float(mapper.intercept_[0]), "cost_ratios": COST_RATIOS,
              "cost_units": "hypothetical review-loss units (a missed cancellation costs `ratio` false alarms)",
              "rows": {"calibration": len(calibration), "threshold_selection": len(selection),
                       "policy_validation": len(validation)}}
    calibrated = map_probability(raw, policy)
    grid, thresholds = [], {}
    for ratio in COST_RATIOS:
        for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2):
            grid.append({"ratio": ratio, "threshold": threshold,
                         "selection_cost": cost_loss(target[selection], calibrated[selection], threshold, ratio).mean()})
        candidates = pd.DataFrame(grid).query("ratio == @ratio").sort_values(["selection_cost", "threshold"],
                                                                             ascending=[True, False])
        thresholds[str(ratio)] = float(candidates.threshold.iloc[0])
    policy["thresholds"] = thresholds
    policy["default_cost_ratio"] = 5
    table(pd.DataFrame(grid), "d2_5_threshold_search.csv")
    save_json(policy, "results/d2_5_policy.json")  # frozen before any test prediction is read
    log("D2-5", "policy frozen", f"{name}: {method} calibration; 5:1 cutoff {thresholds['5']:.2f}")

    test = pd.read_csv(ROOT / "results/phase4_test_predictions.csv")
    scopes = {"training_policy_validation": (target[validation], raw[validation]),
              "test_set": (test.is_canceled.to_numpy(), test["p_" + slug(name)].to_numpy())}
    policy_rows, calibration_rows, comparisons = [], [], []
    for scope, (ys, p) in scopes.items():
        mapped = map_probability(p, policy)
        for option, probability in [("raw", p), ("calibrated", mapped)]:
            for metric in ["brier", "pr_auc"]:
                calibration_rows.append({"scope": scope, "option": option,
                                         **prediction_intervals(ys, probability, [metric], repeats=300)[0]})
        comparisons.append({"scope": scope, "family": "calibration", "option": "calibrated", "reference": "raw",
                            "metric": "brier", **paired_loss_comparison(ys, (mapped - ys) ** 2, (p - ys) ** 2)})
        for ratio in COST_RATIOS:
            for option, probability, threshold in [("raw at 0.5", p, 0.5), ("calibrated at 0.5", mapped, 0.5),
                                                   ("chosen cutoff", mapped, thresholds[str(ratio)])]:
                loss = cost_loss(ys, probability, threshold, ratio)
                policy_rows.append({"scope": scope, "ratio": ratio, "option": option, "threshold": threshold,
                                    "rows": len(ys), "cost_per_1000": 1000 * loss.mean(),
                                    **probability_metrics(ys, probability, threshold)})
                if option != "raw at 0.5":
                    comparisons.append({"scope": scope, "family": "cost", "ratio": ratio, "option": option,
                                        "reference": "raw at 0.5", "metric": "cost_per_booking",
                                        **paired_loss_comparison(ys, loss, cost_loss(ys, p, 0.5, ratio))})
        plt.figure(figsize=(5.5, 4.5))
        plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        for option, probability in [("raw", p), ("calibrated", mapped)]:
            fraction, mean = calibration_curve(ys, probability, n_bins=10, strategy="quantile")
            plt.plot(mean, fraction, "o-", label=option)
        plt.legend()
        figure(f"fig_H_10_calibration_{scope}.png", f"Reliability of {name} probabilities ({scope.replace('_', ' ')})",
               "Mean predicted probability", "Observed cancellation rate")
    policy_scores = table(pd.DataFrame(policy_rows), "d2_5_policy_metrics.csv")
    table(pd.DataFrame(calibration_rows), "d2_5_calibration_metrics.csv")
    table(pd.concat([holm(group) for _, group in pd.DataFrame(comparisons).groupby(["scope", "family"])],
                    ignore_index=True), "d2_5_policy_comparisons.csv")
    shown = policy_scores.query("scope == 'test_set' and option in ['raw at 0.5', 'chosen cutoff']")
    plt.figure(figsize=(7, 4.5))
    for option, group in shown.groupby("option"):
        plt.plot(group.ratio, group.cost_per_1000, "o-", label=option)
    plt.legend()
    figure("fig_H_11_cost_sensitivity.png", "Cost per 1,000 bookings under hypothetical cost ratios (test set)",
           "Cost ratio (missed cancellation : false alarm)", "Cost units per 1,000 bookings")
    return policy_scores


def temporal_check():
    """Train on arrivals before 2017; validate on bookings made from 2017 on (training partition)."""
    X, y = load_partition("train")
    year = load_audit_columns("train", ["arrival_date_year"]).loc[X.index, "arrival_date_year"]
    month = pd.to_datetime(X.arrival_date_month, format="%B").dt.month
    arrival = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": X.arrival_date_day_of_month}))
    booked = arrival - pd.to_timedelta(X.lead_time, unit="D")
    earlier = arrival < CUTOFF
    later = (arrival >= CUTOFF) & (booked >= CUTOFF)
    # Bookings made before the cutoff for later arrivals overlap both periods: purge them.
    purged = (arrival >= CUTOFF) & (booked < CUTOFF)
    table(pd.DataFrame({"source_row": X.index, "arrival_date": arrival.to_numpy(), "booking_date": booked.to_numpy(),
                        "role": np.select([earlier, later, purged], ["earlier_train", "later_validation", "purged"],
                                          default="unassigned")}),
          "d2_5_temporal_manifest.csv")
    save_json({"cutoff": str(CUTOFF.date()), "earlier_train": int(earlier.sum()), "later_validation": int(later.sum()),
               "purged": int(purged.sum()), "booking_date": "arrival date minus lead time",
               "note": "tuned parameters were chosen with CV over all training rows, including the later period"},
              "results/d2_5_temporal_protocol.json")
    tuned = tuned_pipelines()
    pair = load_json("results/phase4_ensemble_selection.json")["pair"]
    voting, _ = ensembles(pair, tuned)
    candidates = {f"{name} (default)": pipeline for name, pipeline in default_pipelines().items()}
    candidates.update({f"{name} (tuned)": pipeline for name, pipeline in tuned.items()})
    candidates["Voting (tuned)"] = voting
    rows, predictions = [], {}
    for name, pipeline in candidates.items():
        p = clone(pipeline).fit(X.loc[earlier], y.loc[earlier]).predict_proba(X.loc[later])[:, 1]
        predictions[name] = p
        rows.append({"model": name, "training_rows": int(earlier.sum()), "validation_rows": int(later.sum()),
                     **probability_metrics(y.loc[later], p)})
        log("D2-5", "temporal", f"{name}: later-period AP {rows[-1]['pr_auc']:.3f}")
    temporal = table(pd.DataFrame(rows), "d2_5_temporal_metrics.csv")
    table(pd.DataFrame([{"model": name, **row} for name, p in predictions.items()
                        for row in prediction_intervals(y.loc[later], p, repeats=300)]), "d2_5_temporal_intervals.csv")
    table(pd.DataFrame([{"model": name, "source_row": row, "is_canceled": label, "probability": prob}
                        for name, p in predictions.items()
                        for row, label, prob in zip(X.index[later], y.loc[later], p)]), "d2_5_temporal_predictions.csv")
    table(paired_ap_comparisons(y.loc[later].to_numpy(), predictions, "Voting (tuned)"), "d2_5_temporal_comparisons.csv")
    drift = []
    for feature in ["lead_time", "adr", "adults", "total_of_special_requests", "previous_cancellations"]:
        before, after = X.loc[earlier, feature].dropna(), X.loc[later, feature].dropna()
        test = ks_2samp(before, after)
        drift.append({"feature": feature, "earlier_median": before.median(), "later_median": after.median(),
                      "ks_statistic": test.statistic, "p_value": test.pvalue})
    table(holm(pd.DataFrame(drift)), "d2_5_temporal_drift.csv")
    random_split = pd.read_csv(ROOT / "results/phase4_test_metrics.csv").set_index("model").pr_auc
    shown = [name for name in candidates if name.endswith("(tuned)")]
    x = np.arange(len(shown))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, [random_split[name.replace(" (tuned)", "")] for name in shown], 0.4,
            label="random test split", color="#999999")
    plt.bar(x + 0.2, temporal.set_index("model").loc[shown, "pr_auc"], 0.4, label="later period", color="#D55E00")
    plt.xticks(x, [name.replace(" (tuned)", "") for name in shown])
    plt.legend()
    figure("fig_H_14_temporal.png", "Average precision: random test split vs later booking period",
           "Model (tuned)", "Average precision")
    return temporal


def hotel_transfer():
    """Train on one hotel, validate on the other (tuned recommended model)."""
    X, y = load_partition("train")
    name = policy_model()
    tuned = tuned_pipelines()
    model = ensembles(load_json("results/phase4_ensemble_selection.json")["pair"], tuned)[0] if name == "Voting" else tuned[name]
    rows = []
    for training_hotel in sorted(X.hotel.unique()):
        train, valid = X.hotel == training_hotel, X.hotel != training_hotel
        p = clone(model).fit(X.loc[train], y.loc[train]).predict_proba(X.loc[valid])[:, 1]
        rows.append({"model": name, "training_hotel": training_hotel, "validation_hotel": X.loc[valid, "hotel"].iloc[0],
                     "training_rows": int(train.sum()), "validation_rows": int(valid.sum()),
                     **probability_metrics(y.loc[valid], p)})
    return table(pd.DataFrame(rows), "d2_5_hotel_transfer.csv")


def decision_validation():
    return calibration_and_policy(), temporal_check(), hotel_transfer()
