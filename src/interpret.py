"""Notebook 10: ensemble diversity, uncertainty of the test results and partial dependence.

All models here are already fitted and frozen; nothing is re-chosen. Bootstrap intervals
resample rows (or whole arrival weeks) of fixed predictions, so they describe uncertainty in the
reported scores, not in the choices made earlier.
"""
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score

from src.common import ROOT, SEED, load_json, log, save_figure, save_json, table
from src.diagnostics import bootstrap_indices, bootstrap_summary, paired_ap_comparisons, prediction_intervals
from src.data import load_partition
from src.evaluate import slug
from src.inference import holm

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]


def diversity(first, second, oof, target, rows):
    """Agreement statistics for two models' out-of-fold predictions at the 0.5 cutoff."""
    p, q, ys = oof[first].to_numpy()[rows], oof[second].to_numpy()[rows], target[rows]
    right_first, right_second = (p > 0.5) == ys, (q > 0.5) == ys
    n11, n00 = np.sum(right_first & right_second), np.sum(~right_first & ~right_second)
    n10, n01 = np.sum(right_first & ~right_second), np.sum(~right_first & right_second)
    return {"q_statistic": (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01),
            "double_fault": np.mean(~right_first & ~right_second),
            "disagreement": np.mean(right_first != right_second), "probability_correlation": np.corrcoef(p, q)[0, 1]}


def diversity_analysis():
    oof = pd.read_csv(ROOT / "results/phase4_oof_predictions.csv")
    target = oof.is_canceled.to_numpy()
    names = list(load_json("results/phase3_best_params.json"))
    pairs = list(combinations(names, 2))
    selected = tuple(load_json("results/phase4_ensemble_selection.json")["pair"])
    observed = {pair: diversity(*pair, oof, target, np.arange(len(target))) for pair in pairs}
    samples = {pair: {metric: [] for metric in observed[pair]} for pair in pairs}
    for rows in bootstrap_indices(target, 600):
        for pair in pairs:
            for metric, value in diversity(*pair, oof, target, rows).items():
                samples[pair][metric].append(value)
    reference = next(pair for pair in pairs if set(pair) == set(selected))
    intervals, contrasts = [], []
    for pair in pairs:
        for metric, values in samples[pair].items():
            intervals.append({"first": pair[0], "second": pair[1], "selected_pair": pair == reference, "metric": metric,
                              **bootstrap_summary(values, observed[pair][metric])})
            if pair != reference and metric != "probability_correlation":
                delta = np.asarray(values) - np.asarray(samples[reference][metric])
                point = observed[pair][metric] - observed[reference][metric]
                p_value = (1 + np.sum(np.abs(delta - point) >= abs(point))) / (len(delta) + 1)
                contrasts.append({"option": " + ".join(pair), "reference": " + ".join(reference), "metric": metric,
                                  **bootstrap_summary(delta, point), "p_value": p_value})
    table(holm(pd.DataFrame(contrasts)), "d2_6_diversity_comparisons.csv")
    return table(pd.DataFrame(intervals), "d2_6_diversity_intervals.csv")


def holdout_uncertainty():
    test = pd.read_csv(ROOT / "results/phase4_test_predictions.csv")
    ys = test.is_canceled.to_numpy()
    names = list(load_json("results/phase3_best_params.json")) + ["Voting", "Stacking"]
    probabilities = {name: test["p_" + slug(name)].to_numpy() for name in names}
    rows = [{"model": name, **row} for name, p in probabilities.items()
            for row in prediction_intervals(ys, p, METRICS, repeats=500)]
    intervals = table(pd.DataFrame(rows), "d2_6_holdout_intervals.csv")
    recommended = load_json("results/phase4_ensemble_selection.json")["recommended"]
    comparison = table(paired_ap_comparisons(ys, probabilities, recommended, repeats=1000), "d2_6_holdout_comparisons.csv")
    shown = intervals[intervals.metric == "pr_auc"]
    plt.figure(figsize=(8, 4))
    plt.errorbar(shown.model, shown.observed, yerr=[shown.observed - shown.ci_low, shown.ci_high - shown.observed],
                 fmt="o", color="#0072B2")
    plt.xticks(rotation=10)
    plt.ylabel("Average precision")
    plt.title("Test-set average precision with bootstrap 95% intervals", fontweight="bold")
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures/fig_H_13_holdout_uncertainty.png", bbox_inches="tight")
    plt.close()
    return intervals, comparison


def week_bootstrap():
    """Later-period AP intervals that resample whole arrival weeks (bookings in a week are related)."""
    manifest = pd.read_csv(ROOT / "results/d2_5_temporal_manifest.csv", parse_dates=["arrival_date"])
    manifest = manifest.query("role == 'later_validation'").set_index("source_row")
    predictions = pd.read_csv(ROOT / "results/d2_5_temporal_predictions.csv")
    pivot = predictions.pivot(index="source_row", columns="model", values="probability").loc[manifest.index]
    ys = predictions.drop_duplicates("source_row").set_index("source_row").loc[manifest.index, "is_canceled"].to_numpy()
    week = manifest.arrival_date.dt.to_period("W").astype(str).to_numpy()
    blocks = [np.flatnonzero(week == value) for value in sorted(set(week))]
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in pivot}
    for _ in range(300):
        rows = np.concatenate([blocks[i] for i in rng.choice(len(blocks), len(blocks), replace=True)])
        for name in pivot:
            samples[name].append(average_precision_score(ys[rows], pivot[name].to_numpy()[rows]))
    return table(pd.DataFrame([{"model": name, "arrival_weeks": len(blocks),
                                **bootstrap_summary(values, average_precision_score(ys, pivot[name]))}
                               for name, values in samples.items()]), "d2_6_temporal_week_intervals.csv")


def partial_dependence():
    """Average predicted probability as one field is set to each grid value for 800 training rows."""
    X, _ = load_partition("train")
    model = joblib.load(ROOT / "models/recommended.joblib")
    anchors = X.sample(n=800, random_state=SEED)
    table(pd.DataFrame({"source_row": anchors.index}), "d2_6_pdp_anchors.csv")
    boot = np.random.default_rng(SEED).choice(len(anchors), (400, len(anchors)), replace=True)
    features = [feature for feature in ["lead_time", "adr", "total_of_special_requests"] if feature in X]
    rows = []
    for feature in features:
        grid = np.unique(np.round(X[feature].quantile(np.linspace(0.05, 0.95, 20)).to_numpy(), 2))
        for value in grid:
            changed = anchors.copy()
            changed[feature] = value
            p = model.predict_proba(changed)[:, 1]
            means = p[boot].mean(axis=1)
            rows.append({"feature": feature, "value": value, "mean": p.mean(), "ci_low": np.quantile(means, 0.025),
                         "ci_high": np.quantile(means, 0.975), "individual_q05": np.quantile(p, 0.05),
                         "individual_q95": np.quantile(p, 0.95)})
    curves = table(pd.DataFrame(rows), "d2_6_partial_dependence.csv")
    partial_dependence_figure(curves)
    return curves


def partial_dependence_figure(curves):
    """One panel per field; a single legend below the panels so it never covers the curves."""
    features = list(dict.fromkeys(curves.feature))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))  # 2 x 2 so text stays readable in print
    for axis, feature in zip(axes.flat, features):
        frame = curves[curves.feature == feature]
        axis.fill_between(frame.value, frame.individual_q05, frame.individual_q95, alpha=0.15,
                          label="middle 90% of individual bookings")
        axis.fill_between(frame.value, frame.ci_low, frame.ci_high, alpha=0.4, label="average, 95% interval")
        axis.plot(frame.value, frame["mean"], linewidth=2, label="average predicted risk")
        axis.set(title=feature, xlabel=feature, ylabel="Predicted cancellation probability", ylim=(0, 1))
        if len(frame) <= 5:  # a count such as special requests: mark only the values it can take
            axis.set_xticks(frame.value)
    legend_cell = axes.flat[-1]
    legend_cell.axis("off")
    legend_cell.legend(*axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=12)
    save_figure(fig, "fig_H_12_partial_dependence.png", "Partial dependence of the recommended model (training anchors)")


def population_check():
    """Test metrics when each row counts as many times as its identical source records."""
    X, y = load_partition("test")
    test = pd.read_csv(ROOT / "results/phase4_test_predictions.csv").set_index("source_row").loc[X.index]
    recommended = load_json("results/phase4_ensemble_selection.json")["recommended"]
    p, ys = test["p_" + slug(recommended)].to_numpy(), y.to_numpy()
    weights = pd.read_csv(ROOT / "results/phase0_duplicate_multiplicity.csv").set_index("source_row").loc[X.index,
                                                                                                         "source_multiplicity"].to_numpy()

    def metrics(labels, probability, weight):
        predicted = probability > 0.5
        return {"pr_auc": average_precision_score(labels, probability, sample_weight=weight),
                "accuracy": accuracy_score(labels, predicted, sample_weight=weight),
                "f1": f1_score(labels, predicted, sample_weight=weight, zero_division=0),
                "roc_auc": roc_auc_score(labels, probability, sample_weight=weight),
                "brier": brier_score_loss(labels, probability, sample_weight=weight)}

    rows = []
    for label, weight in [("unique bookings", np.ones(len(ys))), ("weighted by source duplicates", weights)]:
        observed = metrics(ys, p, weight)
        samples = {metric: [] for metric in observed}
        for rows_index in bootstrap_indices(ys, 300):
            for metric, value in metrics(ys[rows_index], p[rows_index], weight[rows_index]).items():
                samples[metric].append(value)
        for metric, values in samples.items():
            rows.append({"population": label, "rows": len(ys), "effective_rows": float(weight.sum()),
                         "positive_fraction": float(np.average(ys, weights=weight)), "metric": metric,
                         **bootstrap_summary(values, observed[metric])})
    return table(pd.DataFrame(rows), "d2_6_population_intervals.csv")


def ensemble_interpretation():
    save_json({"seed": SEED, "diversity_bootstraps": 600, "test_metric_bootstraps": 500, "test_comparison_bootstraps": 1000,
               "pdp_anchors": 800, "pdp_bootstraps": 400, "week_bootstraps": 300, "population_bootstraps": 300,
               "scope": "frozen models and predictions; intervals describe score uncertainty, not selection"},
              "results/d2_6_protocol.json")
    results = diversity_analysis(), holdout_uncertainty(), week_bootstrap(), partial_dependence(), population_check()
    log("D2-6", "interpretation", "Diversity, test uncertainty, week bootstrap, PDP and population checks saved.")
    return results
