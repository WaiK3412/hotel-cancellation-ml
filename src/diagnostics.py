"""Bootstrap uncertainty and decision costs for fixed predictions (no model is refitted here)."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.common import SEED
from src.inference import holm


def map_probability(p, policy):
    """Apply a frozen calibration policy: a sigmoid on the log-odds, or no change."""
    p = np.asarray(p)
    if policy["calibration"] == "sigmoid":
        return expit(policy["slope"] * logit(np.clip(p, 1e-6, 1 - 1e-6)) + policy["intercept"])
    return p.copy()


def probability_metrics(y, p, threshold=0.5):
    """The seven classification metrics at a cutoff, plus the Brier score of the probabilities."""
    y = np.asarray(y)
    p = np.asarray(p)
    prediction = p > threshold
    return {"accuracy": accuracy_score(y, prediction), "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0), "f1": f1_score(y, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


def cost_loss(y, p, threshold, ratio):
    """Per-booking cost: a missed cancellation costs `ratio` units, a false alarm costs 1."""
    y = np.asarray(y)
    prediction = np.asarray(p) > threshold
    return ratio * ((y == 1) & ~prediction) + ((y == 0) & prediction)


def bootstrap_indices(y, repeats=500):
    """Row resamples that keep the number of each class fixed (stratified bootstrap)."""
    y = np.asarray(y)
    classes = [np.flatnonzero(y == label) for label in np.unique(y)]
    rng = np.random.default_rng(SEED)
    for _ in range(repeats):
        yield np.concatenate([rng.choice(rows, len(rows), replace=True) for rows in classes])


def bootstrap_summary(values, observed):
    """Observed value plus the bootstrap mean, SD and percentile 95% interval."""
    values = np.asarray(values)
    return {"observed": float(observed), "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975)), "n": len(values)}


def centred_p_value(samples, observed):
    """Approximate two-sided p-value: how often a centred resample is as extreme as the observed value."""
    samples = np.asarray(samples)
    return float((1 + np.sum(np.abs(samples - observed) >= abs(observed))) / (len(samples) + 1))


def paired_loss_comparison(y, loss_first, loss_second, repeats=500):
    """Bootstrap the mean difference of two per-row losses computed on the same rows."""
    difference = np.asarray(loss_first) - np.asarray(loss_second)
    observed = difference.mean()
    samples = np.array([difference[rows].mean() for rows in bootstrap_indices(y, repeats)])
    return {**bootstrap_summary(samples, observed), "p_value": centred_p_value(samples, observed)}


def prediction_intervals(y, p, metrics=("pr_auc", "f1"), repeats=300):
    """Bootstrap intervals for the chosen metrics of one set of predictions."""
    y = np.asarray(y)
    p = np.asarray(p)
    observed = probability_metrics(y, p)
    samples = {metric: [] for metric in metrics}
    for rows in bootstrap_indices(y, repeats):
        values = probability_metrics(y[rows], p[rows])
        for metric in metrics:
            samples[metric].append(values[metric])
    return [{"metric": metric, **bootstrap_summary(samples[metric], observed[metric])} for metric in metrics]


def paired_ap_comparisons(y, probabilities, reference, repeats=500):
    """AP difference of each model against a reference on the same resampled rows (Holm-adjusted)."""
    y = np.asarray(y)
    observed = {name: average_precision_score(y, p) for name, p in probabilities.items()}
    samples = {name: [] for name in probabilities if name != reference}
    for rows in bootstrap_indices(y, repeats):
        scores = {name: average_precision_score(y[rows], np.asarray(p)[rows]) for name, p in probabilities.items()}
        for name in samples:
            samples[name].append(scores[name] - scores[reference])
    results = []
    for name, values in samples.items():
        difference = observed[name] - observed[reference]
        results.append({"option": name, "reference": reference, "metric": "pr_auc",
                        **bootstrap_summary(values, difference), "p_value": centred_p_value(values, difference)})
    return holm(pd.DataFrame(results))
