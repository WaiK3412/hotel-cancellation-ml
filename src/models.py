"""Phase 3: naive baselines, three classifier families and their hyperparameter search.

Everything here uses the training partition and the shared group-aware folds; the test set is
never loaded. Tuning evidence has three layers:
1. default vs tuned on the same CV folds, with a corrected paired interval (this notebook);
2. a nested search, where outer folds never influence the choice (notebook 07);
3. default vs tuned on the untouched test set, once, at final evaluation (notebook 04).
"""
import json
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, validation_curve

from src.common import MODEL_COLOURS, ROOT, SEED, figure, load_json, log, save_figure, save_json, table
from src.data import load_partition
from src.inference import corrected_interval
from src.metrics import N_JOBS, SCORING, cv_scores, fold_frame, folds, summarize
from src.preprocess import build_pipeline, selected_options

# Random-search budget per family (candidates x 5 folds), sized to finish in about 30 minutes.
SEARCH_BUDGET = {"Logistic Regression": 20, "Random Forest": 25, "Hist Gradient Boosting": 40}
# One-parameter validation curves around the tuned configuration.
CURVES = {
    "Logistic Regression": [("model__C", list(np.logspace(-3, 2, 8)))],
    "Random Forest": [("model__min_samples_leaf", [1, 2, 4, 8, 16, 32])],
    "Hist Gradient Boosting": [("model__learning_rate", [0.02, 0.05, 0.1, 0.2, 0.3]),
                               ("model__max_leaf_nodes", [7, 15, 31, 63, 127])],
}


def baseline_models():
    """Naive reference points: always predict the majority class, or guess at the class rate."""
    return {"Majority baseline": DummyClassifier(strategy="most_frequent"),
            "Stratified baseline": DummyClassifier(strategy="stratified", random_state=SEED)}


def base_models():
    """Library defaults, apart from a seed, a convergence budget and single-threaded forests."""
    return {"Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
            "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=1),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED)}


def search_spaces():
    """Distributions sampled by the random search (parameter names address the pipeline's model step)."""
    return {
        "Logistic Regression": {
            "model__C": loguniform(1e-3, 1e2),
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"],
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 12, 20, 30],
            "model__min_samples_leaf": [1, 2, 4, 8, 16],
            "model__max_features": ["sqrt", 0.1, 0.2],
            "model__class_weight": [None, "balanced_subsample"],
        },
        "Hist Gradient Boosting": {
            "model__learning_rate": loguniform(0.02, 0.3),
            "model__max_iter": [100, 200, 400],
            "model__max_leaf_nodes": [15, 31, 63, 127],
            "model__min_samples_leaf": [10, 20, 50, 100],
            "model__l2_regularization": loguniform(1e-3, 10),
            "model__max_features": [0.5, 0.75, 1.0],
        },
    }


def default_pipelines():
    return {name: build_pipeline(model, **selected_options()) for name, model in base_models().items()}


def tuned_pipelines():
    """The three families with the parameters chosen in Phase 3 (results/phase3_best_params.json)."""
    best = load_json("results/phase3_best_params.json")
    return {name: pipeline.set_params(**best[name]) for name, pipeline in default_pipelines().items()}


def plain(value):
    """JSON-friendly version of a sampled parameter value."""
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def search_frame(search, name):
    """Every random-search candidate with its parameters and CV metrics."""
    results = search.cv_results_
    rows = []
    for index, params in enumerate(results["params"]):
        row = {"model": name, "candidate": index, "params_json": json.dumps({k: plain(v) for k, v in params.items()}),
               "fit_seconds": results["mean_fit_time"][index]}
        for metric in SCORING:
            row[metric + "_mean"] = results["mean_test_" + metric][index]
            row[metric + "_std"] = results["std_test_" + metric][index]
            row["train_" + metric + "_mean"] = results["mean_train_" + metric][index]
        rows.append(row)
    return pd.DataFrame(rows)


def run_baselines(X, y):
    """Five-fold CV of the two naive baselines."""
    rows = []
    for name, model in baseline_models().items():
        rows.append({"model": name, **summarize(cv_scores(build_pipeline(model, **selected_options()), X, y))})
    return table(pd.DataFrame(rows), "phase3_naive_baselines.csv")


def run_models():
    """Defaults, random search, paired default-vs-tuned comparison and validation curves."""
    X, y = load_partition("train")
    save_json({"primary_metric": "average_precision (sklearn, non-interpolated)", "positive_class": "is_canceled = 1",
               "decision_threshold": 0.5, "training_positive_fraction": float(y.mean()), "train_rows": len(X),
               "folds": "5 group-aware stratified folds", "search_budget": SEARCH_BUDGET, "seed": SEED,
               "selection_rule": "tuned = best search candidate by CV AP, unless the default scores higher"},
              "results/phase3_metrics_policy.json")
    baselines = run_baselines(X, y)
    log("Phase 3", "naive baselines", f"Majority AP {baselines.pr_auc_mean.iloc[0]:.3f} (= class rate).")

    defaults, default_folds, tuning, trials, tuned_folds, best_params, spaces = [], [], [], [], [], {}, {}
    for name, pipeline in default_pipelines().items():
        default_scores = cv_scores(pipeline, X, y)
        defaults.append({"model": name, **summarize(default_scores)})
        default_folds.append(fold_frame(default_scores, name))
        log("Phase 3", "default", f"{name}: CV AP {defaults[-1]['pr_auc_mean']:.3f}")

        start = perf_counter()
        search = RandomizedSearchCV(pipeline, search_spaces()[name], n_iter=SEARCH_BUDGET[name], scoring=SCORING,
                                    refit=False, cv=folds(), n_jobs=N_JOBS, random_state=SEED,
                                    return_train_score=True, error_score="raise")
        search.fit(X, y)
        seconds = perf_counter() - start
        frame = search_frame(search, name)
        trials.append(frame)
        best = int(frame.pr_auc_mean.idxmax())
        search_wins = frame.pr_auc_mean[best] > defaults[-1]["pr_auc_mean"]
        chosen = {k: plain(v) for k, v in search.cv_results_["params"][best].items()} if search_wins else {}
        best_params[name] = chosen
        tuned = frame.iloc[best] if search_wins else None
        tuned_fold_ap = (np.array([search.cv_results_[f"split{i}_test_pr_auc"][best] for i in range(5)])
                         if search_wins else default_scores["test_pr_auc"])
        tuned_folds.append(pd.DataFrame({"configuration": name, "fold": np.arange(1, 6), "pr_auc": tuned_fold_ap}))
        gain = corrected_interval(tuned_fold_ap - default_scores["test_pr_auc"])
        row = {"model": name, "default_pr_auc": defaults[-1]["pr_auc_mean"],
               "best_search_pr_auc": frame.pr_auc_mean[best], "tuned_pr_auc": tuned_fold_ap.mean(),
               "improvement_pr_auc": gain["mean"], "improvement_ci_low": gain["ci_low"],
               "improvement_ci_high": gain["ci_high"], "improvement_p_value": gain["p_value"],
               "candidates": SEARCH_BUDGET[name], "candidates_beating_default": int((frame.pr_auc_mean > defaults[-1]["pr_auc_mean"]).sum()),
               "search_seconds": seconds, "chosen": "search winner" if search_wins else "default kept",
               "best_params_json": json.dumps(chosen)}
        source = tuned if search_wins else pd.Series({k: v for k, v in defaults[-1].items()})
        for metric in SCORING:
            row[metric + "_mean"] = source[metric + "_mean"]
            row[metric + "_std"] = source[metric + "_std"]
            row["train_" + metric + "_mean"] = source["train_" + metric + "_mean"]
        row["fit_seconds"] = source["fit_seconds"]
        tuning.append(row)
        spaces[name] = {key: (str(value.__class__.__name__) + str(value.args) if hasattr(value, "args") else value)
                        for key, value in search_spaces()[name].items()}
        log("Phase 3", "search", f"{name}: default {row['default_pr_auc']:.3f} -> tuned {row['tuned_pr_auc']:.3f} "
            f"(gain {gain['mean']:+.4f}, 95% CI {gain['ci_low']:+.4f} to {gain['ci_high']:+.4f}); {seconds:.0f} s")
        save_json(best_params, "results/phase3_best_params.json")

    table(pd.DataFrame(defaults), "phase3_baseline.csv")
    table(pd.concat(default_folds, ignore_index=True), "phase3_baseline_folds.csv")
    tuning = table(pd.DataFrame(tuning), "phase3_tuning.csv")
    trials = table(pd.concat(trials, ignore_index=True), "phase3_search_trials.csv")
    table(pd.concat(tuned_folds, ignore_index=True), "phase3_tuned_folds.csv")
    save_json(spaces, "results/phase3_search_spaces.json")
    search_figure(trials, tuning)
    curves = validation_curves(X, y)
    return baselines, tuning, curves


def search_figure(trials, tuning):
    """Every candidate's CV AP, with the default and the chosen configuration marked."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))  # 2 x 2 so text stays readable in print
    axes.flat[-1].axis("off")
    for axis, (name, frame) in zip(axes.flat, trials.groupby("model", sort=False)):
        row = tuning.set_index("model").loc[name]
        jitter = np.random.default_rng(SEED).uniform(-0.15, 0.15, len(frame))
        axis.scatter(jitter, frame.pr_auc_mean, s=18, alpha=0.6, color=MODEL_COLOURS[name])
        axis.axhline(row.default_pr_auc, color="black", linestyle="--")
        axis.scatter([0], [row.tuned_pr_auc], marker="*", s=220, color="#D55E00", zorder=3)
        axis.set(title=f"{name}\ndefault {row.default_pr_auc:.3f}, chosen {row.tuned_pr_auc:.3f}", xticks=[],
                 xlabel="candidates (spread sideways for visibility)", ylabel="CV average precision")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color="#777777", label="search candidate"),
               plt.Line2D([], [], color="black", linestyle="--", label="library default"),
               plt.Line2D([], [], marker="*", linestyle="", markersize=12, color="#D55E00", label="chosen setting")]
    axes.flat[-1].legend(handles=handles, loc="center", fontsize=12)
    save_figure(fig, "fig_F_00_search_summary.png", "Random search: every candidate versus the default "
                "(training CV; each panel has its own scale)")


def validation_curves(X, y):
    """Training vs validation AP as one parameter varies around the tuned configuration."""
    rows = []
    for name, pipeline in tuned_pipelines().items():
        for parameter, values in CURVES[name]:
            train, valid = validation_curve(pipeline, X, y, param_name=parameter, param_range=values, cv=folds(),
                                            scoring="average_precision", n_jobs=N_JOBS)
            for value, train_scores, valid_scores in zip(values, train, valid):
                rows.append({"model": name, "parameter": parameter.replace("model__", ""), "value": float(value),
                             "train_pr_auc": train_scores.mean(), "cv_pr_auc": valid_scores.mean(),
                             "cv_pr_auc_std": valid_scores.std()})
            log("Phase 3", "validation curve", f"{name}: {parameter}")
    curves = table(pd.DataFrame(rows), "phase3_validation_curves.csv")
    validation_curves_figure(curves, load_json("results/phase3_best_params.json"))
    return curves


def validation_curves_figure(curves, best):
    """Training and validation AP as one parameter varies (2 x 2 panels, tuned value marked)."""
    panels = list(curves.groupby(["model", "parameter"], sort=False))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for axis, ((name, parameter), frame) in zip(axes.flat, panels):
        axis.plot(frame.value, frame.train_pr_auc, marker="o", color="#999999", label="training folds")
        axis.errorbar(frame.value, frame.cv_pr_auc, yerr=frame.cv_pr_auc_std, marker="o", capsize=3,
                      color=MODEL_COLOURS[name], label="validation folds")
        chosen = best[name].get("model__" + parameter)
        if chosen is not None:
            axis.axvline(chosen, color="#D55E00", linestyle=":", label=f"tuned value ({chosen:.3g})")
        if parameter in {"C", "learning_rate"}:
            axis.set_xscale("log")
        if parameter != "C":  # label every value tried so the edge of the searched range is visible
            axis.set_xticks(frame.value, [f"{value:g}" for value in frame.value])
            axis.minorticks_off()
        axis.set(title=f"{name}: {parameter}", xlabel=parameter, ylabel="Average precision")
        axis.legend(fontsize=8)
    save_figure(fig, "fig_F_01_validation_curves.png", "Validation curves around the tuned configurations")
