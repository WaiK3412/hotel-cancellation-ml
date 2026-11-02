"""Notebook 07: nested search and learning curves (training data only).

Outer folds estimate how well the *tuning procedure* works: inside each outer training part a
fresh search (sampled candidates plus the library default) picks a configuration using inner
folds only, and the outer validation rows then score it. Outer rows never influence the choice,
so the default-vs-nested gap here is not inflated by selection.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, ParameterSampler, train_test_split

from src.common import MODEL_COLOURS, SEED, log, save_figure, save_json, table
from src.data import load_partition
from src.inference import corrected_interval, holm
from src.metrics import N_JOBS, GroupedStratifiedKFold
from src.models import default_pipelines, plain, search_spaces, tuned_pipelines

NESTED_BUDGET = 8
FRACTIONS = [0.25, 0.5, 1.0]


def candidate_grid(name, pipeline):
    """Sampled candidates plus the default point, each as a one-point grid."""
    space = search_spaces()[name]
    sampled = list(ParameterSampler(space, n_iter=NESTED_BUDGET, random_state=SEED))
    default = {key: pipeline.get_params()[key] for key in space}
    return [{key: [plain(value)] for key, value in point.items()} for point in sampled + [default]]


def nested_search():
    X, y = load_partition("train")
    outer = list(GroupedStratifiedKFold(n_splits=3, random_state=SEED).split(X, y))
    save_json({"outer_folds": 3, "inner_folds": 3, "sampled_candidates": NESTED_BUDGET, "default_included": True,
               "spaces": "same as Phase 3 (src/models.py)", "learning_fractions": FRACTIONS, "seed": SEED,
               "scope": "estimates the tuning procedure; architecture and spaces were set in Phases 2-3"},
              "results/d2_3_protocol.json")
    rows, trials = [], []
    for name, pipeline in default_pipelines().items():
        grid = candidate_grid(name, pipeline)
        for fold, (train, valid) in enumerate(outer, 1):
            search = GridSearchCV(pipeline, grid, scoring="average_precision",
                                  cv=GroupedStratifiedKFold(n_splits=3, random_state=SEED), n_jobs=N_JOBS,
                                  error_score="raise")
            search.fit(X.iloc[train], y.iloc[train])
            for index, params in enumerate(search.cv_results_["params"]):
                trials.append({"model": name, "outer_fold": fold, "candidate": index,
                               "is_default": index == len(grid) - 1, "selected": index == search.best_index_,
                               "params_json": json.dumps(params, default=str),
                               "inner_ap": search.cv_results_["mean_test_score"][index]})
            default_fit = clone(pipeline).fit(X.iloc[train], y.iloc[train])
            for option, estimator in [("nested", search.best_estimator_), ("default", default_fit)]:
                rows.append({"model": name, "option": option, "fold": fold,
                             "pr_auc": average_precision_score(y.iloc[valid], estimator.predict_proba(X.iloc[valid])[:, 1]),
                             "default_selected": search.best_index_ == len(grid) - 1})
            log("D2-3", "outer fold", f"{name} fold {fold}: nested {rows[-2]['pr_auc']:.3f}, default {rows[-1]['pr_auc']:.3f}")
        table(pd.DataFrame(rows), "d2_3_nested_folds.csv")
    folds_frame = pd.DataFrame(rows)
    table(pd.DataFrame(trials), "d2_3_search_trials.csv")
    # Three outer folds: each validation part is half the size of its training part.
    summary = table(pd.DataFrame([{"model": name, "option": option, **corrected_interval(group.pr_auc, ratio=0.5)}
                                  for (name, option), group in folds_frame.groupby(["model", "option"], sort=False)]),
                    "d2_3_nested_summary.csv")
    comparisons = []
    for name, group in folds_frame.groupby("model", sort=False):
        pivot = group.pivot(index="fold", columns="option", values="pr_auc")
        comparisons.append({"model": name, "option": "nested", "reference": "default",
                            **corrected_interval(pivot.nested - pivot.default, ratio=0.5)})
    comparison = table(holm(pd.DataFrame(comparisons)), "d2_3_nested_comparisons.csv")
    curves = learning_curves(X, y, outer)
    return summary, comparison, curves


def learning_curves(X, y, outer):
    """Tuned models trained on 25%, 50% and 100% of each outer training part."""
    rows = []
    for name, pipeline in tuned_pipelines().items():
        for fold, (train, valid) in enumerate(outer, 1):
            for fraction in FRACTIONS:
                sample = train if fraction == 1 else train_test_split(train, train_size=fraction, stratify=y.iloc[train],
                                                                      random_state=SEED)[0]
                estimator = clone(pipeline).fit(X.iloc[sample], y.iloc[sample])
                rows.append({"model": name, "fold": fold, "fraction": fraction, "training_rows": len(sample),
                             "train_ap": average_precision_score(y.iloc[sample], estimator.predict_proba(X.iloc[sample])[:, 1]),
                             "validation_ap": average_precision_score(y.iloc[valid], estimator.predict_proba(X.iloc[valid])[:, 1])})
        log("D2-3", "learning curve", name)
    frame = table(pd.DataFrame(rows), "d2_3_learning_folds.csv")
    summary = table(frame.groupby(["model", "fraction"], sort=False).agg(
        training_rows=("training_rows", "mean"), train_ap=("train_ap", "mean"), validation_ap=("validation_ap", "mean"),
        validation_sd=("validation_ap", "std")).reset_index(), "d2_3_learning_summary.csv")
    learning_curves_figure(summary)
    return summary


def learning_curves_figure(summary):
    """One panel per model in a 2 x 2 grid (the fourth cell holds the legend), so text stays readable in print."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, (name, group) in zip(axes.flat, summary.groupby("model", sort=False)):
        axis.plot(group.training_rows, group.train_ap, marker="o", color="#999999", label="training rows")
        axis.errorbar(group.training_rows, group.validation_ap, yerr=group.validation_sd, marker="o",
                      color=MODEL_COLOURS[name], label="validation rows")
        axis.set(title=name, xlabel="Training rows", ylabel="Average precision")
    legend_cell = axes.flat[-1]
    legend_cell.axis("off")
    legend_cell.legend(*axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=12,
                       title="Grey: score on the rows it trained on\nColour: score on unseen validation rows")
    save_figure(fig, "fig_F_04_learning_curves.png", "Learning curves for the tuned models (3 outer folds)")
