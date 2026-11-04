"""Notebook 06: repeated cross-validation with corrected paired intervals (training data only).

Two repeats of the five group-aware folds (different seeds) give ten paired scores per
configuration. Overlapping training folds make plain t-tests too optimistic, so intervals use the
Nadeau-Bengio correction and each family of comparisons gets a Holm adjustment.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate

from src.common import figure, load_json, log, save_json, table
from src.data import load_partition
from src.evaluate import ensembles
from src.inference import corrected_interval, holm
from src.metrics import N_JOBS, REPEAT_SEEDS, repeated_splits
from src.models import default_pipelines, tuned_pipelines
from src.preprocess import VARIANTS, build_pipeline, duplicate_weights, reference_models

def configurations():
    """Every configuration compared: preprocessing variants, default/tuned models, ensembles."""
    configs = {}
    for model_name, model in reference_models().items():
        short = "LR" if model_name == "Logistic Regression" else "HGB"
        for variant, options in VARIANTS.items():
            configs[f"ablation {short}: {variant}"] = build_pipeline(model, **options)
    for name, pipeline in default_pipelines().items():
        configs[f"default: {name}"] = pipeline
    tuned = tuned_pipelines()
    for name, pipeline in tuned.items():
        configs[f"tuned: {name}"] = pipeline
    voting, stacking = ensembles(load_json("results/phase4_ensemble_selection.json")["pair"], tuned)
    configs["tuned: Voting"] = voting
    configs["tuned: Stacking"] = stacking
    return configs


def repeated_uncertainty():
    X, y = load_partition("train")
    splits = repeated_splits(X, y)
    weights = duplicate_weights(X)
    rows = []
    for label, estimator in configurations().items():
        params = {"model__sample_weight": weights} if label.endswith("restore_duplicate_frequency") else None
        scores = cross_validate(estimator, X, y, cv=splits, scoring={"pr_auc": "average_precision", "f1": "f1"},
                                n_jobs=N_JOBS, error_score="raise", params=params)
        for index in range(len(splits)):
            rows.append({"configuration": label, "fold": index + 1, "repeat": index // 5 + 1,
                         "pr_auc": scores["test_pr_auc"][index], "f1": scores["test_f1"][index],
                         "fit_seconds": scores["fit_time"][index]})
        log("D2-2", "repeated CV", f"{label}: AP {scores['test_pr_auc'].mean():.3f}")
        table(pd.DataFrame(rows), "d2_2_repeated_folds.csv")
    folds_frame = pd.DataFrame(rows)
    save_json({"seeds": REPEAT_SEEDS, "folds": 5, "fits_per_configuration": len(splits), "train_rows": len(X),
               "interval": "corrected resampled t (SE = s * sqrt(1/n + 0.25), df = n - 1); Holm within each family",
               "scope": "fixed configurations chosen earlier; repeats measure split sensitivity, not selection bias"},
              "results/d2_2_protocol.json")
    summary = table(pd.DataFrame([{"configuration": label, **corrected_interval(group.pr_auc)}
                                  for label, group in folds_frame.groupby("configuration", sort=False)]),
                    "d2_2_repeated_summary.csv")
    pivot = folds_frame.pivot(index="fold", columns="configuration", values="pr_auc")

    def family(name, pairs):
        frame = pd.DataFrame([{"option": option, "reference": reference,
                               **corrected_interval(pivot[option] - pivot[reference])} for option, reference in pairs])
        return holm(frame).assign(family=name)

    comparisons = []
    for short in ["LR", "HGB"]:
        reference = f"ablation {short}: reference"
        comparisons.append(family(f"ablation {short}", [(f"ablation {short}: {variant}", reference)
                                                         for variant in VARIANTS if variant != "reference"]))
    comparisons.append(family("tuning", [(f"tuned: {name}", f"default: {name}") for name in default_pipelines()]))
    recommended = load_json("results/phase4_ensemble_selection.json")["recommended"]
    reference = f"tuned: {recommended}"
    comparisons.append(family("models", [(label, reference) for label in pivot.columns
                                         if label.startswith("tuned: ") and label != reference]))
    comparison = table(pd.concat(comparisons, ignore_index=True), "d2_2_paired_comparisons.csv")

    shown = summary[~summary.configuration.str.startswith("ablation")].sort_values("mean")
    plt.figure(figsize=(9, 5))
    plt.barh(shown.configuration, shown["mean"], xerr=[shown["mean"] - shown.ci_low, shown.ci_high - shown["mean"]],
             color="#0072B2")
    figure("fig_D_02_repeated_uncertainty.png", "Repeated CV (2 x 5 folds): mean AP with corrected 95% intervals",
           "Average precision", "Configuration")
    return summary, comparison
