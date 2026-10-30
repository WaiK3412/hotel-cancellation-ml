"""Phase 0 extension: do the four screened datasets differ in how well they support Parts B, E and G?

The original screening (results/phase0_scorecard.csv) only counted data-quality issues; nothing was
fitted. This script adds a small, identical modelling check on every candidate:

- the same label-free cleaning idea as the hotel data (trim text, missing tokens -> NaN, drop exact
  duplicate records);
- fields that are documented as known only after the outcome, or that are row identifiers, are dropped;
- a 20% holdout is set aside per dataset and NEVER used; everything below is 5-fold CV on the 80%;
- three library-default models (LR, RF, HGB) through the project's own preprocessing pipeline;
- soft voting of every pair, from the same fold predictions (no refitting);
- K-means on a 4,000-row training sample, compared with the labels (Part B evidence).

For the hotel data the project's real training partition, excluded fields and booking groups are used,
so its numbers are the same population as the rest of the project. It is a screening check with
library defaults, not tuned results.

Run: python scripts/screen_candidate_models.py
Writes: results/phase0_candidate_models.csv, results/phase0_candidate_folds.csv,
        results/phase0_candidate_clustering.csv, results/phase0_candidate_summary.csv
"""
import os
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, average_precision_score, normalized_mutual_info_score, \
    roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.common import ROOT, SEED, table
from src.data import MISSING, booking_groups, load_partition, read_candidate
from src.preprocess import build_pipeline

# Per candidate: target rule, fields dropped (with the reason) and the column that groups rows.
CANDIDATES = {
    "Adult": {
        "target": lambda d: (d["income"] == ">50K").astype(int), "target_column": "income",
        "drop": {"fnlwgt": "census sampling weight, not a property of the person"},
        "categorical_codes": [], "group": None,
    },
    "Bank Marketing": {
        "target": lambda d: (d["y"] == "yes").astype(int), "target_column": "y",
        "drop": {"duration": "call length is known only after the call ends (UCI documentation says to "
                             "discard it for a realistic model)"},
        "categorical_codes": [], "group": None,
    },
    "Diabetes": {
        "target": lambda d: (d["readmitted"] == "<30").astype(int), "target_column": "readmitted",
        "drop": {"encounter_id": "row identifier", "patient_nbr": "patient identifier (used as the group)"},
        # Integer codes that name categories, not quantities.
        "categorical_codes": ["admission_type_id", "discharge_disposition_id", "admission_source_id"],
        "group": "patient_nbr",
    },
}
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
}
CLUSTER_SAMPLE = 4000


def clean_candidate(raw):
    """Label-free cleaning: trim text, missing tokens -> NaN, numbers parsed, exact duplicates removed."""
    clean = raw.copy()
    for column in clean.select_dtypes("object"):
        stripped = clean[column].astype(str).str.strip()
        clean[column] = stripped.mask(stripped.str.lower().isin(MISSING))
        parsed = pd.to_numeric(clean[column], errors="coerce")
        if parsed.notna().sum() == clean[column].notna().sum():
            clean[column] = parsed
    return clean.loc[~clean.duplicated()].reset_index(drop=True)


def prepare(name):
    """Training predictors, labels and groups for one candidate (its 20% holdout is discarded)."""
    if name == "Hotel Booking Demand":
        X, y = load_partition("train")
        return X, y.to_numpy(), booking_groups(X), {"reservation_status etc.": "outcome fields (project rule)",
                                                    "country, deposit_type, parking, year": "Phase 1 exclusions"}
    spec = CANDIDATES[name]
    data = clean_candidate(read_candidate(name))
    y = spec["target"](data).to_numpy()
    groups = data[spec["group"]].to_numpy() if spec["group"] else np.arange(len(data))
    X = data.drop(columns=[spec["target_column"]] + list(spec["drop"]))
    for column in spec["categorical_codes"]:
        X[column] = X[column].astype(str)
    holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED) if spec["group"] \
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, _ = next(holdout.split(X, y, groups))
    return X.iloc[train].reset_index(drop=True), y[train], groups[train], spec["drop"]


def model_folds(name, X, y, groups):
    """Per-fold AP and ROC-AUC for each default model and each soft-voting pair."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (train, valid) in enumerate(splitter.split(X, y, groups), start=1):
        probabilities = {}
        for model_name, model in MODELS.items():
            pipeline = build_pipeline(clone(model), engineer=False)
            start = perf_counter()
            pipeline.fit(X.iloc[train], y[train])
            seconds = perf_counter() - start
            probabilities[model_name] = pipeline.predict_proba(X.iloc[valid])[:, 1]
            rows.append({"dataset": name, "fold": fold, "model": model_name, "fit_seconds": seconds})
        for first, second in combinations(MODELS, 2):
            probabilities[f"Voting {first} + {second}"] = (probabilities[first] + probabilities[second]) / 2
        for model_name, p in probabilities.items():
            record = next((r for r in rows if r["fold"] == fold and r["model"] == model_name
                           and r["dataset"] == name), None)
            if record is None:
                record = {"dataset": name, "fold": fold, "model": model_name, "fit_seconds": np.nan}
                rows.append(record)
            record.update({"pr_auc": average_precision_score(y[valid], p), "roc_auc": roc_auc_score(y[valid], p),
                           "positive_rate": float(y[valid].mean())})
        print(f"  {name}: fold {fold} done", flush=True)
    return pd.DataFrame(rows)


def clustering(name, X, y):
    """K-means on a stratified training sample: silhouette-chosen k and k=2, compared with the labels."""
    rng = np.random.default_rng(SEED)
    take = min(CLUSTER_SAMPLE, len(X))
    index = np.sort(rng.choice(len(X), size=take, replace=False))
    sample, labels = X.iloc[index], y[index]
    prepare_step = build_pipeline(LogisticRegression(), engineer=False)[:-1]  # everything except the model
    Z = prepare_step.fit_transform(sample, labels)
    rows = []
    for k in range(2, 7):
        clusters = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
        rows.append({"dataset": name, "k": k, "silhouette": silhouette_score(Z, clusters, random_state=SEED),
                     "ari_vs_label": adjusted_rand_score(labels, clusters),
                     "nmi_vs_label": normalized_mutual_info_score(labels, clusters), "sample_rows": take,
                     "encoded_columns": Z.shape[1]})
    return pd.DataFrame(rows)


def main():
    os.chdir(ROOT)
    fold_tables, cluster_tables, dropped, sizes = [], [], {}, {}
    for name in ["Hotel Booking Demand", "Adult", "Bank Marketing", "Diabetes"]:
        start = perf_counter()
        X, y, groups, drop = prepare(name)
        dropped[name] = drop
        sizes[name] = (len(X), X.shape[1])
        print(f"{name}: {len(X)} training rows, {X.shape[1]} predictors, positive rate {y.mean():.3f}", flush=True)
        fold_tables.append(model_folds(name, X, y, groups))
        cluster_tables.append(clustering(name, X, y))
        print(f"{name}: finished in {perf_counter() - start:.0f} s", flush=True)

    folds = table(pd.concat(fold_tables, ignore_index=True), "phase0_candidate_folds.csv")
    models = folds.groupby(["dataset", "model"], sort=False).agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"), roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"), positive_rate=("positive_rate", "mean"),
        fit_seconds_mean=("fit_seconds", "mean")).reset_index()
    table(models, "phase0_candidate_models.csv")
    clusters = table(pd.concat(cluster_tables, ignore_index=True), "phase0_candidate_clustering.csv")

    scorecard = pd.read_csv(ROOT / "results/phase0_scorecard.csv").set_index("dataset")
    issue_columns = ["missing_cells", "duplicates", "iqr_outlier_cells", "skewed_features", "conflicting_label_rows",
                     "implausible_rows", "whitespace_rows", "case_variant_groups", "mixed_numeric_rows",
                     "semantic_variant_rows"]
    summary = []
    for name, group in models.groupby("dataset", sort=False):
        singles = group[~group.model.str.startswith("Voting")].set_index("model")
        pairs = group[group.model.str.startswith("Voting")].set_index("model")
        best, worst = singles.pr_auc_mean.idxmax(), singles.pr_auc_mean.idxmin()
        top_two = singles.pr_auc_mean.nlargest(2).index
        pair_name = next(p for p in pairs.index if all(m in p for m in top_two))
        chosen_k = clusters[clusters.dataset == name].sort_values("silhouette").iloc[-1]
        k2 = clusters[(clusters.dataset == name) & (clusters.k == 2)].iloc[0]
        row = scorecard.loc[name]
        summary.append({
            "dataset": name, "training_rows": sizes[name][0], "predictors": sizes[name][1],
            "positive_rate": float(singles.positive_rate.iloc[0]),
            "lr_pr_auc": singles.loc["Logistic Regression", "pr_auc_mean"],
            "rf_pr_auc": singles.loc["Random Forest", "pr_auc_mean"],
            "hgb_pr_auc": singles.loc["Hist Gradient Boosting", "pr_auc_mean"],
            "best_single": best, "best_single_pr_auc": singles.pr_auc_mean.max(),
            "best_single_roc_auc": singles.loc[best, "roc_auc_mean"],
            "model_spread_pr_auc": singles.pr_auc_mean.max() - singles.pr_auc_mean.min(), "weakest_model": worst,
            "top_two_voting_pr_auc": pairs.loc[pair_name, "pr_auc_mean"],
            "voting_gain_over_best": pairs.loc[pair_name, "pr_auc_mean"] - singles.pr_auc_mean.max(),
            "kmeans_k_by_silhouette": int(chosen_k.k), "ari_at_that_k": chosen_k.ari_vs_label,
            "ari_k2": k2.ari_vs_label, "nmi_k2": k2.nmi_vs_label,
            "data_quality_issue_types_present": int((row[issue_columns] > 0).sum()),
            "data_quality_issue_types_checked": len(issue_columns),
            "imbalance_ratio": row.imbalance_ratio,
            "fields_dropped": "; ".join(f"{k}: {v}" for k, v in dropped[name].items()),
        })
    summary = pd.DataFrame(summary)
    table(summary, "phase0_candidate_summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.drop(columns="fields_dropped").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
