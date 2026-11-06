"""Phase 1: exploratory analysis of the training rows and the supervised-vs-clustering comparison.

EDA looks at every source field, including the leakage candidates, because spotting suspicious
patterns is part of the investigation. Test rows are never loaded here.
"""
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import ROOT, SEED, figure, log, save_figure, table
from src.data import CATEGORICAL, OUTCOME_FIELDS, TARGET, booking_groups, load_partition, missing_mask

MEANINGS = {
    "hotel": "City hotel or resort hotel",
    "is_canceled": "Target: 1 if the booking was cancelled, 0 otherwise",
    "lead_time": "Days between entering the booking and the arrival date",
    "arrival_date_year": "Year of arrival",
    "arrival_date_month": "Month of arrival",
    "arrival_date_week_number": "Week number of arrival",
    "arrival_date_day_of_month": "Day of the month of arrival",
    "stays_in_weekend_nights": "Booked weekend nights (Saturday or Sunday)",
    "stays_in_week_nights": "Booked week nights (Monday to Friday)",
    "adults": "Number of adults",
    "children": "Number of children",
    "babies": "Number of babies",
    "meal": "Meal package booked (Undefined merged into SC: no meal)",
    "country": "Country of origin (often only confirmed at check-in)",
    "market_segment": "Market segment (e.g. online or offline travel agent, direct, groups)",
    "distribution_channel": "Booking distribution channel",
    "is_repeated_guest": "1 if the guest had stayed before",
    "previous_cancellations": "Bookings this customer cancelled before the current one",
    "previous_bookings_not_canceled": "Bookings this customer completed before the current one",
    "reserved_room_type": "Code of the room type reserved",
    "assigned_room_type": "Code of the room type assigned (recorded at check-in; excluded)",
    "booking_changes": "Changes made to the booking until check-in or cancellation (excluded)",
    "deposit_type": "From payments recorded before arrival or cancellation: No Deposit, Non Refund "
                    "(paid at least the full stay) or Refundable (paid less)",
    "agent": "Travel agency identifier (a category, not a number)",
    "company": "Paying company identifier (a category, not a number)",
    "days_in_waiting_list": "Days on the waiting list before confirmation (excluded)",
    "customer_type": "Contract, Group, Transient or Transient-Party",
    "adr": "Average daily rate (source currency units)",
    "required_car_parking_spaces": "Car parking spaces requested",
    "total_of_special_requests": "Number of special requests (e.g. twin bed, high floor)",
    "reservation_status": "Final status: Canceled, Check-Out or No-Show (outcome; excluded)",
    "reservation_status_date": "Date the final status was set (outcome; excluded)",
}
NUMERIC_FOR_PLOTS = ["lead_time", "adr", "stays_in_week_nights", "total_of_special_requests"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
LEAD_BANDS = ["0-7", "8-30", "31-90", "91-180", "181-365", ">365"]


def training_frame():
    """All training rows with every predictor (leakage candidates included) plus the target."""
    X, y = load_partition("train", excluded=[])
    return X, y, X.assign(**{TARGET: y})


def dictionary(data):
    """Meaning, type, range or categories and missing share of every field (training rows)."""
    rows = []
    for column in data:
        values = data[column]
        numeric = pd.api.types.is_numeric_dtype(values) and column not in CATEGORICAL
        if numeric:
            span = f"{values.min()} to {values.max()}"
        else:
            categories = sorted(values.dropna().astype(str).unique())
            span = f"{len(categories)} categories" if len(categories) > 12 else "; ".join(categories)
        role = "target" if column == TARGET else "excluded (outcome)" if column in OUTCOME_FIELDS else "predictor"
        rows.append({"feature": column, "type": "numeric" if numeric else "categorical", "meaning": MEANINGS[column],
                     "range_or_categories": span, "missing_pct": values.isna().mean() * 100, "model_role": role})
    return pd.DataFrame(rows)


def cramers_v(first, second):
    """Uncorrected Cramer's V between two categorical columns (0 = none, 1 = perfect)."""
    counts = pd.crosstab(first.fillna("Missing"), second.fillna("Missing"))
    if min(counts.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(counts, correction=False)[0]
    return float(np.sqrt(chi2 / (counts.values.sum() * (min(counts.shape) - 1))))


def annotated_heatmap(matrix, axis, **style):
    """Lower-triangle heatmap with the value printed in each cell, so the strength can be read off."""
    hide = np.triu(np.ones(matrix.shape, dtype=bool), k=1)  # the upper triangle repeats the lower one
    sns.heatmap(matrix, mask=hide, annot=True, fmt=".2f", annot_kws={"fontsize": 7}, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=axis, **style)
    axis.grid(False)
    axis.tick_params(axis="x", labelrotation=90)


def correlation_figures(correlation, association):
    """C_01: Pearson correlation of numeric fields. C_02: Cramér's V between category fields. One per figure,
    so each can be printed at full page width with readable cell values."""
    for matrix, name, title, style in [
        (correlation, "fig_C_01_correlations.png", "Numeric fields: Pearson correlation (-1 to 1, training rows)",
         {"cmap": "coolwarm", "vmin": -1, "vmax": 1}),
        (association, "fig_C_02_associations.png", "Category fields: Cramér's V (0 to 1, training rows)",
         {"cmap": "viridis", "vmin": 0, "vmax": 1}),
    ]:
        fig, axis = plt.subplots(figsize=(9, 7.5))
        annotated_heatmap(matrix, axis, **style)
        axis.set_xlabel("Field")
        axis.set_ylabel("Field")
        save_figure(fig, name, title)


def quality_overview_figure(missing, duplicates, balance, noise):
    """C_13: four data-quality issues in one figure, drawn from the saved phase1 tables."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    top = missing[missing.missing_pct > 0].sort_values("missing_pct")
    axes[0, 0].barh(top.feature, top.missing_pct, color="#0072B2")
    axes[0, 0].set(title="Missing values", xlabel="Training rows missing (%)")
    axes[0, 1].barh(["exact duplicates\nremoved (source)", "training rows sharing\nall booking-time fields"],
                    [duplicates.rows.iloc[0], duplicates.rows.iloc[1]], color="#E69F00")
    axes[0, 1].set(title="Duplicates", xlabel="Rows")
    axes[1, 0].bar(["not cancelled", "cancelled"], balance.bookings, color=["#0072B2", "#D55E00"])
    for x, (count, percent) in enumerate(zip(balance.bookings, balance.percent)):
        axes[1, 0].text(x, count, f"{percent:.1f}%", ha="center", va="bottom")
    axes[1, 0].set(title="Class imbalance", ylabel="Training bookings")
    axes[1, 1].barh(noise.issue.str.replace(" (adults + children + babies = 0)", "", regex=False), noise.rows,
                    color="#009E73")
    axes[1, 1].set(title="Noise and implausible values", xlabel="Training rows")
    save_figure(fig, "fig_C_13_quality_overview.png", "Data-quality issues (removed duplicates counted in the source file; the rest in the training rows)")


def disguised_missing(X):
    """Category levels that mean 'not recorded' without being blank (the brief's disguised missing values)."""
    rows = [{"field": field, "level": "Undefined", "training_rows": int((X[field] == "Undefined").sum())}
            for field in ["market_segment", "distribution_channel"]]
    return table(pd.DataFrame(rows), "phase1_disguised_missing.csv")


def describe_and_correlate(X, y, data):
    table(data.describe(include="all").T.reset_index(names="feature"), "phase1_descriptive_statistics.csv")
    correlation = data.select_dtypes("number").corr()
    table(correlation.reset_index(names="feature"), "phase1_numeric_correlations.csv")
    categorical = [column for column in CATEGORICAL if column in X]
    association = pd.DataFrame({a: {b: cramers_v(X[a], X[b]) for b in categorical} for a in categorical})
    table(association.reset_index(names="feature"), "phase1_categorical_associations.csv")
    correlation_figures(correlation, association)
    target = table(pd.DataFrame([{"feature": column, "cramers_v_to_target": cramers_v(X[column], y)}
                                 for column in categorical]).sort_values("cramers_v_to_target"),
                   "phase1_target_associations.csv")
    plt.figure(figsize=(8, 4.5))
    plt.barh(target.feature, target.cramers_v_to_target, color="#0072B2")
    figure("fig_C_03_target_association.png", "How strongly each category field relates to cancellation",
           "Cramér's V with is_canceled", "Field")


def distributions_figure(X, y):
    """Distributions of the four most informative numeric fields, split by outcome."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, column in zip(axes.flat, NUMERIC_FOR_PLOTS):
        upper = X[column].quantile(0.99)
        counts = X[column].nunique() <= 10
        bins = np.arange(X[column].min(), upper + 2) - 0.5 if counts else np.linspace(X[column].min(), upper, 40)
        for label, colour, name in [(0, "#0072B2", "not cancelled"), (1, "#D55E00", "cancelled")]:
            axis.hist(X.loc[y == label, column].clip(upper=upper).dropna(), bins=bins, alpha=0.55, color=colour,
                      label=name, density=True)
        axis.set(title=column, xlabel=column.replace("_", " "), ylabel="Share of bookings (density)")
        if counts:
            axis.set_xticks(np.arange(X[column].min(), upper + 1))
        axis.legend()
    save_figure(fig, "fig_C_05_distributions.png",
                "Distributions by outcome (training rows; values above the 99th percentile drawn at it)")


def distributions(X, y):
    """Distributions split by outcome, and skewness of every numeric field."""
    distributions_figure(X, y)
    numeric = X.select_dtypes("number")
    skew = table(numeric.skew().rename("skew").rename_axis("feature").reset_index().sort_values("skew"),
                 "phase1_skewness.csv")
    plt.figure(figsize=(8, 5))
    plt.barh(skew.feature, skew["skew"], color="#0072B2")
    plt.axvline(1, color="#D55E00", linestyle="--", label="|skew| = 1")
    plt.axvline(-1, color="#D55E00", linestyle="--")
    plt.legend()
    figure("fig_C_06_skewness.png", "Skewness of numeric fields (training rows)", "Skewness", "Field")


def quality(X, y, data):
    """Missing values, duplicates, outliers, class imbalance, noise and inconsistencies."""
    missing = table(pd.DataFrame({"feature": data.columns, "missing_cells": data.isna().sum().values,
                                  "missing_pct": data.isna().mean().values * 100}), "phase1_missing.csv")
    shown = missing[missing.missing_cells > 0].sort_values("missing_pct")
    plt.figure(figsize=(7, 3.5))
    plt.barh(shown.feature, shown.missing_pct, color="#0072B2")
    figure("fig_C_07_missing_values.png", "Missing values after token standardisation (training rows)",
           "Rows missing (%)", "Field")

    counts = pd.read_csv(ROOT / "results/phase0_cleaning_counts.csv")
    shared_group = pd.Series(booking_groups(X)).duplicated(keep=False).to_numpy()
    duplicates = table(pd.DataFrame([
        {"check": "exact duplicate source records removed (Phase 0)", "rows": int(counts.removed.sum())},
        {"check": "training rows identical to another on all booking-time key fields", "rows": int(shared_group.sum())},
        {"check": "exact duplicates left in cleaned training rows", "rows": int(data.duplicated().sum())},
    ]), "phase1_duplicates.csv")
    plt.figure(figsize=(8, 3))
    plt.barh(duplicates.check, duplicates.rows, color="#0072B2")
    figure("fig_C_08_duplicates.png", "Duplicate records and near-duplicate booking groups", "Rows", "")

    outliers = []
    for column in X.select_dtypes("number"):
        q1, q3 = X[column].quantile([0.25, 0.75])
        flagged = (X[column] < q1 - 1.5 * (q3 - q1)) | (X[column] > q3 + 1.5 * (q3 - q1))
        outliers.append({"feature": column, "q1": q1, "q3": q3, "iqr_flagged_rows": int(flagged.sum()),
                         "flagged_pct": flagged.mean() * 100})
    outliers = table(pd.DataFrame(outliers).sort_values("flagged_pct"), "phase1_outliers.csv")
    plt.figure(figsize=(8, 5))
    plt.barh(outliers.feature, outliers.flagged_pct, color="#0072B2")
    figure("fig_C_09_outliers.png", "Rows flagged by the 1.5 x IQR rule (flags, not confirmed errors)",
           "Rows flagged (%)", "Field")

    balance = table(y.value_counts().sort_index().rename("bookings").rename_axis(TARGET).reset_index()
                    .assign(percent=lambda frame: frame.bookings / frame.bookings.sum() * 100), "phase1_class_balance.csv")
    plt.figure(figsize=(5, 3.5))
    plt.bar(["not cancelled (0)", "cancelled (1)"], balance.bookings, color=["#0072B2", "#D55E00"])
    figure("fig_C_10_class_imbalance.png", "Class balance (training rows)", "Outcome", "Bookings")

    groups = booking_groups(X)
    both_labels = pd.Series(y.to_numpy()).groupby(groups).transform("nunique").to_numpy() > 1
    noise = table(pd.DataFrame([
        {"issue": "rows in booking groups with both outcomes", "rows": int(both_labels.sum())},
        {"issue": "zero guests (adults + children + babies = 0)",
         "rows": int((X.adults + X.children.fillna(0) + X.babies == 0).sum())},
        {"issue": "negative average daily rate", "rows": int((X.adr < 0).sum())},
        {"issue": "zero-night stays", "rows": int((X.stays_in_week_nights + X.stays_in_weekend_nights == 0).sum())},
    ]), "phase1_noise.csv")
    plt.figure(figsize=(8, 3))
    plt.barh(noise.issue, noise.rows, color="#0072B2")
    figure("fig_C_11_noise.png", "Noise indicators in the training rows", "Rows", "")

    inconsistency = []
    for column in data.select_dtypes("object"):
        text = data[column].dropna().astype(str)
        variants = text.str.strip().str.lower()
        inconsistency.append({"feature": column, "whitespace_rows": int((text != text.str.strip()).sum()),
                              "case_variant_groups": int((pd.DataFrame({"v": variants, "t": text}).drop_duplicates()
                                                          .groupby("v").t.nunique() > 1).sum())})
    table(pd.DataFrame(inconsistency), "phase1_inconsistencies.csv")
    raw_meal = pd.read_csv(ROOT / "data/raw/hotels.csv", usecols=["meal"], keep_default_na=False).meal
    summary = table(pd.DataFrame([
        {"issue": "meal code 'Undefined' merged into 'SC' (source rows)", "rows_or_groups": int((raw_meal.str.strip() == "Undefined").sum())},
        {"issue": "leading or trailing whitespace (cleaned)", "rows_or_groups": int(sum(r["whitespace_rows"] for r in inconsistency))},
        {"issue": "upper/lower-case variants (cleaned)", "rows_or_groups": int(sum(r["case_variant_groups"] for r in inconsistency))},
    ]), "phase1_inconsistency_summary.csv")
    plt.figure(figsize=(8, 3))
    plt.barh(summary.issue, summary.rows_or_groups, color="#0072B2")
    figure("fig_C_12_inconsistencies.png", "Representation inconsistencies (zero is a valid finding)", "Count", "")


def cancellation_rates(X, y):
    """Cancellation rate across the strongest booking-time fields (kept predictors only)."""
    bands = pd.cut(X.lead_time, [-1, 7, 30, 90, 180, 365, X.lead_time.max()],
                   labels=["0-7", "8-30", "31-90", "91-180", "181-365", ">365"])
    groups = {"lead time (days)": bands.astype(str), "market segment": X.market_segment.fillna("Missing"),
              "customer type": X.customer_type, "hotel": X.hotel,
              "previous cancellations": np.where(X.previous_cancellations > 0, "one or more", "none"),
              "arrival month": X.arrival_date_month}
    rows = []
    for name, values in groups.items():
        frame = pd.DataFrame({"group": values, "cancelled": y.to_numpy()})
        for group, cancelled in frame.groupby("group").cancelled:
            rows.append({"field": name, "group": group, "rows": len(cancelled), "cancel_rate": cancelled.mean()})
    rates = table(pd.DataFrame(rows), "phase1_cancellation_rates.csv")
    cancellation_rates_figure(rates, y.mean())


def cancellation_rates_figure(rates, overall):
    """One panel per field; ordered groups (lead time, month) read top to bottom in their natural order."""
    orders = {"lead time (days)": LEAD_BANDS, "arrival month": MONTHS}
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))  # 3 x 2 so labels stay readable in print
    for axis, (name, frame) in zip(axes.flat, rates.groupby("field", sort=False)):
        frame = frame[frame.rows >= 50].set_index("group")
        if name in orders:
            frame = frame.reindex([group for group in orders[name] if group in frame.index])
        axis.barh(frame.index.astype(str), frame.cancel_rate, color="#0072B2")
        axis.invert_yaxis()
        axis.axvline(overall, color="#D55E00", linestyle="--", label=f"overall {overall:.3f}")
        axis.set(title=name, xlabel="Cancellation rate", xlim=(0, 1))
        axis.legend(loc="lower right")
    save_figure(fig, "fig_C_04_cancellation_rates.png", "Cancellation rate by booking characteristic (training rows)")


def eda():
    """All ten Part C checks on the training rows; returns the data dictionary."""
    X, y, data = training_frame()
    all_training = pd.read_csv(ROOT / "data/processed/cleaned.csv", index_col="source_row",
                               dtype={column: str for column in CATEGORICAL}).loc[X.index]
    dictionary_frame = table(dictionary(all_training), "phase1_feature_dictionary.csv")
    describe_and_correlate(X, y, data)
    distributions(X, y)
    quality(X, y, data)
    quality_overview_figure(*[pd.read_csv(ROOT / "results" / name) for name in
                              ["phase1_missing.csv", "phase1_duplicates.csv", "phase1_class_balance.csv",
                               "phase1_noise.csv"]])
    disguised_missing(X)
    cancellation_rates(X, y)
    log("Phase 1", "EDA", f"Training rows only: {len(X)}. Tables in results/phase1_*.csv, figures fig_C_*.")
    return dictionary_frame


def paradigms():
    """Part B experiment: supervised logistic regression versus label-free K-Means."""
    from src.metrics import cv_scores, fold_frame, summarize
    from src.preprocess import build_pipeline, reference_models

    X, y = load_partition("train")
    start = perf_counter()
    scores = cv_scores(build_pipeline(reference_models()["Logistic Regression"]), X, y)
    supervised_seconds = perf_counter() - start
    supervised = summarize(scores)
    table(fold_frame(scores, "logistic regression"), "phase1_supervised_folds.csv")
    table(pd.DataFrame([supervised]), "phase1_supervised_cv.csv")

    # K-Means sees a fixed 4,000-row training sample and numeric fields only; labels are used
    # only after clustering, to measure how well clusters line up with cancellations.
    _, sample_rows = train_test_split(np.arange(len(X)), test_size=4000, stratify=y, random_state=SEED)
    numeric = ["lead_time", "adr", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
               "babies", "previous_cancellations", "total_of_special_requests"]
    sample, labels = X.iloc[sample_rows][numeric], y.iloc[sample_rows]
    trials, fitted = [], {}
    start = perf_counter()
    for k in range(2, 7):
        pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                             ("cluster", KMeans(n_clusters=k, n_init=10, random_state=SEED))]).fit(sample)
        clusters = pipeline.named_steps["cluster"].labels_
        silhouette = silhouette_score(pipeline[:-1].transform(sample), clusters, sample_size=1500, random_state=SEED)
        trials.append({"k": k, "inertia": pipeline.named_steps["cluster"].inertia_, "silhouette": silhouette})
        fitted[k] = clusters
    clustering_seconds = perf_counter() - start
    trial_frame = table(pd.DataFrame(trials), "phase1_clustering_k.csv")
    k = int(trial_frame.loc[trial_frame.silhouette.idxmax(), "k"])  # chosen without labels
    clusters = fitted[k]
    table(pd.crosstab(clusters, labels).reset_index(names="cluster"), "phase1_cluster_vs_label.csv")
    profiles = sample.assign(cluster=clusters, cancelled=labels.to_numpy()).groupby("cluster").mean().reset_index()
    profiles["bookings"] = pd.Series(clusters).value_counts().reindex(profiles.cluster).to_numpy()
    table(profiles, "phase1_cluster_profiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(trial_frame.k, trial_frame.inertia, marker="o")
    axes[0].set(title="Elbow", xlabel="Clusters k", ylabel="Inertia (within-cluster squared distance)")
    axes[1].plot(trial_frame.k, trial_frame.silhouette, marker="o")
    axes[1].set(title="Silhouette (chooses k without labels)", xlabel="Clusters k", ylabel="Silhouette")
    save_figure(fig, "fig_B_01_clustering_selection.png", "K-Means on a 4,000-row training sample")
    comparison = table(pd.DataFrame([
        {"approach": "Logistic regression (supervised)", "uses_labels": "yes, to fit", "rows": len(X), "cv_folds": 5,
         "cv_pr_auc": supervised["pr_auc_mean"], "cv_f1": supervised["f1_mean"], "ari": np.nan, "nmi": np.nan,
         "k": np.nan, "wall_seconds": supervised_seconds},
        {"approach": "K-Means (unsupervised)", "uses_labels": "no; only to score agreement afterwards",
         "rows": len(sample), "cv_folds": np.nan, "cv_pr_auc": np.nan, "cv_f1": np.nan,
         "ari": adjusted_rand_score(labels, clusters), "nmi": normalized_mutual_info_score(labels, clusters),
         "k": k, "wall_seconds": clustering_seconds},
    ]), "phase1_paradigm_comparison.csv")
    log("Phase 1", "paradigms", f"LR CV AP {supervised['pr_auc_mean']:.3f}; K-Means k={k}, "
        f"ARI {comparison.ari.iloc[1]:.3f}, NMI {comparison.nmi.iloc[1]:.3f}.")
    return comparison
