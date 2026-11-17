"""Phase 2: the preprocessing pipeline and its ablation study (training CV only).

Every fitted step (imputers, encoder, scaler, selector, PCA, SMOTE) sits inside one pipeline,
so cross-validation refits it on each training fold and validation rows never influence it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from src.common import ROOT, SEED, figure, load_json, log, save_json, table
from src.data import load_partition
from src.metrics import cv_scores, fold_frame, summarize

# Each variant changes one choice relative to the reference pipeline.
VARIANTS = {
    "reference": {},
    "no_scaling": {"scale": False},
    "no_engineering": {"engineer": False},
    "feature_selection": {"selection": True},
    "pca_95pct": {"pca": True},
    "smote": {"sampling": "smote"},
    "class_weight": {"sampling": "class_weight"},
    "ordinal_encoding": {"encoding": "ordinal"},
    "constant_numeric_imputation": {"numeric_imputation": "constant"},
    "restore_duplicate_frequency": {},
}
REFERENCE_OPTIONS = dict(scale=True, engineer=True, selection=False, pca=False, sampling="none",
                         encoding="onehot", numeric_imputation="median")
# An option must beat the reference by at least this much training-CV AP for BOTH reference
# models before it is adopted; smaller gains are within fold noise and add complexity.
MIN_GAIN = 0.001


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Four booking summaries computed from the current row only (no learned statistics)."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        output = X.copy()
        if self.enabled:
            output["total_nights"] = X.stays_in_week_nights + X.stays_in_weekend_nights
            # A missing child count stays missing here and is imputed by the fold-fitted imputer.
            output["party_size"] = X.adults + X.children + X.babies
            output["lead_days_per_night"] = X.lead_time / output.total_nights.clip(lower=1)
            history = X.previous_cancellations + X.previous_bookings_not_canceled
            output["prior_cancellation_fraction"] = X.previous_cancellations / history.clip(lower=1)
        return output

    def get_feature_names_out(self, input_features=None):
        added = ["total_nights", "party_size", "lead_days_per_night", "prior_cancellation_fraction"] if self.enabled else []
        return np.array(list(self.feature_names_in_) + added, dtype=object)


def build_pipeline(model, scale=True, engineer=True, selection=False, pca=False, sampling="none",
                   encoding="onehot", numeric_imputation="median"):
    """Feature engineering -> imputation/encoding/scaling -> optional reduction/sampling -> model."""
    numeric = SkPipeline([
        ("impute", SimpleImputer(strategy=numeric_imputation, fill_value=0, add_indicator=True)),
        ("scale", StandardScaler() if scale else "passthrough"),
    ])
    if encoding == "onehot":
        # Rare categories (<0.5% of the training fold) share one "infrequent" column; unseen ones map there too.
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    categorical = SkPipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                              ("encode", encoder)])
    prepare = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    steps = [
        ("engineer", FeatureEngineer(engineer)),
        ("prepare", prepare),
        ("variance", VarianceThreshold()),
        ("select", SelectPercentile(f_classif, percentile=80) if selection else "passthrough"),
        ("pca", PCA(n_components=0.95, svd_solver="full", random_state=SEED) if pca else "passthrough"),
        ("sample", SMOTE(random_state=SEED) if sampling == "smote" else "passthrough"),
        ("model", clone(model)),
    ]
    if sampling == "class_weight":
        steps[-1][1].set_params(class_weight="balanced")
    return Pipeline(steps)


def selected_options():
    """The preprocessing options chosen by the Phase 2 ablation."""
    return load_json("results/phase2_selected_pipeline.json")["options"]


# ---------------------------------------------------------------------------------------------
# Extra features extension (run after the final evaluation, training CV only).
# ---------------------------------------------------------------------------------------------
MONTHS = {name: number for number, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"], start=1)}
HIGH_CARDINALITY = ["agent", "company"]


class CalendarFeatures(TransformerMixin, BaseEstimator):
    """Calendar fields derived from the arrival date and the lead time (row level; nothing is learned).

    The booking date is the arrival date minus the lead time, so every new field is known when the
    booking is made. The arrival year is needed only to place a date on the calendar (which weekday it
    was); it is dropped afterwards, so the model never sees the year itself and rule (c) still holds.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.drop(columns=["arrival_date_year"])
        if self.enabled:
            arrival = pd.to_datetime(pd.DataFrame({"year": X.arrival_date_year,
                                                   "month": X.arrival_date_month.map(MONTHS),
                                                   "day": X.arrival_date_day_of_month}))
            booked = arrival - pd.to_timedelta(X.lead_time, unit="D")
            # Weekdays and months are labels, not quantities, so they are one-hot encoded as text.
            output["arrival_weekday"] = arrival.dt.dayofweek.astype(str)
            output["booking_weekday"] = booked.dt.dayofweek.astype(str)
            output["booking_month"] = booked.dt.month.astype(str)
        return output


def low_cardinality_text(X):
    """Text columns other than agent and company (those get target encoding in the extra-features variant)."""
    return [column for column in X.select_dtypes(exclude=np.number).columns if column not in HIGH_CARDINALITY]


def extended_pipeline(model, calendar=False, target_encode=False):
    """The chosen reference pipeline plus the two extra-features options.

    With both options off it is the reference pipeline, apart from dropping the arrival year that
    CalendarFeatures needs as input, so its CV score must match the main results.
    target_encode replaces one-hot columns for agent and company with scikit-learn's TargetEncoder,
    which cross-fits inside each training fold so a row's own label never sets its encoding.
    """
    pipeline = build_pipeline(model, **selected_options())
    steps = list(pipeline.steps)
    steps.insert(1, ("calendar", CalendarFeatures(calendar)))
    if target_encode:
        numeric, categorical = [clone(step) for _, step, _ in pipeline.named_steps["prepare"].transformers]
        target = SkPipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                             ("encode", TargetEncoder(target_type="binary", random_state=SEED))])
        prepare = ColumnTransformer([
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, low_cardinality_text),
            ("target", target, HIGH_CARDINALITY),
        ])
        steps = [(name, prepare if name == "prepare" else step) for name, step in steps]
    return Pipeline(steps)


# ---------------------------------------------------------------------------------------------
# Long tails (13 November 2026): treating long tails and extreme values, on the training folds only.
# ---------------------------------------------------------------------------------------------
class ExtremeValues(TransformerMixin, BaseEstimator):
    """Optionally cap numeric fields at the training fold's 1st and 99th percentiles and/or log them.

    Everything is learned in fit() from the training fold only. The log transform, log(1 + x), is
    applied to fields whose training-fold skewness is above 1; negative values (one negative rate)
    are set to 0 first, because the logarithm needs non-negative inputs. Missing values stay
    missing for the imputer that follows.
    """

    def __init__(self, cap=False, log=False):
        self.cap = cap
        self.log = log

    def fit(self, X, y=None):
        numeric = X.select_dtypes(include=np.number)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.columns_ = list(numeric.columns)
        self.low_, self.high_ = numeric.quantile(0.01), numeric.quantile(0.99)
        skewness = (numeric.clip(self.low_, self.high_, axis=1) if self.cap else numeric).skew()
        self.log_columns_ = [column for column in self.columns_ if skewness[column] > 1]
        return self

    def transform(self, X):
        output = X.copy()
        if self.cap:
            output[self.columns_] = output[self.columns_].clip(self.low_, self.high_, axis=1)
        if self.log:
            output[self.log_columns_] = np.log1p(output[self.log_columns_].clip(lower=0))
        return output

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


EXTREME_OPTIONS = {"reference": {}, "cap at 1st/99th percentile": {"cap": True},
                   "log transform of skewed fields": {"log": True}, "cap and log": {"cap": True, "log": True}}


def extreme_value_pipeline(model, cap=False, log=False):
    """The chosen reference pipeline with ExtremeValues inserted after the engineered fields."""
    steps = list(build_pipeline(model, **selected_options()).steps)
    steps.insert(1, ("extreme", ExtremeValues(cap, log)))
    return Pipeline(steps)


def extreme_value_check():
    """Long tails: do capping or log transforms help? Repeated 2 x 5 group-aware CV, training rows only.

    Uses the same ten splits as the repeated comparison of Table 7 (notebook 06), with corrected
    paired intervals and a Holm adjustment over the six comparisons. Run after the final evaluation,
    so, as for the extra features, the recommended model is not changed whatever the result.
    """
    from sklearn.model_selection import cross_validate

    from src.inference import corrected_interval, holm
    from src.metrics import N_JOBS, repeated_splits

    X, y = load_partition("train")
    splits = repeated_splits(X, y)
    rows = []
    for model_name, model in reference_models().items():
        for label, options in EXTREME_OPTIONS.items():
            scores = cross_validate(extreme_value_pipeline(model, **options), X, y, cv=splits,
                                    scoring="average_precision", n_jobs=N_JOBS, error_score="raise")
            rows += [{"model": model_name, "configuration": label, "fold": index + 1, "pr_auc": value}
                     for index, value in enumerate(scores["test_score"])]
            log("Long tails", "extreme values", f"{model_name}, {label}: CV AP {scores['test_score'].mean():.4f}")
    folds_frame = table(pd.DataFrame(rows), "d2_12_extreme_values_folds.csv")
    summary = table(folds_frame.groupby(["model", "configuration"], sort=False).pr_auc.agg(["mean", "std"])
                    .reset_index(), "d2_12_extreme_values_summary.csv")
    comparisons = []
    for model_name, group in folds_frame.groupby("model", sort=False):
        pivot = group.pivot(index="fold", columns="configuration", values="pr_auc")
        for label in list(EXTREME_OPTIONS)[1:]:
            comparisons.append({"model": model_name, "option": label, "reference": "reference",
                                **corrected_interval(pivot[label] - pivot["reference"])})
    comparisons = table(holm(pd.DataFrame(comparisons)), "d2_12_extreme_values_comparisons.csv")
    return summary, comparisons


def reference_models():
    """A linear and a tree model: preprocessing choices can help one family and not the other."""
    return {"Logistic Regression": LogisticRegression(solver="liblinear", max_iter=2000, random_state=SEED),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED)}


def duplicate_weights(X):
    """How many identical source records each training row represents (1 for most rows)."""
    multiplicity = pd.read_csv(ROOT / "results/phase0_duplicate_multiplicity.csv").set_index("source_row")
    return multiplicity.loc[X.index, "source_multiplicity"].to_numpy(dtype=float)


def run_ablation():
    """Ten preprocessing variants x two reference models, five group-aware CV folds each."""
    X, y = load_partition("train")
    weights = duplicate_weights(X)
    summaries, fold_tables = [], []
    for model_name, model in reference_models().items():
        for name, options in VARIANTS.items():
            pipeline = build_pipeline(model, **options)
            # Weighting by source frequency changes the training loss only; validation rows stay unique.
            fit_params = {"model__sample_weight": weights} if name == "restore_duplicate_frequency" else None
            scores = cv_scores(pipeline, X, y, fit_params)
            summaries.append({"model": model_name, "configuration": name, **summarize(scores)})
            fold_tables.append(fold_frame(scores, name).assign(model=model_name))
            log("Phase 2", "ablation", f"{model_name}, {name}: AP {summaries[-1]['pr_auc_mean']:.3f}")
    ablation = pd.DataFrame(summaries)
    reference = ablation[ablation.configuration == "reference"].set_index("model").pr_auc_mean
    ablation["delta_pr_auc_vs_reference"] = ablation.pr_auc_mean - ablation.model.map(reference)
    table(ablation, "phase2_ablation.csv")
    table(pd.concat(fold_tables, ignore_index=True), "phase2_ablation_folds.csv")

    gains = ablation.pivot(index="configuration", columns="model", values="delta_pr_auc_vs_reference")
    eligible = gains.drop(index=["reference", "restore_duplicate_frequency"])
    helps_both = eligible[(eligible >= MIN_GAIN).all(axis=1)]
    chosen = helps_both.mean(axis=1).idxmax() if len(helps_both) else "reference"
    options = dict(REFERENCE_OPTIONS, **VARIANTS[chosen])
    save_json({"configuration": chosen, "options": options, "metric": "average_precision",
               "selection_rule": f"keep the reference unless an option gains >= {MIN_GAIN} training-CV AP for both "
                                 "reference models; duplicate weighting is a sensitivity check, never selected",
               "cv_rows": len(X), "folds": 5,
               "reference_pr_auc": reference.to_dict()}, "results/phase2_selected_pipeline.json")

    order = list(VARIANTS)
    positions = np.arange(len(order))
    plt.figure(figsize=(9, 5.5))
    for offset, (model_name, frame) in zip([-0.2, 0.2], ablation.groupby("model", sort=False)):
        frame = frame.set_index("configuration").loc[order]
        plt.barh(positions + offset, frame.delta_pr_auc_vs_reference, height=0.4, label=model_name)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.yticks(positions, order)
    plt.legend()
    figure("fig_D_01_ablation.png", "Preprocessing ablation: change in AP versus the reference pipeline",
           "Change in average precision (5-fold group CV)", "Pipeline variant")
    log("Phase 2", "pipeline decision", f"Selected {chosen}: {options}")
    return ablation
