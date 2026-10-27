"""Shared evaluation settings: the seven metrics, the group-aware folds and CV helpers.

Every experiment uses the same folds, so differences between configurations are paired
comparisons on identical validation rows.
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from src.common import SEED

# Average precision ("pr_auc") is the primary metric: the task is to rank the minority of bookings
# that cancel. The other six describe threshold behaviour and overall discrimination.
SCORING = {"pr_auc": "average_precision", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
           "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
# Parallel CV workers. Three keeps peak memory (several Random Forests at once) under ~4 GB.
N_JOBS = 3


class GroupedStratifiedKFold(StratifiedGroupKFold):
    """Stratified K-fold that keeps each booking group (identical booking-time fields) in one fold.

    When scikit-learn calls split(X, y) without groups (as searches and StackingClassifier do),
    the groups are computed from X itself, so the same rule holds everywhere automatically.
    Computing the folds for ~70,000 rows takes scikit-learn about 20 s, and many experiments
    reuse the same rows, so identical inputs reuse the cached result (same folds, same order).
    """

    _cache = {}

    def __init__(self, n_splits=5, shuffle=True, random_state=SEED):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        if groups is None:
            from src.data import booking_groups
            groups = booking_groups(X)
        key = (self.n_splits, self.random_state, hashlib.sha1(np.ascontiguousarray(groups).tobytes()).hexdigest(),
               hashlib.sha1(np.ascontiguousarray(np.asarray(y)).tobytes()).hexdigest())
        if key not in self._cache:
            self._cache[key] = [(train, test) for train, test in super().split(X, y, groups)]
        return iter(self._cache[key])


def folds(seed=SEED):
    """The five group-aware stratified folds used by every training-set experiment."""
    return GroupedStratifiedKFold(n_splits=5, random_state=seed)


REPEAT_SEEDS = [SEED, SEED + 1]


def repeated_splits(X, y):
    """Ten (train, validation) index pairs: two seeds x five group-aware folds."""
    return [split for seed in REPEAT_SEEDS for split in folds(seed).split(X, y)]


def cv_scores(pipeline, X, y, fit_params=None):
    """Five-fold CV scores (validation and training) for one fixed pipeline."""
    return cross_validate(pipeline, X, y, cv=folds(), scoring=SCORING, return_train_score=True,
                          n_jobs=N_JOBS, error_score="raise", params=fit_params)


def summarize(scores):
    """Mean and SD over folds for every metric, plus mean training scores and fit time."""
    result = {"fit_seconds": float(np.mean(scores["fit_time"])), "fit_seconds_std": float(np.std(scores["fit_time"]))}
    for metric in SCORING:
        result[metric + "_mean"] = float(np.mean(scores["test_" + metric]))
        result[metric + "_std"] = float(np.std(scores["test_" + metric]))
        result["train_" + metric + "_mean"] = float(np.mean(scores["train_" + metric]))
    return result


def fold_frame(scores, label):
    """One row per fold with every validation metric, for paired comparisons later."""
    records = []
    for index in range(len(scores["fit_time"])):
        row = {"configuration": label, "fold": index + 1, "fit_seconds": scores["fit_time"][index]}
        row.update({metric: scores["test_" + metric][index] for metric in SCORING})
        records.append(row)
    return pd.DataFrame(records)
