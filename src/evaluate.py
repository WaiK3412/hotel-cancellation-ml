"""Phase 4: ensemble design on training CV, then the one final evaluation on the test set.

Order matters. The ensemble pair and the recommendation rule are fixed from training CV first;
only then is the test set loaded. Nothing measured on the test set changes any choice.
"""
import hashlib
from itertools import combinations
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_predict, train_test_split

from src.common import MODEL_COLOURS, ROOT, SEED, figure, load_json, log, save_figure, save_json, table
from src.data import CATEGORICAL, load_audit_columns, load_partition
from src.diagnostics import paired_ap_comparisons
from src.inference import paired_comparisons
from src.metrics import N_JOBS, cv_scores, fold_frame, folds, summarize
from src.models import baseline_models, default_pipelines, tuned_pipelines
from src.preprocess import build_pipeline, selected_options

THRESHOLD = 0.5
NOISE_FEATURES = ["lead_time", "adr", "stays_in_week_nights", "stays_in_weekend_nights"]
NOISE_LEVELS = [0.05, 0.1, 0.25, 0.5]
SEEDS = [0, 1, 2, 3, 4]


def slug(name):
    return name.lower().replace(" ", "_")


def performance(y, probability):
    """The seven report metrics at the fixed 0.5 cutoff (ties go to 'not cancelled')."""
    predicted = (np.asarray(probability) > THRESHOLD).astype(int)
    return {"accuracy": accuracy_score(y, predicted), "balanced_accuracy": balanced_accuracy_score(y, predicted),
            "precision": precision_score(y, predicted, zero_division=0),
            "recall": recall_score(y, predicted, zero_division=0), "f1": f1_score(y, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y, probability), "pr_auc": average_precision_score(y, probability)}


def ensembles(pair, models):
    """Soft voting averages the pair's probabilities; stacking learns the weights out-of-fold."""
    members = [("first", clone(models[pair[0]])), ("second", clone(models[pair[1]]))]
    voting = VotingClassifier(estimators=members, voting="soft", n_jobs=1)
    stacking = StackingClassifier(estimators=members, final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
                                  stack_method="predict_proba", cv=folds(), n_jobs=1, passthrough=False)
    return voting, stacking


def train_ensemble_design(X, y):
    """Choose the ensemble pair and the recommendation from training CV only."""
    models = tuned_pipelines()
    ranking = pd.read_csv(ROOT / "results/phase3_tuning.csv").sort_values("tuned_pr_auc", ascending=False)
    pair = ranking.model.iloc[:2].tolist()
    table(ranking[["model", "tuned_pr_auc", "pr_auc_std"]].assign(selected=lambda f: f.model.isin(pair)),
          "phase4_base_model_selection.csv")

    oof = pd.DataFrame(index=X.index)
    for name, pipeline in models.items():
        oof[name] = cross_val_predict(pipeline, X, y, cv=folds(), method="predict_proba", n_jobs=N_JOBS)[:, 1]
        log("Phase 4", "out-of-fold", f"{name}: OOF AP {average_precision_score(y, oof[name]):.3f}")
    table(oof.reset_index().assign(is_canceled=y.to_numpy()), "phase4_oof_predictions.csv")
    errors = oof.gt(THRESHOLD).ne(y, axis=0).astype(int)
    diversity = []
    for first, second in combinations(models, 2):
        diversity.append({"first": first, "second": second,
                          "probability_correlation": oof[first].corr(oof[second]),
                          "error_correlation": errors[first].corr(errors[second]),
                          "disagreement": float((oof[first].gt(THRESHOLD) != oof[second].gt(THRESHOLD)).mean()),
                          "both_wrong": float(((errors[first] == 1) & (errors[second] == 1)).mean()),
                          "selected_pair": {first, second} == set(pair)})
    diversity = table(pd.DataFrame(diversity), "phase4_pair_diversity.csv")
    diversity_figure(oof, y)

    voting, stacking = ensembles(pair, models)
    rows = [{"model": name, **{key: ranking.set_index("model").loc[name, key]
                               for key in ranking.columns if key.endswith(("_mean", "_std")) or key == "fit_seconds"}}
            for name in ranking.model]
    fold_tables = [pd.read_csv(ROOT / "results/phase3_tuned_folds.csv")]
    for name, model in [("Voting", voting), ("Stacking", stacking)]:
        scores = cv_scores(model, X, y)
        rows.append({"model": name, **summarize(scores)})
        fold_tables.append(fold_frame(scores, name)[["configuration", "fold", "pr_auc"]])
        log("Phase 4", "ensemble CV", f"{name}: AP {rows[-1]['pr_auc_mean']:.3f}")
    cv_table = table(pd.DataFrame(rows), "phase4_model_cv.csv")
    table(pd.concat(fold_tables, ignore_index=True), "phase4_cv_folds.csv")

    # Prefer the simplest model whose CV AP is within 0.001 of the best (decided before the test set).
    complexity = {name: 1 for name in models} | {"Voting": 2, "Stacking": 3}
    candidates = cv_table.assign(complexity=cv_table.model.map(complexity))
    eligible = candidates[candidates.pr_auc_mean >= candidates.pr_auc_mean.max() - 0.001]
    chosen = eligible.sort_values(["complexity", "pr_auc_mean"], ascending=[True, False]).model.iloc[0]
    save_json({"pair": pair, "excluded": ranking.model.iloc[2], "recommended": chosen,
               "rule": "simplest model within 0.001 of the best training-CV AP; ties favour higher AP",
               "primary_metric": "average_precision", "threshold": THRESHOLD, "test_used_for_selection": False},
              "results/phase4_ensemble_selection.json")
    log("Phase 4", "recommendation fixed", f"pair {pair}; recommended {chosen} (before test access).")
    return models, voting, stacking, cv_table, chosen


def fit_and_score(name, estimator, X, y, Xtest, ytest):
    start = perf_counter()
    estimator.fit(X, y)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    probability = estimator.predict_proba(Xtest)[:, 1]
    predict_seconds = perf_counter() - start
    return estimator, probability, fit_seconds, predict_seconds


def perturb(Xtest, Xtrain, level, rng):
    """Copy of the test rows with Gaussian noise of `level` x training SD on measured quantities."""
    noisy = Xtest.copy()
    for column in NOISE_FEATURES:
        sigma = level * Xtrain[column].std()
        noisy[column] = (noisy[column] + rng.normal(0, sigma, len(noisy))).clip(lower=0)
        if column != "adr":
            noisy[column] = noisy[column].round()
    return noisy


SHORT_NAMES = {"Logistic Regression": "LR", "Random Forest": "RF", "Hist Gradient Boosting": "HGB"}


def diversity_figure(oof, y):
    """Correlation of the three models' out-of-fold probabilities and of their 0.5-cutoff errors."""
    errors = oof.gt(THRESHOLD).ne(y, axis=0).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for axis, matrix, title in [(axes[0], oof.corr(), "Probability correlation"),
                                (axes[1], errors.corr(), "Error correlation (0.5 cutoff)")]:
        matrix = matrix.rename(index=SHORT_NAMES, columns=SHORT_NAMES)
        sns.heatmap(matrix, annot=True, fmt=".3f", vmin=0.5, vmax=1, cmap="viridis", square=True,
                    cbar_kws={"shrink": 0.8}, ax=axis)
        axis.grid(False)
        axis.tick_params(axis="both", labelrotation=0)
        axis.set(title=title)
    fig.text(0.5, -0.02, "LR = logistic regression, RF = random forest, HGB = histogram gradient boosting",
             ha="center", fontsize=9)
    save_figure(fig, "fig_G_01_oof_diversity.png", "How similar the three models' out-of-fold predictions are")


def curves_figure(ytest, predictions, names):
    # Voting and stacking almost coincide, so stacking is dotted and voting is drawn last and thicker;
    # the legend gives each model's AP or ROC-AUC so overlapping lines can still be told apart.
    style = {"Voting": {"linewidth": 2.6, "zorder": 5}, "Stacking": {"linestyle": ":", "linewidth": 1.6}}
    order = [name for name in names if name != "Voting"] + [name for name in names if name == "Voting"]
    for kind in ["roc", "pr"]:
        plt.figure(figsize=(7, 5))
        for name in order:
            p = predictions[name]
            if kind == "roc":
                x_values, y_values, _ = roc_curve(ytest, p)
                label = f"{name} (ROC-AUC {roc_auc_score(ytest, p):.3f})"
            else:
                y_values, x_values, _ = precision_recall_curve(ytest, p)
                label = f"{name} (AP {average_precision_score(ytest, p):.3f})"
            plt.plot(x_values, y_values, label=label, color=MODEL_COLOURS.get(name), **style.get(name, {}))
        if kind == "roc":
            plt.plot([0, 1], [0, 1], "--", color="grey", label="chance")
            labels = ("False positive rate", "True positive rate")
        else:
            plt.axhline(ytest.mean(), linestyle="--", color="grey", label=f"class rate {ytest.mean():.3f}")
            labels = ("Recall", "Precision")
        plt.legend(fontsize=8)
        figure(f"fig_H_0{2 if kind == 'roc' else 3}_{kind}_curves.png",
               f"Test set: {'ROC' if kind == 'roc' else 'precision-recall'} curves", *labels)


def error_analysis(name, probability, Xtest, ytest):
    """Where the recommended model misses: recall, precision and errors by booking segment."""
    audit = load_audit_columns("test", ["country", "deposit_type"]).loc[Xtest.index]
    predicted = (probability > THRESHOLD).astype(int)
    bands = pd.cut(Xtest.lead_time, [-1, 7, 30, 90, 180, 365, Xtest.lead_time.max()],
                   labels=["0-7", "8-30", "31-90", "91-180", "181-365", ">365"])
    segments = {"lead time (days)": bands.astype(str), "hotel": Xtest.hotel, "market segment": Xtest.market_segment,
                "customer type": Xtest.customer_type, "deposit type (not a model input)": audit.deposit_type}
    rows = []
    for field, values in segments.items():
        for group in pd.unique(values.dropna()):
            mask = (values == group).to_numpy()
            ys, ps, pred = ytest.to_numpy()[mask], probability[mask], predicted[mask]
            if mask.sum() < 50:
                continue
            rows.append({"field": field, "group": group, "rows": int(mask.sum()), "cancel_rate": ys.mean(),
                         "recall": recall_score(ys, pred, zero_division=0),
                         "precision": precision_score(ys, pred, zero_division=0),
                         "false_negatives": int(((ys == 1) & (pred == 0)).sum()),
                         "false_positives": int(((ys == 0) & (pred == 1)).sum()),
                         "pr_auc": average_precision_score(ys, ps) if 0 < ys.sum() < len(ys) else np.nan})
    errors = table(pd.DataFrame(rows), "phase4_error_analysis.csv")
    order = ["0-7", "8-30", "31-90", "91-180", "181-365", ">365"]
    lead = errors[errors.field == "lead time (days)"].set_index("group").reindex(order).dropna(subset=["rows"])
    x = np.arange(len(lead))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, lead.recall, 0.4, label="recall (cancellations caught)", color="#D55E00")
    plt.bar(x + 0.2, lead.precision, 0.4, label="precision (warnings that were right)", color="#0072B2")
    plt.plot(x, lead.cancel_rate, "k--o", label="cancellation rate")
    plt.xticks(x, lead.index)
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    figure("fig_H_07_errors_by_lead_time.png", f"{name}: where it misses cancellations (test set)",
           "Lead time band (days between booking and arrival)", "Share")
    outcome = np.select([(ytest == 1) & (predicted == 1), (ytest == 1) & (predicted == 0),
                         (ytest == 0) & (predicted == 1)], ["caught cancellation", "missed cancellation",
                                                            "false alarm"], "correct non-cancellation")
    profile = Xtest.assign(outcome=outcome, repeated_guest=Xtest.is_repeated_guest,
                           prior_cancellation=(Xtest.previous_cancellations > 0).astype(int),
                           online_ta=(Xtest.market_segment == "Online TA").astype(int),
                           groups=(Xtest.market_segment == "Groups").astype(int),
                           city_hotel=(Xtest.hotel == "City Hotel").astype(int))
    summary = profile.groupby("outcome").agg(rows=("lead_time", "size"), median_lead_time=("lead_time", "median"),
                                             median_adr=("adr", "median"), mean_requests=("total_of_special_requests", "mean"),
                                             share_prior_cancellation=("prior_cancellation", "mean"),
                                             share_online_ta=("online_ta", "mean"), share_groups=("groups", "mean"),
                                             share_city_hotel=("city_hotel", "mean")).reset_index()
    table(summary, "phase4_error_profile.csv")
    return errors


def fairness(name, probability, Xtest, ytest):
    """Error rates by country group. Country is NOT a model input; it is kept only for this audit."""
    country = load_audit_columns("test", ["country"]).loc[Xtest.index, "country"]
    top = load_audit_columns("train", ["country"]).country.value_counts().index[:5]
    group = country.where(country.isin(top), "Other").fillna("Missing")
    predicted = (probability > THRESHOLD).astype(int)
    rows = []
    for value in list(top) + ["Other", "Missing"]:
        mask = (group == value).to_numpy()
        if mask.sum() < 30:
            continue
        ys, pred = ytest.to_numpy()[mask], predicted[mask]
        negatives = (ys == 0).sum()
        rows.append({"country_group": value, "rows": int(mask.sum()), "cancel_rate": ys.mean(),
                     "flagged_rate": pred.mean(), "recall": recall_score(ys, pred, zero_division=0),
                     "false_positive_rate": ((pred == 1) & (ys == 0)).sum() / negatives if negatives else np.nan,
                     "precision": precision_score(ys, pred, zero_division=0),
                     "pr_auc": average_precision_score(ys, probability[mask]) if 0 < ys.sum() < len(ys) else np.nan})
    frame = table(pd.DataFrame(rows), "phase4_fairness_country.csv")
    fairness_figure(frame, name)
    return frame


def fairness_figure(frame, name):
    """Actual cancellation rate, recall and false-positive rate per country group, with group sizes."""
    x = np.arange(len(frame))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.27, frame.cancel_rate, 0.27, label="actual cancellation rate", color="#999999")
    plt.bar(x, frame.recall, 0.27, label="recall (cancellations caught)", color="#D55E00")
    plt.bar(x + 0.27, frame.false_positive_rate, 0.27, label="false positive rate", color="#0072B2")
    plt.xticks(x, [f"{group}\n(n = {rows:,})" for group, rows in zip(frame.country_group, frame.rows)])
    plt.ylim(0, 1)
    plt.legend(fontsize=8)
    figure("fig_H_08_fairness_country.png", f"{name}: error rates by guest country (test set; country not used)",
           "Country group (top five training countries, then the rest; n = test bookings)", "Rate")


def confusion_figure(matrices, shown=("Logistic Regression", "Voting")):
    """Test-set confusion matrices at 0.5 for the transparent baseline and the recommended model.

    The other models' counts are in results/phase4_confusion_matrices.csv and the report's metric table.
    """
    matrices = [item for name in shown for item in matrices if item["model"] == name]
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.2 * len(matrices), 4.2))
    for axis, item in zip(np.atleast_1d(axes), matrices):
        matrix = np.array([[item["tn"], item["fp"]], [item["fn"], item["tp"]]]).astype(int)
        sns.heatmap(matrix, annot=True, fmt="d", cbar=False, cmap="Blues", ax=axis, annot_kws={"fontsize": 12},
                    xticklabels=["not cancelled", "cancelled"], yticklabels=["not cancelled", "cancelled"])
        axis.set(title=item["model"], xlabel="Predicted", ylabel="Actual")
        axis.grid(False)
    save_figure(fig, "fig_H_01_confusion_matrices.png", "Test-set confusion matrices at the 0.5 cutoff")


def final_evaluation(models, voting, stacking, cv_table, chosen, X, y):
    """Fit every model on all training rows; score the test set once."""
    Xtest, ytest = load_partition("test")
    log("Phase 4", "test set opened", f"{len(Xtest)} rows; every choice is already fixed.")
    table(pd.DataFrame([{"test_rows": len(Xtest), "positives": int(ytest.sum()), "positive_fraction": ytest.mean(),
                         "training_rows": len(X)}]), "phase4_test_characteristics.csv")
    candidates = {}
    for name, model in baseline_models().items():
        candidates[name] = build_pipeline(model, **selected_options())
    candidates.update({name: clone(pipeline) for name, pipeline in models.items()})
    candidates.update({"Voting": clone(voting), "Stacking": clone(stacking)})

    predictions = pd.DataFrame({"source_row": Xtest.index, "is_canceled": ytest.to_numpy()})
    fitted, metrics, train_metrics, overfit, efficiency, matrices = {}, [], [], [], [], []
    cv = cv_table.set_index("model")
    for name, estimator in candidates.items():
        estimator, probability, fit_seconds, predict_seconds = fit_and_score(name, estimator, X, y, Xtest, ytest)
        fitted[name] = estimator
        predictions["p_" + slug(name)] = probability
        metrics.append({"model": name, **performance(ytest, probability)})
        train = performance(y, estimator.predict_proba(X)[:, 1])
        train_metrics.append({"model": name, **train})
        if name in cv.index:
            overfit.append({"model": name, "train_pr_auc": train["pr_auc"], "cv_pr_auc": cv.loc[name, "pr_auc_mean"],
                            "cv_pr_auc_std": cv.loc[name, "pr_auc_std"], "test_pr_auc": metrics[-1]["pr_auc"],
                            "train_minus_cv": train["pr_auc"] - cv.loc[name, "pr_auc_mean"],
                            "cv_minus_test": cv.loc[name, "pr_auc_mean"] - metrics[-1]["pr_auc"]})
        efficiency.append({"model": name, "fit_seconds": fit_seconds,
                           "predict_seconds_per_1000": predict_seconds / len(Xtest) * 1000})
        tn, fp, fn, tp = confusion_matrix(ytest, (probability > THRESHOLD).astype(int), labels=[0, 1]).ravel()
        matrices.append({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
        log("Phase 4", "test result", f"{name}: AP {metrics[-1]['pr_auc']:.3f}, F1 {metrics[-1]['f1']:.3f}")
    metrics = table(pd.DataFrame(metrics), "phase4_test_metrics.csv")
    table(pd.DataFrame(train_metrics), "phase4_train_metrics.csv")
    table(pd.DataFrame(overfit), "phase4_overfitting.csv")
    table(predictions, "phase4_test_predictions.csv")
    table(pd.DataFrame(matrices), "phase4_confusion_matrices.csv")

    # Deployment cost: compressed model size on disk.
    sizes = []
    for name, estimator in fitted.items():
        path = ROOT / "work" / "models" / f"{slug(name)}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, path, compress=3)
        sizes.append(path.stat().st_size / 1024)
    efficiency = table(pd.DataFrame(efficiency).assign(model_size_kib=sizes), "phase4_efficiency.csv")

    shown = [name for name in candidates if "baseline" not in name]
    confusion_figure(matrices, ("Logistic Regression", chosen))
    curves_figure(ytest, {name: predictions["p_" + slug(name)].to_numpy() for name in shown}, shown)

    tuning_check(X, y, Xtest, ytest, predictions)
    importance(fitted, chosen, Xtest, ytest)
    robustness(fitted, X, Xtest, ytest)
    seed_variance(models, voting, X, y, Xtest, ytest)
    errors = error_analysis(chosen, predictions["p_" + slug(chosen)].to_numpy(), Xtest, ytest)
    fairness_table = fairness(chosen, predictions["p_" + slug(chosen)].to_numpy(), Xtest, ytest)
    save_recommended(fitted[chosen], chosen, X, Xtest)
    return metrics, efficiency, errors, fairness_table


def tuning_check(X, y, Xtest, ytest, predictions):
    """Default vs tuned on the untouched test set: the unbiased check on tuning (paired bootstrap)."""
    probabilities, rows = {}, []
    for name, pipeline in default_pipelines().items():
        probabilities[f"{name} (default)"] = clone(pipeline).fit(X, y).predict_proba(Xtest)[:, 1]
        probabilities[f"{name} (tuned)"] = predictions["p_" + slug(name)].to_numpy()
    for name in default_pipelines():
        pair = {k: v for k, v in probabilities.items() if k.startswith(name)}
        comparison = paired_ap_comparisons(ytest.to_numpy(), pair, f"{name} (default)", repeats=1000)
        rows.append(comparison.assign(model=name))
        log("Phase 4", "tuning check", f"{name}: tuned - default test AP {comparison['observed'].iloc[0]:+.4f}")
    table(pd.concat(rows, ignore_index=True), "phase4_tuning_test_check.csv")
    table(pd.DataFrame([{"model": k, **performance(ytest, v)} for k, v in probabilities.items()]),
          "phase4_default_vs_tuned_test.csv")


def importance(fitted, chosen, Xtest, ytest):
    """Permutation importance on a fixed 2,000-row stratified sample of the test set."""
    _, rows = train_test_split(np.arange(len(Xtest)), test_size=2000, stratify=ytest, random_state=SEED)
    frames = []
    for name in ["Logistic Regression", "Random Forest", "Hist Gradient Boosting", "Voting"]:
        result = permutation_importance(fitted[name], Xtest.iloc[rows], ytest.iloc[rows], scoring="average_precision",
                                        n_repeats=5, random_state=SEED, n_jobs=N_JOBS)
        frames.append(pd.DataFrame({"model": name, "feature": Xtest.columns, "importance_mean": result.importances_mean,
                                    "importance_std": result.importances_std}))
    frame = table(pd.concat(frames, ignore_index=True), "phase4_permutation_importance.csv")
    top = frame[frame.model == chosen].sort_values("importance_mean").tail(12)
    plt.figure(figsize=(8, 5))
    plt.barh(top.feature, top.importance_mean, xerr=top.importance_std, color=MODEL_COLOURS.get(chosen))
    figure("fig_H_04_importance.png", f"{chosen}: permutation importance (2,000 test rows, 5 repeats)",
           "Drop in average precision when the field is shuffled", "Field")
    coefficients(fitted["Logistic Regression"])


def coefficients(pipeline):
    names = pipeline.named_steps["prepare"].get_feature_names_out()
    names = names[pipeline.named_steps["variance"].get_support()]
    if pipeline.named_steps["select"] != "passthrough":
        names = names[pipeline.named_steps["select"].get_support()]
    values = pipeline.named_steps["model"].coef_.ravel()
    frame = table(pd.DataFrame({"transformed_feature": names, "coefficient": values,
                                "absolute_coefficient": np.abs(values)}).sort_values("absolute_coefficient",
                                                                                     ascending=False),
                  "phase4_linear_coefficients.csv")
    top = frame.head(15).sort_values("coefficient")
    plt.figure(figsize=(8, 5.5))
    plt.barh(top.transformed_feature, top.coefficient,
             color=np.where(top.coefficient > 0, "#D55E00", "#0072B2"))
    figure("fig_H_05_linear_coefficients.png", "Logistic regression: largest coefficients (scaled inputs)",
           "Change in log-odds of cancellation per unit", "Encoded field")


def robustness(fitted, X, Xtest, ytest):
    """Test AP when measured quantities carry increasing Gaussian noise (5% to 50% of their SD)."""
    rows = []
    for level in [0.0] + NOISE_LEVELS:
        noisy = Xtest if level == 0 else perturb(Xtest, X, level, np.random.default_rng(SEED))
        for name in ["Logistic Regression", "Random Forest", "Hist Gradient Boosting", "Voting"]:
            rows.append({"model": name, "noise_fraction_of_sd": level,
                         "pr_auc": average_precision_score(ytest, fitted[name].predict_proba(noisy)[:, 1])})
    frame = table(pd.DataFrame(rows), "phase4_robustness.csv")
    plt.figure(figsize=(7, 4.5))
    for name, group in frame.groupby("model", sort=False):
        plt.plot(group.noise_fraction_of_sd, group.pr_auc, marker="o", label=name, color=MODEL_COLOURS[name])
    plt.legend(fontsize=8)
    figure("fig_H_06_noise_robustness.png", "Test AP as noise is added to lead time, ADR and nights",
           "Noise standard deviation (fraction of the field's training SD)", "Average precision")
    log("Phase 4", "robustness", "Noise levels " + ", ".join(map(str, NOISE_LEVELS)) + " evaluated.")


def seed_variance(models, voting, X, y, Xtest, ytest):
    """How much test AP moves when only the random seed changes (same data, same parameters)."""
    rows = []
    for seed in SEEDS:
        for name in ["Random Forest", "Hist Gradient Boosting", "Voting"]:
            estimator = clone(voting if name == "Voting" else models[name])
            estimator.set_params(**{key: seed for key in estimator.get_params() if key.endswith("random_state")})
            probability = estimator.fit(X, y).predict_proba(Xtest)[:, 1]
            rows.append({"model": name, "seed": seed, "pr_auc": average_precision_score(ytest, probability),
                         "f1": f1_score(ytest, probability > THRESHOLD)})
        log("Phase 4", "seed variance", f"seed {seed} done")
    frame = table(pd.DataFrame(rows), "phase4_seed_variance.csv")
    table(frame.groupby("model").agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                                     pr_auc_min=("pr_auc", "min"), pr_auc_max=("pr_auc", "max"),
                                     f1_mean=("f1", "mean"), f1_sd=("f1", "std")).reset_index(),
          "phase4_seed_variance_summary.csv")


def alternative_pairs():
    """Soft voting of every pair of tuned models on the shared training folds; the test set is not loaded.

    Added after the final evaluation to test the reason for leaving logistic regression out of the
    ensemble. It reports evidence only: the recommendation was fixed before the test set was opened.
    """
    X, y = load_partition("train")
    models = tuned_pipelines()
    selected = set(load_json("results/phase4_ensemble_selection.json")["pair"])
    rows, fold_tables = [], []
    for pair in combinations(models, 2):
        voting, _ = ensembles(pair, models)
        label = " + ".join(pair)
        scores = cv_scores(voting, X, y)
        summary = summarize(scores)
        rows.append({"pair": label, "selected_pair": set(pair) == selected, "pr_auc_mean": summary["pr_auc_mean"],
                     "pr_auc_std": summary["pr_auc_std"], "train_pr_auc_mean": summary["train_pr_auc_mean"],
                     "fit_seconds": summary["fit_seconds"]})
        fold_tables.append(fold_frame(scores, label))
        log("Alternative pairs", "alternative pair", f"Voting of {label}: CV AP {summary['pr_auc_mean']:.3f}")
    summary = table(pd.DataFrame(rows), "d2_7_pair_cv.csv")
    fold_results = table(pd.concat(fold_tables, ignore_index=True), "d2_7_pair_cv_folds.csv")
    reference = summary.loc[summary.selected_pair, "pair"].iloc[0]
    comparisons = table(paired_comparisons(fold_results, reference), "d2_7_pair_comparisons.csv")
    return summary, comparisons


def booking_time_features():
    """Extra features: calendar features and target encoding for agent/company, on the training folds only.

    Run after the final evaluation, because only four engineered fields had been tried. The test
    set is not loaded, and the recommended model is not changed: choosing new
    features now, after the test set has been used, would make the test score optimistic.
    """
    from src.preprocess import extended_pipeline

    X, y = load_partition("train")
    X = X.join(load_audit_columns("train", ["arrival_date_year"]))  # needed only to find the weekday
    best = load_json("results/phase3_best_params.json")
    families = {name: model for name, model in default_pipelines().items() if name != "Logistic Regression"}
    options = {"reference": {}, "calendar features": {"calendar": True},
               "target-encoded agent and company": {"target_encode": True},
               "both": {"calendar": True, "target_encode": True}}
    rows, comparisons = [], []
    for name, default in families.items():
        model = default.named_steps["model"]
        fold_tables = []
        for label, option in options.items():
            pipeline = extended_pipeline(model, **option).set_params(**best[name])
            scores = cv_scores(pipeline, X, y)
            summary = summarize(scores)
            rows.append({"model": name, "configuration": label, "pr_auc_mean": summary["pr_auc_mean"],
                         "pr_auc_std": summary["pr_auc_std"], "roc_auc_mean": summary["roc_auc_mean"],
                         "fit_seconds": summary["fit_seconds"]})
            fold_tables.append(fold_frame(scores, label))
            log("Extra features", "booking-time features", f"{name}, {label}: CV AP {summary['pr_auc_mean']:.4f}")
        folds_frame = pd.concat(fold_tables, ignore_index=True).assign(model=name)
        comparisons.append(paired_comparisons(folds_frame, "reference").assign(model=name))
        table(folds_frame, f"d2_8_booking_time_folds_{name.split()[0].lower()}.csv")
    from src.inference import holm
    comparisons = holm(pd.concat(comparisons, ignore_index=True).drop(columns="p_holm"))  # Holm over all six
    return table(pd.DataFrame(rows), "d2_8_booking_time_cv.csv"), table(comparisons, "d2_8_booking_time_comparisons.csv")


def precision_at_capacity(shares=(0.05, 0.10, 0.20)):
    """Precision and recall when staff can review only the riskiest share of test bookings.

    This is the business success measure stated in Section 1.1. It reads the saved test predictions;
    nothing is refitted and no modelling choice depends on it.
    """
    predictions = pd.read_csv(ROOT / "results/phase4_test_predictions.csv")
    y = predictions.is_canceled.to_numpy()
    rows = []
    for name in ["Logistic Regression", "Random Forest", "Hist Gradient Boosting", "Voting", "Stacking"]:
        order = np.argsort(-predictions["p_" + slug(name)].to_numpy(), kind="stable")  # riskiest first
        for share in shares:
            reviewed = order[:int(round(share * len(y)))]
            rows.append({"model": name, "review_share": share, "bookings_reviewed": len(reviewed),
                         "precision": y[reviewed].mean(), "recall": y[reviewed].sum() / y.sum()})
    return table(pd.DataFrame(rows), "d2_9_precision_at_capacity.csv")


def save_recommended(estimator, chosen, X, Xtest):
    """Save the recommended model with an input contract and check that reloading is exact."""
    path = ROOT / "models/recommended.joblib"
    joblib.dump(estimator, path, compress=3)
    save_json({"recommended": chosen, "artifact": "models/recommended.joblib",
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "expected_input_columns": X.columns.tolist(),
               "categorical_as_text": [column for column in CATEGORICAL if column in X],
               "input_contract": "Cleaned booking rows (see results/phase1_feature_dictionary.csv). Excluded leakage "
                                 "fields and outcome fields are not needed and are ignored if present.",
               "metric_source": "results/phase4_test_metrics.csv",
               "selection_source": "results/phase4_ensemble_selection.json"}, "models/recommended_metadata.json")
    example = Xtest.iloc[:5]
    expected = estimator.predict_proba(example)[:, 1]
    assert np.allclose(joblib.load(path).predict_proba(example)[:, 1], expected, atol=0, rtol=0)
    table(pd.DataFrame({"source_row": example.index, "expected_probability": expected}), "phase4_reload_check.csv")


def run_evaluation():
    X, y = load_partition("train")
    models, voting, stacking, cv_table, chosen = train_ensemble_design(X, y)
    return final_evaluation(models, voting, stacking, cv_table, chosen, X, y)
