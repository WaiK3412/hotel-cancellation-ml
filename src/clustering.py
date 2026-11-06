"""Notebook 08: label-free booking segmentation and its stability (Part B extension).

K-Means is fitted twice: on numeric fields only, and on a mixed representation (robust-scaled,
clipped numbers plus one-hot categories, each block weighted so neither dominates). k is chosen
by silhouette without labels. Labels are used only afterwards, to measure agreement (ARI, NMI).
Stability is checked by refitting on 20 random subsamples and comparing assignments of the same
500 held-aside "anchor" bookings.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

from src.common import ROOT, SEED, figure, log, save_figure, save_json, table
from src.data import load_partition
from src.inference import holm

NUMERIC = ["lead_time", "adr", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
           "previous_cancellations", "total_of_special_requests"]
CATEGORICAL = ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type", "customer_type"]
POOL, ANCHORS, RESAMPLES, SUBSAMPLE = 4000, 500, 20, 3200


def clustering_pipeline(kind, k, numeric, categorical):
    if kind == "numeric":
        prepare = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                           ("scale", StandardScaler())]), numeric)])
    else:
        numbers = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()),
                            ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -5.0, "a_max": 5.0}))])
        categories = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                               ("encode", OneHotEncoder(min_frequency=0.02, handle_unknown="infrequent_if_exist",
                                                        sparse_output=False))])
        # Block weights make the numeric and categorical blocks contribute comparable distances.
        prepare = ColumnTransformer([("numeric", numbers, numeric), ("categorical", categories, categorical)],
                                    transformer_weights={"numeric": 1 / np.sqrt(len(numeric)),
                                                         "categorical": 1 / np.sqrt(2 * len(categorical))})
    return Pipeline([("prepare", prepare), ("cluster", KMeans(n_clusters=k, n_init=10, random_state=SEED))])


def mean_interval(values):
    values = np.asarray(values)
    margin = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": values.mean(), "sd": values.std(ddof=1), "ci_low": values.mean() - margin,
            "ci_high": values.mean() + margin, "n": len(values)}


def mixed_clustering():
    X, y = load_partition("train")
    numeric = [column for column in NUMERIC if column in X]
    categorical = [column for column in CATEGORICAL if column in X]
    # Sampling never looks at the labels.
    rows = np.random.default_rng(SEED).choice(len(X), POOL + ANCHORS, replace=False)
    pool, anchor = X.iloc[rows[:POOL]], X.iloc[rows[POOL:]]
    anchor_labels = y.loc[anchor.index].to_numpy()
    save_json({"seed": SEED, "pool_rows": POOL, "anchor_rows": ANCHORS, "resamples": RESAMPLES,
               "subsample_rows": SUBSAMPLE, "k_candidates": [2, 3, 4, 5, 6], "numeric": numeric,
               "categorical": categorical}, "results/d2_4_protocol.json")
    trials, references, profiles = [], {}, []
    for kind in ["numeric", "mixed"]:
        fitted = {}
        for k in range(2, 7):
            model = clustering_pipeline(kind, k, numeric, categorical).fit(pool)
            embedded = model[:-1].transform(pool)
            trials.append({"geometry": kind, "k": k, "inertia": model[-1].inertia_,
                           "silhouette": silhouette_score(embedded, model[-1].labels_, sample_size=1000, random_state=SEED)})
            fitted[k] = model
        frame = pd.DataFrame(trials).query("geometry == @kind")
        best_k = int(frame.loc[frame.silhouette.idxmax(), "k"])
        references[kind] = fitted[best_k]
        for cluster, group in pool.assign(cluster=fitted[best_k][-1].labels_).groupby("cluster"):
            profiles.append({"geometry": kind, "k": best_k, "cluster": cluster, "rows": len(group),
                             "cancellation_fraction": y.loc[group.index].mean(),
                             "median_lead_time": group.lead_time.median(), "median_adr": group.adr.median(),
                             "modal_hotel": group.hotel.mode().iloc[0],
                             "modal_market_segment": group.market_segment.mode().iloc[0]})
        log("D2-4", "k selected", f"{kind}: k = {best_k} (silhouette, no labels)")
    trial_frame = table(pd.DataFrame(trials), "d2_4_k_selection.csv")
    table(pd.DataFrame(profiles), "d2_4_cluster_profiles.csv")

    rng = np.random.default_rng(SEED)
    scores = []
    for resample in range(1, RESAMPLES + 1):
        subsample = rng.choice(POOL, SUBSAMPLE, replace=False)
        for kind, reference in references.items():
            refit = clustering_pipeline(kind, reference[-1].n_clusters, numeric, categorical).fit(pool.iloc[subsample])
            assigned = refit.predict(anchor)
            scores.append({"geometry": kind, "resample": resample,
                           "stability_ari": adjusted_rand_score(reference.predict(anchor), assigned),
                           "label_ari": adjusted_rand_score(anchor_labels, assigned),
                           "label_nmi": normalized_mutual_info_score(anchor_labels, assigned)})
    score_frame = table(pd.DataFrame(scores), "d2_4_resampling_scores.csv")
    metrics = ["stability_ari", "label_ari", "label_nmi"]
    summary = table(pd.DataFrame([{"geometry": kind, "metric": metric, **mean_interval(group[metric])}
                                  for kind, group in score_frame.groupby("geometry") for metric in metrics]),
                    "d2_4_summary.csv")
    comparisons = []
    for metric in metrics:
        pivot = score_frame.pivot(index="resample", columns="geometry", values=metric)
        difference = pivot.mixed - pivot.numeric
        p_value = wilcoxon(difference).pvalue if np.any(difference != 0) else 1.0
        comparisons.append({"option": "mixed", "reference": "numeric", "metric": metric, **mean_interval(difference),
                            "p_value": p_value})
    comparison = table(holm(pd.DataFrame(comparisons)), "d2_4_comparisons.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, (kind, group) in zip(axes, trial_frame.groupby("geometry", sort=False)):
        axis.plot(group.k, group.silhouette, marker="o")
        axis.set(title=f"{kind} representation", xlabel="Clusters k", ylabel="Silhouette")
    save_figure(fig, "fig_B_02_mixed_k.png", "Choosing k without labels (silhouette)")
    stability_figure(summary)
    return trial_frame, summary, comparison


# ---------------------------------------------------------------------------------------------
# D15 (13 November 2026): two clustering methods built for mixed data, on the same pool, anchors
# and subsamples as mixed_clustering(), so the four methods can be compared resample by resample.
# Labels are used only afterwards, to measure agreement, exactly as above.
# ---------------------------------------------------------------------------------------------
GAMMA = 0.5  # weight of one category mismatch; Huang's rule: about half the SD of the scaled numbers


def mixed_arrays(frame, numeric, categorical, medians):
    """Numbers (missing values filled with the pool medians) and categories (missing -> "Missing")."""
    numbers = frame[numeric].fillna(medians).to_numpy(dtype=float)
    categories = frame[categorical].fillna("Missing").astype(str).to_numpy()
    return numbers, categories


def column_modes(categories):
    """Most frequent value of each column (ties go to the alphabetically first value)."""
    modes = []
    for column in categories.T:
        values, counts = np.unique(column, return_counts=True)
        modes.append(values[counts.argmax()])
    return np.array(modes, dtype=object)


def kprototypes_cost(numbers, categories, centres, modes, gamma=GAMMA):
    """Rows x clusters: squared distance on the numbers plus gamma times the number of category mismatches."""
    numeric_part = ((numbers[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    category_part = (categories[:, None, :] != modes[None, :, :]).sum(axis=2)
    return numeric_part + gamma * category_part


class KPrototypes:
    """k-prototypes (Huang): K-means for the numbers and K-modes for the categories, in one cost.

    Each cluster has a prototype: the mean of its members' numbers and the mode of each category.
    Rows are assigned to the cheapest prototype, prototypes are recomputed, and this repeats until
    no row moves. Several random starts are tried and the cheapest solution is kept.
    """

    def __init__(self, k, gamma=GAMMA, n_init=5, max_iter=100, random_state=SEED):
        self.k, self.gamma, self.n_init, self.max_iter, self.random_state = k, gamma, n_init, max_iter, random_state

    def fit(self, numbers, categories):
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            start = rng.choice(len(numbers), self.k, replace=False)
            centres, modes, labels = numbers[start].copy(), categories[start].copy(), None
            for _ in range(self.max_iter):
                new = kprototypes_cost(numbers, categories, centres, modes, self.gamma).argmin(axis=1)
                if labels is not None and np.array_equal(new, labels):
                    break
                labels = new
                for cluster in range(self.k):
                    members = labels == cluster
                    if members.any():  # an empty cluster keeps its old prototype
                        centres[cluster] = numbers[members].mean(axis=0)
                        modes[cluster] = column_modes(categories[members])
            cost = kprototypes_cost(numbers, categories, centres, modes, self.gamma)[np.arange(len(labels)), labels].sum()
            if best is None or cost < best[0]:
                best = (cost, centres, modes, labels)
        self.cost_, self.centres_, self.modes_, self.labels_ = best
        return self

    def predict(self, numbers, categories):
        return kprototypes_cost(numbers, categories, self.centres_, self.modes_, self.gamma).argmin(axis=1)


def gower_distances(numbers_a, categories_a, numbers_b, categories_b, ranges):
    """Gower distance: each field contributes a value from 0 to 1 and the contributions are averaged.

    Numbers contribute |difference| / range; categories contribute 1 when they differ, else 0.
    """
    total = np.zeros((len(numbers_a), len(numbers_b)), dtype=np.float32)
    for j in range(numbers_a.shape[1]):
        total += (np.abs(numbers_a[:, [j]] - numbers_b[None, :, j]) / ranges[j]).astype(np.float32)
    for j in range(categories_a.shape[1]):
        total += (categories_a[:, [j]] != categories_b[None, :, j])
    return total / (numbers_a.shape[1] + categories_a.shape[1])


def k_medoids(distances, k, random_state=SEED, n_init=5, max_iter=100):
    """Alternating k-medoids on a precomputed distance matrix; returns (medoid positions, labels).

    A medoid is the member with the smallest total distance to the rest of its cluster, so every
    cluster centre is a real booking (unlike a mean, which needs numbers).
    """
    rng = np.random.default_rng(random_state)
    best = None
    for _ in range(n_init):
        medoids = rng.choice(len(distances), k, replace=False)
        for _ in range(max_iter):
            labels = distances[:, medoids].argmin(axis=1)
            new = medoids.copy()
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                if len(members):
                    new[cluster] = members[distances[np.ix_(members, members)].sum(axis=1).argmin()]
            if np.array_equal(new, medoids):
                break
            medoids = new
        labels = distances[:, medoids].argmin(axis=1)
        cost = distances[np.arange(len(distances)), medoids[labels]].sum()
        if best is None or cost < best[0]:
            best = (cost, medoids, labels)
    return best[1], best[2]


def mixed_data_clustering():
    """D15: k-prototypes and Gower k-medoids, compared with the two K-means versions above."""
    X, y = load_partition("train")
    numeric = [column for column in NUMERIC if column in X]
    categorical = [column for column in CATEGORICAL if column in X]
    rows = np.random.default_rng(SEED).choice(len(X), POOL + ANCHORS, replace=False)  # same draw as mixed_clustering()
    pool, anchor = X.iloc[rows[:POOL]], X.iloc[rows[POOL:]]
    anchor_labels = y.loc[anchor.index].to_numpy()
    # Label-free settings learned once on the pool: medians, scaling and Gower ranges.
    medians = pool[numeric].median()
    pool_numbers, pool_categories = mixed_arrays(pool, numeric, categorical, medians)
    anchor_numbers, anchor_categories = mixed_arrays(anchor, numeric, categorical, medians)
    mean, sd = pool_numbers.mean(axis=0), pool_numbers.std(axis=0)
    # Standardise, then clip at 5 SD so a rate of 5,400 cannot dominate the squared distances.
    scale = lambda values: np.clip((values - mean) / sd, -5, 5)
    pool_scaled, anchor_scaled = scale(pool_numbers), scale(anchor_numbers)
    ranges = pool_numbers.max(axis=0) - pool_numbers.min(axis=0)
    pool_distances = gower_distances(pool_numbers, pool_categories, pool_numbers, pool_categories, ranges)
    anchor_distances = gower_distances(anchor_numbers, anchor_categories, pool_numbers, pool_categories, ranges)

    trials, references = [], {}
    for k in range(2, 7):
        # Silhouette on Gower distances for both methods, so their k choices use the same yardstick.
        prototypes = KPrototypes(k).fit(pool_scaled, pool_categories)
        medoids, labels = k_medoids(pool_distances, k)
        for method, assigned, model in [("k-prototypes", prototypes.labels_, prototypes),
                                        ("Gower k-medoids", labels, medoids)]:
            silhouette = silhouette_score(pool_distances, assigned, metric="precomputed", sample_size=1000,
                                          random_state=SEED) if len(set(assigned)) > 1 else np.nan
            trials.append({"method": method, "k": k, "silhouette_gower": silhouette,
                           "smallest_cluster": int(np.bincount(assigned).min())})
            references.setdefault(method, {})[k] = model
    trial_frame = table(pd.DataFrame(trials), "d2_11_mixed_methods_k_selection.csv")
    best_k = {method: int(group.loc[group.silhouette_gower.idxmax(), "k"]) for method, group in trial_frame.groupby("method")}
    chosen = {method: references[method][k] for method, k in best_k.items()}
    for method, k in best_k.items():
        log("D15", "k selected", f"{method}: k = {k} (silhouette on Gower distance, no labels)")

    def assign(method, model, numbers, categories, distances_to_pool, subset=None):
        if method == "k-prototypes":
            return model.predict(numbers, categories)
        positions = model if subset is None else subset[model]
        return distances_to_pool[:, positions].argmin(axis=1)

    reference_anchor = {method: assign(method, model, anchor_scaled, anchor_categories, anchor_distances)
                        for method, model in chosen.items()}
    rng = np.random.default_rng(SEED)  # same subsamples as mixed_clustering(), resample by resample
    scores = []
    for resample in range(1, RESAMPLES + 1):
        subsample = rng.choice(POOL, SUBSAMPLE, replace=False)
        for method, k in best_k.items():
            if method == "k-prototypes":
                model = KPrototypes(k).fit(pool_scaled[subsample], pool_categories[subsample])
                assigned = assign(method, model, anchor_scaled, anchor_categories, None)
            else:
                medoids, _ = k_medoids(pool_distances[np.ix_(subsample, subsample)], k)
                assigned = assign(method, medoids, None, None, anchor_distances, subset=subsample)
            scores.append({"geometry": method, "resample": resample,
                           "stability_ari": adjusted_rand_score(reference_anchor[method], assigned),
                           "label_ari": adjusted_rand_score(anchor_labels, assigned),
                           "label_nmi": normalized_mutual_info_score(anchor_labels, assigned)})
    score_frame = pd.DataFrame(scores)
    kmeans = pd.read_csv(ROOT / "results/d2_4_resampling_scores.csv").replace(
        {"geometry": {"numeric": "K-means numeric", "mixed": "K-means mixed"}})
    all_scores = table(pd.concat([kmeans, score_frame], ignore_index=True), "d2_11_mixed_methods_scores.csv")
    metrics = ["stability_ari", "label_ari", "label_nmi"]
    summary = table(pd.DataFrame([{"geometry": method, "metric": metric, **mean_interval(group[metric])}
                                  for method, group in all_scores.groupby("geometry", sort=False) for metric in metrics]),
                    "d2_11_mixed_methods_summary.csv")
    # Does a method built for mixed data agree with cancellations more than K-means on mixed fields?
    comparisons = []
    pivot = all_scores.pivot(index="resample", columns="geometry", values="label_ari")
    for method in best_k:
        difference = pivot[method] - pivot["K-means mixed"]
        comparisons.append({"option": method, "reference": "K-means mixed", "metric": "label_ari",
                            **mean_interval(difference), "p_value": wilcoxon(difference).pvalue})
    comparison = table(holm(pd.DataFrame(comparisons)), "d2_11_mixed_methods_comparisons.csv")
    save_json({"pool_rows": POOL, "anchor_rows": ANCHORS, "resamples": RESAMPLES, "subsample_rows": SUBSAMPLE,
               "gamma": GAMMA, "numeric_scaling": "z-score on the pool, clipped at +/-5 (k-prototypes)",
               "gower_ranges": "pool minimum to maximum", "chosen_k": best_k,
               "k_rule": "highest silhouette on Gower distance, k = 2 to 6, no labels",
               "note": "settings learned once on the pool; labels used only to score agreement"},
              "results/d2_11_mixed_methods_protocol.json")
    four_methods_figure(summary)
    return trial_frame, summary, comparison


def four_methods_figure(summary):
    """Stability and agreement with cancellations for all four clusterings (mean, 95% CI, 20 subsamples)."""
    metrics = ["stability_ari", "label_ari", "label_nmi"]
    methods = ["K-means numeric", "K-means mixed", "k-prototypes", "Gower k-medoids"]
    plt.figure(figsize=(9, 4.4))
    for offset, method in zip([-0.24, -0.08, 0.08, 0.24], methods):
        group = summary[summary.geometry == method].set_index("metric").loc[metrics]
        plt.errorbar(np.arange(3) + offset, group["mean"],
                     yerr=[group["mean"] - group.ci_low, group.ci_high - group["mean"]], fmt="o", capsize=3, label=method)
    plt.xticks(np.arange(3), ["stability (ARI)", "agreement with\ncancellations (ARI)", "agreement with\ncancellations (NMI)"])
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right", fontsize=8)
    figure("fig_B_04_mixed_methods.png", "Four clusterings: none recovers cancellations (20 subsamples)",
           "", "Score (1 = identical, 0 = chance)")


def stability_figure(summary):
    """Stability (left) and agreement with cancellations (middle, right), mean with 95% CI over 20 subsamples."""
    metrics = ["stability_ari", "label_ari", "label_nmi"]
    plt.figure(figsize=(8, 4.2))
    for offset, (kind, group) in zip([-0.12, 0.12], summary.groupby("geometry")):
        group = group.set_index("metric").loc[metrics]
        plt.errorbar(np.arange(3) + offset, group["mean"], yerr=[group["mean"] - group.ci_low, group.ci_high - group["mean"]],
                     fmt="o", capsize=3, label=f"{kind} fields")
    plt.xticks(np.arange(3), ["stability (ARI)", "agreement with\ncancellations (ARI)", "agreement with\ncancellations (NMI)"])
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right")
    figure("fig_B_03_cluster_stability.png", "Cluster stability and agreement with cancellations (20 subsamples)",
           "", "Score (1 = identical, 0 = chance)")
