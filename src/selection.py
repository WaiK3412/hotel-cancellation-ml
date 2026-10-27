"""Phase 0: screen four public datasets against the brief, then clean and split the chosen one.

Screening uses only counts and simple statistics; nothing is fitted.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import figure, log, table
from src.data import OUTCOME_FIELDS, SOURCES, clean_and_split, missing_mask, read_candidate, record_sources


def numeric_columns(data, mask):
    """Columns whose every non-missing value parses as a number."""
    columns = {}
    for column in data:
        parsed = pd.to_numeric(data[column].mask(mask[column]), errors="coerce")
        if parsed.notna().any() and parsed.notna().sum() == (~mask[column]).sum():
            columns[column] = parsed
    return pd.DataFrame(columns)


def inconsistency_counts(data, mask):
    """Whitespace padding, case variants and mixed number/text values per column."""
    rows = []
    for column in data:
        text = data[column].astype(str)
        stripped = text.str.strip()
        values = pd.DataFrame({"canonical": stripped.str.lower(), "original": text}).drop_duplicates()
        parsed = pd.to_numeric(text, errors="coerce")[~mask[column]]
        mixed = data[column].dtype == object and parsed.notna().any() and parsed.isna().any()
        rows.append({"feature": column, "whitespace_rows": int((text != stripped).sum()),
                     "case_variant_groups": int((values.groupby("canonical").original.nunique() > 1).sum()),
                     "mixed_numeric_rows": int(parsed.notna().sum()) if mixed else 0})
    return pd.DataFrame(rows)


def conflicting_label_rows(data, target):
    """Rows whose predictors (outcome fields excluded) match another row with a different label."""
    predictors = data.drop(columns=[target] + [c for c in OUTCOME_FIELDS if c in data])
    hashes = pd.util.hash_pandas_object(predictors, index=False)
    labels_per_group = data[target].groupby(hashes).nunique()
    return int(hashes.isin(labels_per_group[labels_per_group > 1].index).sum())


def screen():
    """Scorecard of the four candidates, then the Hotel data is cleaned and split."""
    scorecard, outlier_rows = [], []
    for name, (_, url, target, licence) in SOURCES.items():
        data = read_candidate(name)
        short = name.split()[0].lower()
        mask = missing_mask(data)
        numeric = numeric_columns(data.drop(columns=[target]), mask)
        for column in numeric:
            q1, q3 = numeric[column].quantile([0.25, 0.75])
            flagged = (numeric[column] < q1 - 1.5 * (q3 - q1)) | (numeric[column] > q3 + 1.5 * (q3 - q1))
            outlier_rows.append({"dataset": name, "feature": column, "outlier_rows": int(flagged.sum()),
                                 "skew": numeric[column].skew()})
        inconsistencies = table(inconsistency_counts(data, mask), f"phase0_{short}_inconsistencies.csv")
        table(pd.DataFrame({"feature": data.columns, "missing_cells": mask.sum().values,
                            "missing_pct": mask.mean().values * 100}), f"phase0_{short}_missing.csv")
        counts = data[target].value_counts()
        implausible = 0
        if name == "Hotel Booking Demand":
            guests = data.adults + pd.to_numeric(data.children, errors="coerce").fillna(0) + data.babies
            implausible = int(((guests == 0) | (data.adr < 0)).sum())
        scorecard.append({
            "dataset": name, "rows": len(data), "attributes": data.shape[1] - 1, "target": target,
            "classes": len(counts), "missing_cells": int(mask.sum().sum()), "duplicates": int(data.duplicated().sum()),
            "iqr_outlier_cells": int(sum(row["outlier_rows"] for row in outlier_rows if row["dataset"] == name)),
            "skewed_features": int((numeric.skew().abs() > 1).sum()), "imbalance_ratio": counts.max() / counts.min(),
            "conflicting_label_rows": conflicting_label_rows(data, target), "implausible_rows": implausible,
            "whitespace_rows": int(inconsistencies.whitespace_rows.sum()),
            "case_variant_groups": int(inconsistencies.case_variant_groups.sum()),
            "mixed_numeric_rows": int(inconsistencies.mixed_numeric_rows.sum()),
            "semantic_variant_rows": int((data.meal == "Undefined").sum()) if name == "Hotel Booking Demand" else 0,
            "varying_numeric_features": int((numeric.nunique() > 1).sum()), "source": url, "licence": licence,
        })
    result = table(pd.DataFrame(scorecard), "phase0_scorecard.csv")
    table(pd.DataFrame(outlier_rows), "phase0_outliers_skew.csv")
    record_sources()
    shown = result.set_index("dataset")[["missing_cells", "duplicates", "conflicting_label_rows", "semantic_variant_rows"]]
    shown.replace(0, np.nan).plot.bar(ax=plt.figure(figsize=(9, 4)).gca(), logy=True, rot=0)
    figure("fig_C_00_candidate_quality.png", "Data-quality issues in the four candidate datasets (log scale)",
           "Candidate dataset", "Cells or rows")
    clean_and_split()
    log("Phase 0", "screening", "Four public candidates screened; Hotel Booking Demand cleaned and split.")
    return result
