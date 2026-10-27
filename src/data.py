"""Phase 0 data layer: download the sources, clean them deterministically and split train/test.

Design:
- Cleaning never looks at the outcome. Only exact duplicate records (all 32 source columns
  identical) are removed; no row is dropped because of its label.
- The split is group-aware. Bookings that share every booking-time field in KEY_COLUMNS form one
  group, and a group is never divided between training and test (or between CV folds). This
  stops near-identical bookings leaking across the split without using the labels to filter.
"""
import hashlib
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src.common import ROOT, SEED, load_json, log, save_json, table

HOTEL_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv"
SOURCES = {
    "Hotel Booking Demand": ("hotels.csv", HOTEL_URL, "is_canceled", "CC BY 4.0 (Antonio, de Almeida and Nunes, 2019)"),
    "Adult": ("adult.zip", "https://archive.ics.uci.edu/static/public/2/adult.zip", "income", "CC BY 4.0 (UCI)"),
    "Bank Marketing": ("bank.zip", "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip", "y", "CC BY 4.0 (UCI)"),
    "Diabetes": ("diabetes.zip", "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip", "readmitted", "CC BY 4.0 (UCI)"),
}
TARGET = "is_canceled"
# Tokens that mean "no value" in the source files.
MISSING = {"", "?", "unknown", "na", "nan", "null"}
# Fields that record the outcome or what happened after booking: never predictors.
OUTCOME_FIELDS = ["reservation_status", "reservation_status_date", "assigned_room_type",
                  "booking_changes", "days_in_waiting_list"]
# Identifiers and codes: categories, not measured quantities.
CATEGORICAL = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
               "reserved_room_type", "deposit_type", "customer_type", "agent", "company"]
# Fields whose day-before-arrival snapshot value may differ from what was known at booking.
# Phase 1 (src/features.py) applies a written rule to each and records the decision.
LEAKAGE_CANDIDATES = {
    "country": "The dataset paper states that nationality is often unknown until check-in.",
    "required_car_parking_spaces": "Parking is requested for the stay itself; checked for an outcome-tied pattern.",
    "deposit_type": "The dataset paper derives it from payments made before arrival or cancellation.",
    "total_of_special_requests": "Requests can be added at any time up to arrival.",
    "adr": "The average daily rate can be revised after booking.",
    "arrival_date_year": "A calendar year cannot be learned for future years.",
}
# Booking-time fields that define a booking group for the split (all predictors except the
# leakage candidates). Bookings identical on these fields stay on one side of every split.
KEY_COLUMNS = ["hotel", "lead_time", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month",
               "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "meal",
               "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
               "previous_bookings_not_canceled", "reserved_room_type", "agent", "company", "customer_type"]


def download(name):
    """Return the local path of a raw source, downloading it once if absent."""
    filename, url, _, _ = SOURCES[name]
    path = ROOT / "data/raw" / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_candidate(name):
    """Read one screening candidate exactly as published (no cleaning)."""
    path = download(name)
    if name == "Hotel Booking Demand":
        return pd.read_csv(path, keep_default_na=False, low_memory=False)
    with zipfile.ZipFile(path) as archive:
        if name == "Adult":
            names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
                     "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
                     "native_country", "income"]
            parts = [pd.read_csv(archive.open(part), names=names, comment="|", skipinitialspace=True,
                                 keep_default_na=False) for part in ["adult.data", "adult.test"]]
            data = pd.concat(parts, ignore_index=True)
            # The test file writes labels with a trailing full stop ("<=50K."); harmonise them.
            data["income"] = data.income.str.rstrip(".")
            return data
        if name == "Bank Marketing":
            with zipfile.ZipFile(io.BytesIO(archive.read("bank-additional.zip"))) as nested:
                return pd.read_csv(nested.open("bank-additional/bank-additional-full.csv"), sep=";",
                                   keep_default_na=False)
        member = next(item for item in archive.namelist() if item.endswith("diabetic_data.csv"))
        return pd.read_csv(archive.open(member), keep_default_na=False)


def record_sources():
    """Write byte sizes and SHA-256 hashes of every raw source to data/raw/SOURCES.json."""
    records = []
    for name, (_, url, _, licence) in SOURCES.items():
        path = download(name)
        records.append({"name": name, "url": url, "licence": licence, "bytes": path.stat().st_size,
                        "sha256": sha256(path)})
    save_json(records, "data/raw/SOURCES.json")


def missing_mask(data):
    """True where a cell is NaN or one of the source's missing-value tokens."""
    return data.apply(lambda column: column.isna() | column.astype(str).str.strip().str.lower().isin(MISSING))


def standardise(raw):
    """Deterministic, label-free cleaning of the hotel source.

    Trims text, turns missing-value tokens into NaN, maps the meal code "Undefined" to "SC"
    (the source dictionary defines both as "no meal package"), converts measured fields to
    numbers and keeps identifiers (agent, company, codes) as text categories.
    """
    clean = raw.copy()
    for column in clean.select_dtypes("object"):
        stripped = clean[column].str.strip()
        clean[column] = stripped.mask(stripped.str.lower().isin(MISSING))
    clean["meal"] = clean["meal"].replace({"Undefined": "SC"})
    text_fields = set(CATEGORICAL) | {"reservation_status", "reservation_status_date", "assigned_room_type"}
    for column in clean.columns:
        if column not in text_fields:
            clean[column] = pd.to_numeric(clean[column], errors="raise")
    for column in CATEGORICAL:
        clean[column] = clean[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return clean


def booking_groups(frame):
    """Group id per row: a hash of the booking-time KEY_COLUMNS (ignores the outcome)."""
    return pd.util.hash_pandas_object(frame[KEY_COLUMNS], index=False).to_numpy()


def clean_and_split():
    """Build data/processed/: cleaned bookings plus a group-aware stratified 80/20 split."""
    from src.metrics import GroupedStratifiedKFold

    raw = pd.read_csv(download("Hotel Booking Demand"), keep_default_na=False, low_memory=False)
    clean = standardise(raw)
    clean.index.name = "source_row"
    duplicated = clean.duplicated()
    # How many identical source records each kept row stands for (used by a weighting ablation).
    full_hash = pd.util.hash_pandas_object(clean, index=False)
    multiplicity = full_hash.map(full_hash.value_counts()).to_numpy()
    clean = clean.loc[~duplicated]
    table(pd.DataFrame({"source_row": clean.index, "source_multiplicity": multiplicity[~duplicated.to_numpy()]}),
          "phase0_duplicate_multiplicity.csv")

    groups = booking_groups(clean)
    # One fold of a 5-fold group-stratified split = a 20% holdout that keeps groups intact.
    splitter = GroupedStratifiedKFold(n_splits=5, random_state=SEED)
    train_pos, test_pos = next(splitter.split(clean, clean[TARGET], groups))
    train, test = clean.index[train_pos].to_numpy(), clean.index[test_pos].to_numpy()
    assert not set(train) & set(test)
    assert not set(groups[train_pos]) & set(groups[test_pos]), "A booking group spans the holdout"

    for name, rows in [("train", train), ("test", test)]:
        pd.DataFrame({"source_row": rows}).to_csv(ROOT / f"data/processed/{name}_indices.csv", index=False,
                                                  lineterminator="\n")
    clean.to_csv(ROOT / "data/processed/cleaned.csv", lineterminator="\n")

    group_sizes = pd.Series(groups).value_counts()
    label_spread = pd.Series(clean[TARGET].to_numpy()).groupby(groups).nunique()
    counts = [
        {"step": "raw source rows", "rows": len(raw), "removed": 0},
        {"step": "missing tokens, trimming, types, meal codes", "rows": len(raw), "removed": 0},
        {"step": "exact duplicate records (all 32 columns)", "rows": len(clean), "removed": int(duplicated.sum())},
    ]
    table(pd.DataFrame(counts), "phase0_cleaning_counts.csv")
    save_json({
        "seed": SEED, "method": "StratifiedGroupKFold(5), first fold held out",
        "rows": len(clean), "train_rows": len(train), "test_rows": len(test),
        "train_positive_fraction": float(clean.loc[train, TARGET].mean()),
        "test_positive_fraction": float(clean.loc[test, TARGET].mean()),
        "booking_groups": int(len(group_sizes)), "rows_in_shared_groups": int(group_sizes[group_sizes > 1].sum()),
        "groups_with_both_labels": int((label_spread > 1).sum()),
        "group_key_columns": KEY_COLUMNS, "outcome_fields_never_used": OUTCOME_FIELDS,
        "source_sha256": sha256(download("Hotel Booking Demand")),
        "policy": "label-free cleaning; group-aware split before any fitting; holdout first evaluated in Phase 4",
    }, "data/processed/split_manifest.json")
    log("Phase 0", "clean and split",
        f"{len(raw)} raw rows -> {len(clean)} after removing {int(duplicated.sum())} exact duplicates; "
        f"{len(train)} train / {len(test)} test, {len(group_sizes)} booking groups, none split.")
    return clean


def excluded_features():
    """Leakage candidates excluded by the Phase 1 decision (results/phase1_feature_set.json)."""
    path = ROOT / "results/phase1_feature_set.json"
    if not path.exists():
        raise FileNotFoundError("Run notebook 01 first: it records the feature-set decision.")
    return load_json("results/phase1_feature_set.json")["excluded"]


def load_partition(partition="train", excluded=None):
    """Predictors X and target y for the train or test partition.

    By default the leakage-excluded fields from Phase 1 are dropped. Pass excluded=[] to keep
    every candidate (used only for EDA and for the leakage sensitivity experiments).
    """
    assert partition in {"train", "test"}
    clean = pd.read_csv(ROOT / "data/processed/cleaned.csv", index_col="source_row",
                        dtype={column: str for column in CATEGORICAL})
    rows = pd.read_csv(ROOT / f"data/processed/{partition}_indices.csv").source_row
    data = clean.loc[rows]
    drop = [TARGET] + OUTCOME_FIELDS + list(excluded_features() if excluded is None else excluded)
    return data.drop(columns=drop), data[TARGET].astype(int)


def load_audit_columns(partition, columns):
    """Fields kept for auditing only (e.g. country for the fairness check), never for fitting."""
    clean = pd.read_csv(ROOT / "data/processed/cleaned.csv", index_col="source_row",
                        dtype={column: str for column in CATEGORICAL}, usecols=["source_row"] + list(columns))
    rows = pd.read_csv(ROOT / f"data/processed/{partition}_indices.csv").source_row
    return clean.loc[rows]
