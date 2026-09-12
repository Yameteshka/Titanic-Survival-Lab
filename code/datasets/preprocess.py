import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from code.config import (
    CLEANING_REPORT_PATH,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    RAW_FEATURES,
    TARGET,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from code.datasets.validate import validate_raw_data

LOGGER = logging.getLogger(__name__)
NUMERICAL_IMPUTE = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
CATEGORICAL_IMPUTE = ["Sex", "Embarked"]
OUTLIER_COLUMNS = ["Age", "Fare"]


def clean_and_split_data(
    raw_path: Path = RAW_DATA_PATH,
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    report_path: Path = CLEANING_REPORT_PATH,
) -> dict:
    validate_raw_data(raw_path)
    data = pd.read_csv(raw_path)
    initial_rows = len(data)
    initial_missing = {column: int(value) for column, value in data.isna().sum().items()}

    before_target = len(data)
    data = data.dropna(subset=[TARGET]).copy()
    target_rows_removed = before_target - len(data)
    data[TARGET] = data[TARGET].astype(int)

    before_duplicates = len(data)
    data = data.drop_duplicates().copy()
    duplicates_removed = before_duplicates - len(data)

    imputation_values: dict[str, object] = {}
    for column in NUMERICAL_IMPUTE:
        value = float(data[column].median())
        imputation_values[column] = value
        data[column] = data[column].fillna(value)
    for column in CATEGORICAL_IMPUTE:
        modes = data[column].mode(dropna=True)
        if modes.empty:
            raise ValueError(f"Cannot impute {column}: no non-missing values")
        value = str(modes.iloc[0])
        imputation_values[column] = value
        data[column] = data[column].fillna(value)

    bounds: dict[str, dict[str, float]] = {}
    outlier_mask = pd.Series(False, index=data.index)
    for column in OUTLIER_COLUMNS:
        q1 = float(data[column].quantile(0.25))
        q3 = float(data[column].quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bounds[column] = {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper}
        outlier_mask |= (data[column] < lower) | (data[column] > upper)
    outlier_rows_removed = int(outlier_mask.sum())
    data = data.loc[~outlier_mask].copy()

    clean_columns = [*RAW_FEATURES, TARGET]
    train, test = train_test_split(
        data[clean_columns], test_size=0.2, random_state=RANDOM_SEED, stratify=data[TARGET]
    )
    train = train.sort_index().reset_index(drop=True)
    test = test.sort_index().reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    report = {
        "initial_row_count": initial_rows,
        "initial_missing_values": initial_missing,
        "target_missing_rows_removed": target_rows_removed,
        "duplicate_rows_removed": duplicates_removed,
        "imputation_values": imputation_values,
        "outlier_method": "IQR with 1.5 multiplier",
        "outlier_bounds": bounds,
        "outlier_rows_removed": outlier_rows_removed,
        "rows_after_cleaning": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "split": {"test_size": 0.2, "random_state": RANDOM_SEED, "stratified_by": TARGET},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Cleaned %d rows into train=%d and test=%d", len(data), len(train), len(test))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    clean_and_split_data()
