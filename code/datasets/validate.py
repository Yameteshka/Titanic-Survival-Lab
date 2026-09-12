import logging
from pathlib import Path

import pandas as pd

from code.config import RAW_DATA_PATH, RAW_FEATURES, TARGET

LOGGER = logging.getLogger(__name__)
NUMERIC_COLUMNS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
ALLOWED_CATEGORIES = {"Sex": {"male", "female"}, "Embarked": {"C", "Q", "S"}}


class DataValidationError(ValueError):
    """Raised when raw data cannot safely enter the pipeline."""


def validate_raw_data(path: Path = RAW_DATA_PATH) -> dict:
    if not path.is_file():
        raise DataValidationError(f"Raw data file does not exist: {path}")
    data = pd.read_csv(path)
    required = [*RAW_FEATURES, TARGET]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    if data.empty or len(data) < 20:
        raise DataValidationError(f"Raw dataset is too small: {len(data)} rows")

    invalid: dict[str, int] = {}
    for column in [*NUMERIC_COLUMNS, TARGET]:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_count = int((data[column].notna() & converted.isna()).sum())
        if invalid_count:
            invalid[f"{column}_non_numeric"] = invalid_count

    target_values = set(pd.to_numeric(data[TARGET], errors="coerce").dropna().unique())
    if not target_values.issubset({0, 1}):
        invalid["Survived_outside_0_1"] = int(
            (~pd.to_numeric(data[TARGET], errors="coerce").isin([0, 1]) & data[TARGET].notna()).sum()
        )

    numeric = data[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    checks = {
        "Age_negative": numeric["Age"] < 0,
        "Fare_negative": numeric["Fare"] < 0,
        "SibSp_negative": numeric["SibSp"] < 0,
        "Parch_negative": numeric["Parch"] < 0,
        "Pclass_invalid": ~numeric["Pclass"].isin([1, 2, 3]) & numeric["Pclass"].notna(),
    }
    for name, mask in checks.items():
        count = int(mask.sum())
        if count:
            invalid[name] = count
    for column, allowed in ALLOWED_CATEGORIES.items():
        mask = ~data[column].isin(allowed) & data[column].notna()
        if int(mask.sum()):
            invalid[f"{column}_invalid"] = int(mask.sum())
    if invalid:
        raise DataValidationError(f"Invalid raw values detected: {invalid}")

    report = {
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": {column: int(value) for column, value in data[required].isna().sum().items()},
        "target_distribution": {str(int(k)): int(v) for k, v in data[TARGET].dropna().value_counts().items()},
    }
    LOGGER.info("Raw data validation passed: %s", report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    validate_raw_data()
