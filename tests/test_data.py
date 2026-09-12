import pandas as pd
import pytest

from code.datasets.validate import DataValidationError, validate_raw_data


def test_data_validation_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Survived": [0, 1] * 10}).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_raw_data(path)
