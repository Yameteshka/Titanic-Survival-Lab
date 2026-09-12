import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = [
    "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson", "LogFare"
]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "AgeGroup"]


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    family_size = data["SibSp"] + data["Parch"] + 1
    data["FamilySize"] = family_size
    data["IsAlone"] = (family_size == 1).astype(int)
    data["FarePerPerson"] = np.divide(
        data["Fare"], family_size, out=np.zeros(len(data), dtype=float), where=family_size.to_numpy() != 0
    )
    data["LogFare"] = np.log1p(data["Fare"].clip(lower=0))
    data["AgeGroup"] = pd.cut(
        data["Age"], bins=[-np.inf, 12, 17, 29, 59, np.inf],
        labels=["child", "teenager", "young_adult", "adult", "senior"], right=True,
    ).astype(object)
    return data


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: object = None) -> "TitanicFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_derived_features(X)


def build_preprocessor() -> Pipeline:
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("numerical", numerical, NUMERICAL_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)
    return Pipeline([("features", TitanicFeatureEngineer()), ("columns", columns)])
