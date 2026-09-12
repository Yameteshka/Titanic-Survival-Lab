import pandas as pd

from code.models.features import add_derived_features, build_preprocessor


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"Pclass":1,"Sex":"female","Age":10,"SibSp":1,"Parch":1,"Fare":90.0,"Embarked":"C"},
        {"Pclass":3,"Sex":"male","Age":31,"SibSp":0,"Parch":0,"Fare":8.0,"Embarked":"S"},
        {"Pclass":2,"Sex":"female","Age":65,"SibSp":0,"Parch":1,"Fare":30.0,"Embarked":"Q"},
    ])


def test_feature_engineering_values():
    result = add_derived_features(sample_frame())
    assert result.loc[0, "FamilySize"] == 3
    assert result.loc[1, "IsAlone"] == 1
    assert result.loc[0, "FarePerPerson"] == 30
    assert result.loc[2, "AgeGroup"] == "senior"


def test_preprocessing_output_shape_is_stable():
    preprocessor = build_preprocessor()
    fitted = preprocessor.fit_transform(sample_frame())
    transformed = preprocessor.transform(sample_frame().iloc[:1])
    assert fitted.shape[0] == 3
    assert transformed.shape == (1, fitted.shape[1])
