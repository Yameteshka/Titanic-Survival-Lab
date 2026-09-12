import pandas as pd

from code.models.inference import TitanicPredictor


def test_model_inference_returns_probability():
    passenger = pd.DataFrame([{"Pclass":1,"Sex":"female","Age":35,"SibSp":1,"Parch":0,"Fare":75.0,"Embarked":"C"}])
    result = TitanicPredictor().predict_frame(passenger)[0]
    assert result["predicted_class"] in (0, 1)
    assert 0 <= result["probability_survived"] <= 1
