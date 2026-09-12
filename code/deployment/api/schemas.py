from typing import Literal

from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    pclass: Literal[1, 2, 3]
    sex: Literal["male", "female"]
    age: float = Field(ge=0, le=120)
    sibsp: int = Field(ge=0, le=20)
    parch: int = Field(ge=0, le=20)
    fare: float = Field(ge=0, le=10000)
    embarked: Literal["C", "Q", "S"]

    def as_model_record(self) -> dict:
        return {
            "Pclass": self.pclass, "Sex": self.sex, "Age": self.age,
            "SibSp": self.sibsp, "Parch": self.parch, "Fare": self.fare, "Embarked": self.embarked,
        }


class PredictionOutput(BaseModel):
    predicted_class: int
    probability_survived: float
    probability_not_survived: float
    threshold: float
    model_version: str
