import sys
from pathlib import Path

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features: pd.DataFrame) -> list:
        try:
            MODEL_PATH: Path = Path("artifacts") / "model.pkl"
            PREPROCESSOR_PATH: Path = Path("artifacts") / "preprocessor.pkl"

            # Load pkl files
            model = load_object(MODEL_PATH)
            preprocessor = load_object(PREPROCESSOR_PATH)
            feature_preprocessor = preprocessor["preprocessor"]
            mlb = preprocessor["mlb"]

            # Process new features
            processed_features = feature_preprocessor.transform(features)

            # Return predictions
            pred = model.predict(processed_features)

            # Get labels
            labels = mlb.inverse_transform(pred)

            return ", ".join(labels[0])

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, description: str, director: str, cast: str) -> None:
        self.description = description
        self.director = director
        self.cast = cast

    def get_data_as_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_disc = {
                "description": [self.description],
                "director": [self.director],
                "cast": [self.cast],
            }

            return pd.DataFrame(custom_data_input_disc)

        except Exception as e:
            raise CustomException(e, sys)
