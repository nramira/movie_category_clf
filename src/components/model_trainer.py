import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path: Path = Path("artifacts") / "model.pkl"
    models: Dict[str, MultiOutputClassifier] = field(
        default_factory=lambda: {
            "XGBoost": MultiOutputClassifier(XGBClassifier()),
            "CatBoost": MultiOutputClassifier(CatBoostClassifier(verbose=0)),
            "Multi Layer Perceptron": MultiOutputClassifier(MLPClassifier()),
            "Random Forest": MultiOutputClassifier(RandomForestClassifier()),
            "Extra Trees": MultiOutputClassifier(ExtraTreesClassifier()),
            "Decision Tree": MultiOutputClassifier(DecisionTreeClassifier()),
            "Extra Tree": MultiOutputClassifier(ExtraTreeClassifier()),
            "K Neighbors": MultiOutputClassifier(KNeighborsClassifier()),
            "Logistic Regression": MultiOutputClassifier(LogisticRegression()),
        }
    )


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr: tuple, test_arr: tuple) -> dict:
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_arr
            X_test, y_test = test_arr

            logging.info("Evaluating each model")
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, self.model_trainer_config.models)

            logging.info("Selecting model with best jaccard_score")
            best_model_name = sorted(model_report, key=lambda k: model_report[k][1])[-1]
            best_model_jaccard_score = model_report[best_model_name][1]
            best_model = self.model_trainer_config.models[best_model_name]

            if best_model_jaccard_score < 0.4:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found in testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info(f"Best model saved to {self.model_trainer_config.trained_model_file_path}")

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
