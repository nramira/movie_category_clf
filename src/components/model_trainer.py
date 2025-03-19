import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
            "Multi Layer Perceptron": MultiOutputClassifier(MLPClassifier()),
            "Extra Trees": MultiOutputClassifier(ExtraTreesClassifier()),
            "K Neighbors": MultiOutputClassifier(KNeighborsClassifier()),
        }
    )
    params: Dict[str, any] = field(
        default_factory=lambda: {
            "XGBoost": {
                "estimator__n_estimators": [50, 100, 200, 300],
                "estimator__max_depth": [3, 5, 7, 9],
                "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "estimator__subsample": [0.7, 0.8, 0.9, 1.0],
                "estimator__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "estimator__min_child_weight": [1, 3, 5],
                "estimator__gamma": [0, 0.1, 0.2],
            },
            "Multi Layer Perceptron": {
                "estimator__activation": ["relu", "tanh"],
                "estimator__hidden_layer_sizes": [(64,), (64, 32)],
                "estimator__solver": ["adam"],
                "estimator__alpha": [0.0001],
                "estimator__batch_size": [256],  # Larger batch size for faster processing
                "estimator__learning_rate": ["adaptive"],
                "estimator__learning_rate_init": [0.01],
                "estimator__max_iter": [100],
                "estimator__early_stopping": [True],
                "estimator__validation_fraction": [0.1],
                "estimator__tol": [1e-3],
                "estimator__n_iter_no_change": [10],
                "estimator__verbose": [False],
            },
            "Extra Trees": {
                "estimator__n_estimators": [50, 100, 200],
                "estimator__max_depth": [10, 20, 30],
                "estimator__min_samples_split": [2, 5, 10],
                "estimator__min_samples_leaf": [1, 2, 5],
                "estimator__max_features": ["sqrt", "log2"],
                "estimator__class_weight": ["balanced", None],
            },
            "K Neighbors": {
                "estimator__n_neighbors": [3, 5, 7, 9, 15, 21],
                "estimator__weights": ["uniform", "distance"],
                "estimator__metric": ["euclidean", "manhattan"],
                "estimator__leaf_size": [20, 30, 40],
            },
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
            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, self.model_trainer_config.models, self.model_trainer_config.params
            )

            logging.info("Selecting model with best jaccard_score")
            best_model_name = sorted(model_report, key=lambda k: model_report[k]["jaccard_score"])[-1]
            best_model_jaccard_score = model_report[best_model_name]["jaccard_score"]
            best_estimator = model_report[best_model_name]["Best Estimator"]

            if best_model_jaccard_score < 0.4:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found in testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_estimator,
            )
            logging.info(
                f"Best estimator for {best_model_name} saved to {self.model_trainer_config.trained_model_file_path}"
            )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
