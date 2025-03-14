import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer

from src.exception import CustomException
from src.logger import logging
from src.utils import clean_text, save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: Path = Path("artifacts") / "preprocessor.pkl"
    feature_columns: List[str] = field(default_factory=lambda: ["description", "director", "cast"])
    target_column: str = "listed_in"


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.mlb = MultiLabelBinarizer()

    def get_data_transformer_object(self):
        """
        Creates a preprocessing pipeline that handles text features.
        """
        try:
            logging.info(f"Feature columns: {self.data_transformation_config.feature_columns}")
            logging.info(f"Target columns: {self.data_transformation_config.target_column}")

            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                    ("text_cleaner", FunctionTransformer(clean_text)),
                    ("tfidf", TfidfVectorizer(stop_words="english")),
                ]
            )

            logging.info("Columns processing pipeline created")

            return pipe

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Performs the data transformation process:
        1. Loads train and test data
        2. Applies preprocessing to features
        3. Transforms target labels using MultiLabelBinarizer
        4. Saves the preprocessing object for inference

        Returns:
            Tuple containing:
            - Processed training data (features, targets)
            - Processed test data (features, targets)
            - Path to the saved preprocessor
        """
        try:
            logging.info("Starting data transformation process")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train dataset shape: {train_df.shape}, Test dataset shape: {test_df.shape}")

            # Extract features and target
            features_train_df = train_df[self.data_transformation_config.feature_columns]
            features_test_df = test_df[self.data_transformation_config.feature_columns]

            target_train_df = train_df[self.data_transformation_config.target_column].str.split(", ")
            target_test_df = test_df[self.data_transformation_config.target_column].str.split(", ")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on features")
            features_train_arr = preprocessing_obj.fit_transform(features_train_df)
            features_test_arr = preprocessing_obj.transform(features_test_df)

            logging.info(
                f"Transformed feature matrix shapes - Train: {features_train_arr.shape}, Test: {features_test_arr.shape}"
            )

            logging.info("Applying multi label binarizer on target")
            target_train_arr = self.mlb.fit_transform(target_train_df)
            target_test_arr = self.mlb.transform(target_test_df)

            logging.info(f"Target matrix shapes - Train: {target_train_arr.shape}, Test: {target_test_arr.shape}")
            logging.info(f"Number of unique genre categories: {len(self.mlb.classes_)}")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={"preprocessor": preprocessing_obj, "mlb": self.mlb},
            )

            logging.info(f"Preprocessing objects saved to {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                (features_train_arr, target_train_arr),
                (features_test_arr, target_test_arr),
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
