import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer

from src.exception import CustomException
from src.logger import logging
from src.utils import clean_text, group_categories, save_object


@dataclass(frozen=True)
class DataTransformationConfig:
    preprocessor_obj_file_path: Path = Path("artifacts") / "preprocessor.pkl"
    feature_columns: List[str] = field(default_factory=lambda: ["description", "director", "cast"])
    target_column: str = "listed_in"
    genre_groups: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "international": [
                "International Movies",
                "International TV Shows",
                "British TV Shows",
                "Spanish-Language TV Shows",
                "Korean TV Shows",
            ],
            "drama": [
                "TV Dramas",
                "Dramas",
            ],
            "comedy": [
                "TV Comedies",
                "Comedies",
                "Stand-Up Comedy",
                "Stand-Up Comedy & Talk Shows",
            ],
            "action_adventure": [
                "Action & Adventure",
                "TV Action & Adventure",
            ],
            "documentary": [
                "Documentaries",
                "Docuseries",
                "Science & Nature TV",
            ],
            "family_children": [
                "Children & Family Movies",
                "Kids' TV",
            ],
            "romance": [
                "Romantic Movies",
                "Romantic TV Shows",
            ],
            "speculative": [
                "Horror Movies",
                "TV Horror",
                "Sci-Fi & Fantasy",
                "TV Sci-Fi & Fantasy",
            ],
            "thriller_crime": [
                "Thrillers",
                "TV Thrillers",
                "Crime TV Shows",
            ],
            "anime": [
                "Anime Series",
                "Anime Features",
            ],
            "independent": ["Independent Movies"],
            "reality": ["Reality TV"],
            "classic": [
                "Classic Movies",
                "Classic & Cult TV",
                "Cult Movies",
            ],
            "special_interest": [
                "LGBTQ Movies",
                "Faith & Spirituality",
                "Sports Movies",
                "Music & Musicals",
            ],
            "other": [
                "Movies",
                "TV Shows",
                "Teen TV Shows",
                "TV Mysteries",
            ],
        }
    )


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.mlb = MultiLabelBinarizer()

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates a preprocessing pipeline that handles text features.
        """
        try:
            logging.info(f"Feature columns: {self.data_transformation_config.feature_columns}")
            logging.info(f"Target columns: [{self.data_transformation_config.target_column}]")

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

            # Extract features
            features_train_df = train_df[self.data_transformation_config.feature_columns]
            features_test_df = test_df[self.data_transformation_config.feature_columns]

            # Extract targets and group categories
            target_train_df = (
                train_df[self.data_transformation_config.target_column]
                .str.split(", ")
                .apply(lambda x: group_categories(self.data_transformation_config.genre_groups, x))
            )
            target_test_df = (
                test_df[self.data_transformation_config.target_column]
                .str.split(", ")
                .apply(lambda x: group_categories(self.data_transformation_config.genre_groups, x))
            )

            logging.info("Features and targets extracted")

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
