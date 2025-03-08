import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from pathlib import Path
from typing import List
from dataclasses import dataclass, field

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, clean_text


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: Path = Path("artifacts") / "preprocessor.pkl"
    feature_columns: List[str] = field(default_factory=lambda: ['description', 'director', 'cast'])
    target_column: str = 'listed_in'

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            logging.info(f"Feature columns: {self.data_transformation_config.feature_columns}")
            logging.info(f"Target columns: {self.data_transformation_config.target_column}")

            pipe = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                    ("text_cleaner", FunctionTransformer(clean_text)),
                    ("tfidf", TfidfVectorizer(stop_words="english"))
                ]
            )

            logging.info("Columns processing completed")

            return pipe
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Extract features and target from train and test data")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            features_train_df = train_df[self.data_transformation_config.feature_columns]
            features_test_df = test_df[self.data_transformation_config.feature_columns]

            target_train_df = train_df[self.data_transformation_config.target_column].str.split(', ')
            target_test_df = test_df[self.data_transformation_config.target_column].str.split(', ')

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on features")

            features_train_csr_matrix = preprocessing_obj.fit_transform(features_train_df)
            features_test_csr_matrix = preprocessing_obj.transform(features_test_df)

            logging.info("Applying multi label binarizer on target")

            mlb = MultiLabelBinarizer()
            target_train_arr = mlb.fit_transform(target_train_df)
            target_test_arr = mlb.transform(target_test_df)

            train = (features_train_csr_matrix, target_train_arr)
            test = (features_test_csr_matrix, target_test_arr)
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train,
                test,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)