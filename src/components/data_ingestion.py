import json
import sys
from dataclasses import dataclass
from pathlib import Path

import kaggle
import nltk
import pandas as pd
from environs import Env
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

# Initialize environs
env = Env()
env.read_env()


@dataclass(frozen=True)
class DataIngestionConfig:
    artifacts_path: Path = Path("artifacts")
    train_data_path: Path = artifacts_path / "train.csv"
    test_data_path: Path = artifacts_path / "test.csv"
    raw_data_path: Path = artifacts_path / "netflix_titles.csv"
    kaggle_dataset: str = "anandshaw2001/netflix-movies-and-tv-shows"
    kaggle_path: Path = Path.home() / ".kaggle"
    kaggle_config_path: Path = kaggle_path / "kaggle.json"
    username: str = env.str("KAGGLE_USERNAME")
    key: str = env.str("KAGGLE_KEY")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def create_kaggle_credentials(self) -> None:
        """
        Creates kaggle.json file using secrets saved as environtment variables
        """
        logging.info("Setting up Kaggle configuration")
        self.ingestion_config.kaggle_path.mkdir(exist_ok=True)

        config = {
            "username": self.ingestion_config.username,
            "key": self.ingestion_config.key,
        }

        with open(self.ingestion_config.kaggle_config_path, "w") as f:
            json.dump(config, f)

        Path.chmod(self.ingestion_config.kaggle_config_path, 0o600)
        logging.info(f"Secrets sucessfully stored in {self.ingestion_config.kaggle_config_path}")

    def download_nltk_datasets(self) -> None:
        """
        Downloads the stopwords and punks datasets from the NLTK library
        """
        logging.info("Downloading necessary NLTK datasets")
        nltk.download("stopwords")

    def initiate_data_ingestion(self) -> tuple[Path, Path]:
        logging.info("Entered the data ingestion component")
        try:
            logging.info(f"Create {self.ingestion_config.artifacts_path} directory")

            logging.info(f"Download dataset from Kaggle: {self.ingestion_config.kaggle_dataset}")
            kaggle.api.dataset_download_files(
                self.ingestion_config.kaggle_dataset,
                path=self.ingestion_config.artifacts_path,
                unzip=True,
            )

            logging.info("Read dataset")
            df = pd.read_csv(self.ingestion_config.raw_data_path)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=7)

            logging.info(f"Save train and test files in {self.ingestion_config.artifacts_path}")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestor = DataIngestion()
    data_ingestor.create_kaggle_credentials()
    data_ingestor.download_nltk_datasets()
    train_path, test_path = data_ingestor.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
