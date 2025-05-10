import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


def run_pipeline() -> dict:
    """
    Main function that orchestrates the entire ML pipeline from data ingestion to model training
    """
    try:
        logging.info("Starting the ML pipeline")

        # Initialize data ingestion
        data_ingestor = DataIngestion()
        data_ingestor.create_kaggle_credentials()
        data_ingestor.download_nltk_datasets()
        train_path, test_path = data_ingestor.initiate_data_ingestion()

        # Initialize data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

        # Initialize model training
        model_trainer = ModelTrainer()
        model_report = model_trainer.initiate_model_trainer(train_arr, test_arr)

        return model_report

    except Exception as e:
        logging.error("Pipeline execution failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    model_report = run_pipeline()
    print(model_report)
