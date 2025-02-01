import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df=pd.read_csv('data/netflix_titles.csv')
            logging.info('Read dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # add train test split if neccesary
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.train_data_path, index=False)

            logging.info("Data ingestion completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    data_ingestor=DataIngestion()
    data_ingestor.initiate_data_ingestion()
