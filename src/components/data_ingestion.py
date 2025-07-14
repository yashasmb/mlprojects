import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            dataset_path = os.path.join('notebook', 'data', 'stud.csv')
            df = pd.read_csv(dataset_path)
            logging.info(f'Dataset loaded from {dataset_path}. Shape: {df.shape}')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Raw dataset saved to {self.ingestion_config.raw_data_path}')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f'Train-test split completed. Train: {len(train_set)}, Test: {len(test_set)}')

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of data is completed successfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f'Error occurred during data ingestion: {e}')
            raise CustomException(e, sys)

# Driver code for standalone execution
if __name__ == '__main__':
    try:
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        logging.info(f'Train data saved at: {train_data}')
        logging.info(f'Test data saved at: {test_data}')
        

    except Exception as e:
        logging.error(f'An error occurred during ingestion: {e}')
        sys.exit(1)
