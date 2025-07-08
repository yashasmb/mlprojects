import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# class DataIngestionConfig:
#     def __init__(self, raw_data_path, train_data_path, test_data_path, base_dir):
#         self.raw_data_path = os.path.join(base_dir, raw_data_path)
#         self.train_data_path = os.path.join(base_dir, train_data_path)
#         self.test_data_path = os.path.join(base_dir, test_data_path)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str =   os.path.join('artifacts', 'test_data.csv')

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.injection_config = config

    def  initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')   

            os.makedirs(os.path.dirname(self.injection_config.raw_data_path), exist_ok=True)
            df.to_csv(self.injection_config.raw_data_path, index=False, header=True)
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info('Train test split completed')
            train_set.to_csv(self.injection_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.injection_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of data is completed')
            return (
                self.injection_config.train_data_path,
                self.injection_config.test_data_path,
                self.injection_config.raw_data_path
            )
        except Exception as e:
            logging.error(f'Error occurred during data ingestion: {e}')
            raise e



if __name__ == '__main__':
    try:
        config = DataIngestionConfig()
        data_ingestion = DataIngestion(config=config)
        train_data, test_data, raw_data = data_ingestion.initiate_data_ingestion()
        logging.info(f'Train data path: {train_data}')
        logging.info(f'Test data path: {test_data}')
        logging.info(f'Raw data path: {raw_data}')
    except Exception as e:
        logging.error(f'An error occurred: {e}')
        sys.exit(1)



