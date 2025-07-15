import os
from dataclasses import dataclass
import pandas as pd
import sys
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()    

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("Initializing models")
            models = {
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=0),
                'XGB Regressor': XGBRegressor(eval_metric='rmse')
            }
            model_report = {}
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)  
           

            best_model_score = max(model_report.values())
         
            best_model_name = max(model_report, key=model_report.get)


            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            
            logging.info("Saving the best model")
            save_obj(self.model_trainer_config.trained_model_file_path, best_model)

            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)
            logging.info(f"R2 score of the best model on test data: {r2_square}")


        except Exception as e:
            raise CustomException(e, sys)