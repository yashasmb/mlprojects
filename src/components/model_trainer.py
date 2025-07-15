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
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Nearest Neighbors': KNeighborsRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(verbose=0),
                'XGBRegressor': XGBRegressor(eval_metric='rmse'),
                'AdaBoost Regressor': AdaBoostRegressor()
            }


            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Nearest Neighbors":{
                    'n_neighbors':[5,7,9,11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report = {}
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params= params)  
           

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