import os
import sys

import dill
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    model_report = {}
    logging.info("____________________Evaluating models______________________________")
    for model_name, model in models.items():
        try:
            logging.info(f"+++++++++++++++++++++++Training {model_name}+++++++++++++++++++++++++++")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            logging.info(f"Evaluating {model_name}")
            r2_square = r2_score(y_test, y_pred)
            model_report[model_name] = r2_square
            logging.info(f"{model_name} R2 Score: {r2_square}")
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")
            model_report[model_name] = None

    return model_report