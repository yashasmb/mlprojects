import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj



class  PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            preprocessor = load_obj(preprocessor_path)
            model_path = os.path.join('artifacts', 'model.pkl')
            model = load_obj(model_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                gender :str,
                race_ethnicity :str,
                parental_level_of_education :str,
                lunch :str,
                test_preparation_course :str,
                reading_score :int,
                writing_score :int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)