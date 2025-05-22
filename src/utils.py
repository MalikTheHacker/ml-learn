import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    try:
        report = dict()
        for i, model in enumerate(list(models.values())):
            
            model.fit(X_train, y_train)
            y_preds = model.predict(X_test)

            score = r2_score(y_test, y_preds)

            report[list(models.keys())[i]] = score
        return report
    except Exception as e:
        raise CustomException(e, sys)