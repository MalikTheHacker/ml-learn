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
    

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    try:
        report = dict()
        for i, model in enumerate(list(models.values())):
            model_name = list(models.keys())[i]
            param_grid = params[model_name]
            grid = GridSearchCV(model, param_grid, cv=5, scoring="r2")
            grid.fit(X_train, y_train)
            model=grid.best_estimator_
            y_preds = model.predict(X_test)

            score = r2_score(y_test, y_preds)

            report[model_name] = [model, score]
        return report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(path):
    try:
        with open(path, "rb") as file_obj:
            data = pickle.load(file_obj)
        return data
    except Exception as e:
        raise CustomException(e, sys)
        