import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_arr, test_Arr):
        try:
            logging.info("initiating model trainer")
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1],test_Arr[:,:-1], test_Arr[:,-1])
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            param_grid = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4],
        # "bootstrap": [True, False]
    },
    "Decision Tree": {
        "max_depth": [None, 10, 20, 30],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error", "friedman_mse", "absolute_error"]
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        # "max_depth": [3, 5, 10],
        # "min_samples_split": [2, 5],
        # "min_samples_leaf": [1, 2]
    },
    "Linear Regression": {
        "fit_intercept": [True, False],
        "positive": [True, False]    # Only available in newer versions of sklearn
    },
    "XGBRegressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        # "subsample": [0.6, 0.8, 1.0],
        # "colsample_bytree": [0.6, 0.8, 1.0]
    },
    "CatBoosting Regressor": {
        # "iterations": [100, 200],
        "learning_rate": [0.01, 0.1],
        "depth": [4, 6, 10],
        "l2_leaf_reg": [1, 3, 5],
        "border_count": [32, 64, 128]
    },
    "AdaBoost Regressor": {
        # "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0],
        "loss": ["linear", "square", "exponential"]
    }
}
            


            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=param_grid)

            best_model_name = list(model_report.keys())[int(max(list(model_report.values())))]
            best_model = model_report[best_model_name]
            best_score = model_report[best_model_name]

            if best_score < 0.6:
                raise CustomException("no good enough model fits on the data",sys)
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info("model initiated and done")
            return model_report[best_model_name]


        except Exception as e:
            raise CustomException(e, sys)