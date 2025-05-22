import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            numerical_columns = ["reading_score","writing_score"]

            categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder())])
            numerical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])

            preprocessor = ColumnTransformer([("numerical_pipeline", numerical_pipeline, numerical_columns),("categorical_pipeline", categorical_pipeline, categorical_columns)])
            logging.info("preprocessor is created")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("trsin and test data read succesfully")

            preprocessor = self.get_data_transformer_object()
            logging.info("preprocessor obtained")

            target_column = "math_score"
            train_input = train_df.drop(target_column, axis=1)
            train_output = train_df[target_column]

            test_input = test_df.drop(target_column, axis=1)
            test_output = test_df[target_column]
            logging.info("splitting done succesfully")

            train_input_arr = preprocessor.fit_transform(train_input)
            test_input_arr = preprocessor.transform(test_input)
            logging.info("Preprocessing done successfully")

            train_arr = np.c_[train_input_arr, np.array(train_output)]
            test_arr = np.c_[test_input_arr, np.array(test_output)]
            logging.info("preprocessed and joined")

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
