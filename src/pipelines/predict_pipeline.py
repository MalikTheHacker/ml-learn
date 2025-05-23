import sys, os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    
    def __init__(self,gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course,
                 reading_score,writing_score):
        self.data = CustomData( gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course,
                 reading_score,writing_score)
        
    def predict(self):
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        model_path = os.path.join("artifacts", "model.pkl")
        df = self.data.custom_data_to_dataframe()
        preprocessor = load_object(preprocessor_path)
        preprocessed_arr = preprocessor.transform(df)
        model = load_object(model_path)
        result = model.predict(preprocessed_arr)
        if hasattr(result, "__len__"):
            return result[0]
        return result
    


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course,
                 reading_score,writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def custom_data_to_dataframe(self):
        columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
       'test_preparation_course', 'reading_score','writing_score']
        df = pd.DataFrame([[self.gender, self.race_ethnicity, self.parental_level_of_education,
                                self.lunch, self.test_preparation_course, 
                                self.reading_score, self.writing_score]], columns=columns)
        return df
        
        
        
        
if __name__ == "__main__":
    data = PredictPipeline('male', 'group B', 'some college', 'standard', 'completed', 80,90)
    result = data.predict()
    print("final result: ", result)