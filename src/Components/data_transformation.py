import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exceptions import CustomException
from src.logger import logging
from src.utilis import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_filepth = os.path.join("artifacts","preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation. 
        
        '''
        try:
            numerical_feat = ["writing_score","reading_score"]
            categorical_feat = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            #Creating pipelines to run on the training dataset, and transform in the test dataset.
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")), #To handle missing values
                    ("scaler",StandardScaler())
                ]  
            )
            category_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")), #Handling Missing Values
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Numerical Columns: {numerical_feat}")
            logging.info(f"Categorical Columns: {categorical_feat}")
            
            #To combine both num_pipeline and the category_pipeline pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_feat),
                    ("categorical_pipeline", category_pipeline, categorical_feat)
                ]
            )
            
            return preprocessor            
            
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_datatransformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read Train and Test data completed.")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_col_name = "math_score"
            numerical_col = ["writing_score","reading_score"]
            
            input_feature_train_df = train_df.drop(columns = [target_col_name], axis = 1)
            target_feature_train_df = train_df[target_col_name]
            
            input_feature_test_df = test_df.drop(columns = [target_col_name], axis = 1)
            target_feature_test_df = test_df[target_col_name]
            
            logging.info(f"Applying preprcocessing obj on training dataframe and testing dataframe.")
        
            inp_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            inp_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                inp_feature_train_arr, np.array(target_feature_train_df)]
            #np.c is a short-hand utility object to concatenate arrays along the second axis.
            
            test_arr = np.c_[
                inp_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object")
            
            save_object(
                file_path = self.data_tranformation_config.preprocessor_ob_filepth,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_ob_filepth
            )
            
        except Exception as e:
            raise CustomException(e,sys)