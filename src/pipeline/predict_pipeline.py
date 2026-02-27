import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        

    def predict(self, features: pd.DataFrame):
        """
        Perform prediction on new data

        Args:
            features (pd.DataFrame): Input features

        Returns:
            np.ndarray: Predictions
        """
        try:
            logging.info("Loading preprocessor and model")

            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            
            required_columns = preprocessor.feature_names_in_
            missing_cols = set(required_columns) - set(features.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in input data: {missing_cols}")
            
            logging.info("Transforming input features")
            data_transformed = preprocessor.transform(features)

            logging.info("Generating predictions")
            probs = model.predict_proba(data_transformed)[:, 1]

            threshold = 0.3  # or configurable
            predictions = (probs >= threshold).astype(int)

            return predictions, probs
        
        except Exception as e:
            logging.error("Error during prediction")
            raise CustomException(e, sys)


class CustomData:
    """
    Converts raw user input into a DataFrame
    """

    def __init__(
        self,
        account_length: int,
        total_day_minutes: float,
        total_eve_minutes: float,
        total_night_minutes: float,
        total_intl_minutes: float,
        customer_service_calls: int,
        number_vmail_messages: int,
        total_day_calls: int,
        total_eve_calls: int,
        total_night_calls: int,
        total_intl_calls: int,
        international_plan: str,
        voice_mail_plan: str,
        area_code: int
    ):
        self.account_length = account_length
        self.total_day_minutes = total_day_minutes
        self.total_eve_minutes = total_eve_minutes
        self.total_night_minutes = total_night_minutes
        self.total_intl_minutes = total_intl_minutes
        self.customer_service_calls = customer_service_calls
        self.number_vmail_messages = number_vmail_messages
        self.total_day_calls = total_day_calls
        self.total_eve_calls = total_eve_calls
        self.total_night_calls = total_night_calls
        self.total_intl_calls = total_intl_calls
        self.international_plan = international_plan.title()
        self.voice_mail_plan = voice_mail_plan.title()
        self.area_code = area_code

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "Account length": [self.account_length],
                "Total day minutes": [self.total_day_minutes],
                "Total eve minutes": [self.total_eve_minutes],
                "Total night minutes": [self.total_night_minutes],
                "Total intl minutes": [self.total_intl_minutes],
                "Customer service calls": [self.customer_service_calls],
                "Number vmail messages": [self.number_vmail_messages],
                "Total day calls": [self.total_day_calls],
                "Total eve calls": [self.total_eve_calls],
                "Total night calls": [self.total_night_calls],
                "Total intl calls": [self.total_intl_calls],
                "International plan": [self.international_plan],
                "Voice mail plan": [self.voice_mail_plan],
                "Area code": [self.area_code]
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
