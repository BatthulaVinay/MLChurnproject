import os
import sys
import pickle
from sklearn.metrics import f1_score

from src.logger import logging
from src.exception import CustomException


def save_object(file_path: str, obj):
    """
    Save any Python object using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load a pickled Python object
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Train and evaluate multiple models using F1 score

    Returns:
        dict: {model_name: f1_score}
    """
    try:
        model_report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Fit model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate
            f1 = f1_score(y_test, y_pred)
            model_report[model_name] = f1

            logging.info(f"{model_name} F1 Score: {f1:.4f}")

        return model_report

    except Exception as e:
        raise CustomException(e, sys)
