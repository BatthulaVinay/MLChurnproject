import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             precision_score, 
                             recall_score,
                             classification_report)
from xgboost import XGBClassifier
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Evaluating models")
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBoost": XGBClassifier()
            }
            
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            
            # Find the best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            # Add a threshold check
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance")

            logging.info(f"Best model found: {best_model_name} with F1: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            # Use a different name for the variable than the function
            final_f1 = f1_score(y_test, predicted)
            
            return final_f1

        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    from src.components.data_ingestion import DataIngestion
    
    # Data Ingestion
    data_ingestion = DataIngestion()
    raw_path, train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    # Data Transformation
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
        train_path, test_path,
    )
    
    # Model Training
    model_trainer = ModelTrainer()
    f1 = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
    
    print(f"Model training completed with f1 score: {f1}")

