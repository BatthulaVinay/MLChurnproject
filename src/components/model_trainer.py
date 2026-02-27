import os
import sys
from dataclasses import dataclass
from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training and evaluation")
            
            scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
            
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    max_depth=10,
                    random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    random_state=42,
                    class_weight="balanced"
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42
                )
            }

            # Evaluate all models (returns model_name -> f1_score)
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance")
            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            
            logging.info(f"Best Model: {best_model_name} with F1 Score: {best_model_score}")
            
            # Get model instance
            best_model = models[best_model_name]
            
            # Fit on full training data
            best_model.fit(X_train, y_train)
            
            # Cross-validation on training data
            cv_score = cross_val_score(
            best_model,
            X_train,
            y_train,
            cv=5,
            scoring="f1"
            ).mean()

            logging.info(f"Cross-validated F1 score: {cv_score}")

            # Final test prediction
            predicted = best_model.predict(X_test)
            final_f1 = f1_score(y_test, predicted)

            logging.info(f"Final Test F1 Score: {final_f1}")

            # Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Final Test F1 Score: {final_f1}")
            return final_f1

        except Exception as e:
            logging.error("Error during model training")
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

