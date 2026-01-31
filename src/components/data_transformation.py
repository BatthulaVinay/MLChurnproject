import os
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
        # Define feature groups based on notebook analysis
        self.numeric_features = [
            "Account length",
            "Total day minutes",
            "Total eve minutes",
            "Total night minutes",
            "Total intl minutes",
            "Customer service calls",
            "Number vmail messages"
        ]
        
        self.binary_features = [
            "International plan",
            "Voice mail plan"
        ]
        
        self.categorical_features = [
            "Area code"
        ]
        
        self.drop_columns = [
            "State",
            "Total day charge",
            "Total eve charge",
            "Total night charge",
            "Total intl charge"
        ]
        
        self.target_column = "Churn"

    def get_data_transformer_object(self):
        """
        Create preprocessing pipeline for features
        
        Returns:
            ColumnTransformer: Preprocessor object with transformers for different feature types
        """
        try:
            logging.info("Creating preprocessing pipeline")
            
            # Numeric features: StandardScaler
            numeric_transformer = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            logging.info("Numeric transformer: StandardScaler")
            
            # Categorical features: OneHotEncoder
            categorical_transformer = Pipeline(steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            logging.info("Categorical transformer: OneHotEncoder")
            
            # Binary features: passthrough (no transformation)
            logging.info("Binary features: passthrough (no transformation)")
            
            # Combine all transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, self.numeric_features),
                    ("cat", categorical_transformer, self.categorical_features),
                    ("bin", "passthrough", self.binary_features)
                ],
                remainder="drop"
            )
            
            logging.info("Preprocessing pipeline created successfully")
            return preprocessor
            
        except Exception as e:
            logging.error(f"Error creating data transformer object: {str(e)}")
            raise CustomException(e, __file__)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Apply data transformation on train and test data
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to testing data
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, preprocessor_obj_path)
        """
        try:
            logging.info("Loading train and test data")
            
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Drop unnecessary columns
            logging.info(f"Dropping columns: {self.drop_columns}")
            train_df = train_df.drop(columns=self.drop_columns)
            test_df = test_df.drop(columns=self.drop_columns)
            
            # Separate target variable
            logging.info(f"Separating target variable: {self.target_column}")
            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]
            
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            # Encode binary features (convert 'Yes'/'No' to 1/0)
            logging.info("Encoding binary features")
            for col in self.binary_features:
                X_train[col] = (X_train[col] == 'Yes').astype(int)
                X_test[col] = (X_test[col] == 'Yes').astype(int)
            
            logging.info("Binary features encoded")
            
            # Create preprocessing pipeline
            logging.info("Creating preprocessing pipeline")
            preprocessor = self.get_data_transformer_object()
            
            # Fit and transform training data
            logging.info("Fitting preprocessor on training data")
            X_train_scaled = preprocessor.fit_transform(X_train)
            
            # Transform test data
            logging.info("Transforming test data")
            X_test_scaled = preprocessor.transform(X_test)
            
            logging.info(f"X_train_scaled shape: {X_train_scaled.shape}")
            logging.info(f"X_test_scaled shape: {X_test_scaled.shape}")
            
            # Encode target variable
            logging.info(f"Encoding target variable")
            y_train_encoded = (y_train == 'Yes').astype(int)
            y_test_encoded = (y_test == 'Yes').astype(int)
            
            logging.info(f"Target variable encoded")
            logging.info(f"Training set - Churn distribution: {y_train_encoded.value_counts().to_dict()}")
            logging.info(f"Test set - Churn distribution: {y_test_encoded.value_counts().to_dict()}")
            
            # Save preprocessor object
            os.makedirs(os.path.dirname(self.transformation_config.preprocessor_obj_file_path), exist_ok=True)
            
            with open(self.transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            logging.info(f"Preprocessor saved to {self.transformation_config.preprocessor_obj_file_path}")
            
            logging.info("Data transformation completed successfully")
            
            return (
                X_train_scaled,
                X_test_scaled,
                y_train_encoded,
                y_test_encoded,
                self.transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, __file__)


if __name__ == "__main__":
    # Run data transformation
    from src.components.data_ingestion import DataIngestion
    
    data_ingestion = DataIngestion()
    raw_path, train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
        train_path, 
        test_path
    )
    
    print(f"Data transformation completed:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Preprocessor saved to: {preprocessor_path}")
