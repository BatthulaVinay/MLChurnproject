import os
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

# Top-level function to replace lambda for binary encoding

def yes_no_to_int(x):
    return (x == "Yes").astype(int)

def encode_target(y):
    if y.dtype == bool:
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    y_clean = (
        y.astype(str)
         .str.strip()
         .str.lower()
    )

    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
    y_mapped = y_clean.map(mapping)

    if y_mapped.isna().any():
        bad_values = y[y_mapped.isna()].unique()
        raise ValueError(f"Unexpected target labels found: {bad_values}")

    return y_mapped.astype(int)


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
        # Feature groups
        self.numeric_features = [
            "Account length",
            "Total day minutes",
            "Total eve minutes",
            "Total night minutes",
            "Total intl minutes",
            "Customer service calls",
            "Number vmail messages",
            "Total day calls",
            "Total eve calls",
             "Total night calls",
             "Total intl calls"
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
        """Create preprocessing pipeline for features"""
        
        try:
            logging.info("Creating preprocessing pipeline")
            
            # Numeric: StandardScaler
            numeric_transformer = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            
            
            categorical_transformer = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
            
            # Binary: top-level FunctionTransformer
            binary_transformer = Pipeline(steps=[
                ("binary_encode", FunctionTransformer(yes_no_to_int, validate=False))
            ])
            
            # Combine all transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, self.numeric_features),
                    ("cat", categorical_transformer, self.categorical_features),
                    ("bin", binary_transformer, self.binary_features)
                ],
                remainder="drop"
            )
            
            logging.info("Preprocessing pipeline created successfully")
            return preprocessor
        
        except Exception as e:
            logging.error(f"Error creating data transformer object: {str(e)}")
            raise CustomException(e, __file__)

    def initiate_data_transformation(self, train_path, test_path):
        """Apply preprocessing on train and test data"""
        
        try:
            logging.info("Loading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            
            # Drop unneeded columns
            train_df = train_df.drop(columns=self.drop_columns)
            test_df = test_df.drop(columns=self.drop_columns)
            
            # Separate target
            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]
            
        
            # Preprocessor
            preprocessor = self.get_data_transformer_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            
            # Encode target
            y_train_encoded = encode_target(y_train)
            y_test_encoded = encode_target(y_test)
            
        
            # Save preprocessor
            os.makedirs(os.path.dirname(self.transformation_config.preprocessor_obj_file_path), exist_ok=True)
            with open(self.transformation_config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            logging.info(f"Preprocessor saved to {self.transformation_config.preprocessor_obj_file_path}")
            logging.info("Data transformation completed successfully")
            
            return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, self.transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, __file__)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    # Run data ingestion
    data_ingestion = DataIngestion()
    raw_path, train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    # Run data transformation
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
        train_path, test_path,
    )
    
    print(f"Data transformation completed:")
 

