import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Read data from CSV file and split into train and test sets
        
        Returns:
            tuple: (raw_data_path, train_data_path, test_data_path)
        """
        logging.info("Entered the data ingestion method")
        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/Telecom_churn.csv')
            logging.info(f"Dataset shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")
            
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                logging.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
            else:
                logging.info("No missing values found in dataset")
            
            # Display basic statistics
            logging.info(f"\nDataset Info:")
            logging.info(f"Shape: {df.shape}")
            logging.info(f"\nData Types:\n{df.dtypes}")
            logging.info(f"\nBasic Statistics:\n{df.describe()}")
            
            # Check target variable distribution
            if 'Churn' in df.columns:
                churn_dist = df['Churn'].value_counts()
                logging.info(f"\nTarget Variable (Churn) Distribution:\n{churn_dist}")
                logging.info(f"Churn Rate: {churn_dist.get('Yes', 0) / len(df) * 100:.2f}%")
            
            # Split data into train and test sets
            logging.info("Splitting data into train and test sets (80-20 split)")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df['Churn'] if 'Churn' in df.columns else None
            )
            
            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
            logging.info(f"Train data saved to {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")
            
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, __file__)


if __name__ == "__main__":
    # Run data ingestion
    data_ingestion = DataIngestion()
    raw_path, train_path, test_path = data_ingestion.initiate_data_ingestion()
    print(f"Data ingestion completed:")
    print(f"  Raw data: {raw_path}")
    print(f"  Train data: {train_path}")
    print(f"  Test data: {test_path}")
