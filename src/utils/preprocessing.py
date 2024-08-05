import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def preprocess(self):
        logging.info("Starting preprocessing of data.")
        self.map_loan_status()
        self.map_self_employed()
        self.map_education()
        self.impute_missing_values()

        logging.info("Performing train-test-validation split.")
        return self.train_test_val_split()


    def map_loan_status(self):
        self.data['loan_status'] = self.data['loan_status'].map({' Approved': 1, ' Rejected': 0})
    
    def map_self_employed(self):
        self.data['self_employed'] = self.data['self_employed'].map({' Yes': 1, ' No': 0})

    def map_education(self):
        self.data['education'] = self.data['education'].map({' Graduate': 1, ' Not Graduate': 0})

    def impute_missing_values(self):
        self.data['education'].fillna(self.data['education'].mode()[0], inplace=True)
        self.data['City'].fillna(self.data['City'].mode()[0], inplace=True)

    def train_test_val_split(self):
        
        X = self.data[self.config.train_cols].drop(columns=[self.config.target])
        y = self.data[self.config.target]
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=self.config.seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.config.seed)
        
        logging.info("Train, validation, and test sets created.")
        return X_train, X_val, X_test, y_train, y_val, y_test
