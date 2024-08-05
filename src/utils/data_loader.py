import pandas as pd
import logging
from io import StringIO


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, file_path, encoding='latin1', sep=';'):
        self.file_path = file_path
        self.encoding = encoding
        self.sep = sep
        self.data = None
        self.missing_percentages = None
        self.statistics = None
        self.loan_status_distribution = None

    def load_data(self):
        logging.info("Loading data from file: %s", self.file_path)
        self.data = pd.read_csv(self.file_path, encoding=self.encoding, sep=self.sep)
        self.data.columns = [col_name.strip() for col_name in self.data.columns]
        logging.info("Data loaded successfully")
        self.store_info()

    def store_info(self):
        self.store_shape()
        self.store_overview()
        self.store_info_summary()
        self.store_missing_values()
        self.store_statistics()
        self.store_loan_status_distribution()

    def store_shape(self):
        logging.info("Dataset shape: %s", self.data.shape)

    def store_overview(self):
        logging.info("Data Overview:\n%s", self.data.head().to_markdown(index=False, numalign='left', stralign='left'))

    def store_info_summary(self):
        buffer = StringIO()
        self.data.info(buf=buffer)
        info_str = buffer.getvalue()
        logging.info("Data Info:\n%s", info_str)

    def store_missing_values(self):
        self.missing_percentages = (self.data.isnull().sum() / len(self.data)) * 100
        logging.info("Missing Values (Percentages):\n%s", self.missing_percentages.to_markdown(numalign='left', stralign='left'))

    def store_statistics(self):
        self.statistics = self.data.describe()
        logging.info("Descriptive Statistics:\n%s", self.statistics.to_markdown(numalign='left', stralign='left'))

    def store_loan_status_distribution(self):
        self.loan_status_distribution = self.data['loan_status'].value_counts()
        logging.info("Loan Status Distribution:\n%s", self.loan_status_distribution.to_markdown(numalign='left', stralign='left'))
