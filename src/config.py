import os
from pathlib import Path

class Settings:
    def __init__(self):

        # file paths
        self.data_path = Path('./data/loan_approval_dataset_updated.csv')
        self.model_dir = Path('./models')
        self.assets_dir = Path('./assets')
        self.model_path = self.model_dir /  "trained_model_2024-08-04.pkl"
        self.logs_dir = Path('./logs/app.log')

        # general
        self.seed = 2041
        self.target = 'loan_status'
        self.train_cols = [
            'no_of_dependents',
            'education',
            'self_employed',
            'income_annum',
            'loan_amount',
            'loan_term',
            'loan_status',
            'cibil_score',
            'residential_assets_value',
            'commercial_assets_value',
            'luxury_assets_value',
            'bank_asset_value'
        ]

        # model parameters
        self.params = {
            "boosting_type": "gbdt",
            "feature_fraction": 0.5,
            "learning_rate": 0.1,
            "max_depth": -1,
            "metric": "binary_logloss",
            "n_jobs": -1,
            "num_leaves": 31,
            "objective": "binary",
            "random_state": self.seed,
            "bagging_fraction": 0.5,
            "verbosity": -1,
            "subsample_freq": 5,
            "bagging_seed": self.seed,
        }

