

import numpy as np
import pandas as pd
import logging
import joblib
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LightGBMModel:
    def __init__(self, params, n_tree=500):
        self.params = params
        self.n_tree = n_tree
        self.model = None
        self.feature_importances_ = None

    def train(self, X_train, y_train, X_val, y_val):

        import lightgbm as lgb

        logging.info("Starting training process.")
        dtrain = lgb.Dataset(X_train, y_train, free_raw_data=True)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_tree,
            callbacks=[lgb.log_evaluation(period=10), lgb.early_stopping(5)],
            valid_sets=(dtrain, dval),
        )

        logging.info("Training completed. Generating predictions.")
        val_preds = self.model.predict(X_val).round(2)
        self.evaluate_model(y_val, val_preds)
        self.feature_importances_ = self.feature_importances()

        logging.info("Model training and evaluation completed.")
        return self.model, self.feature_importances_

    def evaluate_model(self, y_test, prediction):

        from sklearn.metrics import (
                                    accuracy_score,
                                    f1_score,
                                    recall_score,
                                    precision_score,
                                    roc_auc_score,
                                    classification_report,
                                )
        
        logging.info("Evaluating model performance.")
        accuracy = np.round(accuracy_score(y_test, np.round(prediction)))
        f1 = f1_score(y_test, np.round(prediction)).round(4)
        precision = precision_score(y_test, np.round(prediction)).round(4)
        recall = recall_score(y_test, np.round(prediction)).round(4)
        roc_auc = roc_auc_score(y_test, prediction).round(4)

        logging.info("Accuracy: %s", accuracy)
        logging.info("F1: %s", f1)
        logging.info("Precision: %s", precision)
        logging.info("Recall: %s", recall)
        logging.info("ROC: %s", roc_auc)
        
        print("\n")
        print("\t\t\t Evaluation")
        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"ROC: {roc_auc}")
        print("\t\t\t Classification Report")
        print(classification_report(y_test, np.round(prediction)))

    def feature_importances(self):
        if self.model is None:
            logging.error("Attempted to retrieve feature importances before model training.")
            raise ValueError("Model has not been trained yet.")
        
        logging.info("Retrieving feature importances.")
        fi_df = pd.DataFrame(
            zip(
                self.model.feature_name(),
                self.model.feature_importance(importance_type="gain"),
                self.model.feature_importance(importance_type="split"),
            ),
            columns=["feature", "gain", "split"],
        ).sort_values("split", ascending=False).set_index("feature")
        
        logging.info("Feature importances retrieved successfully.")
        return fi_df
    
    def save_model(self, file_path):

        import joblib

        if self.model is None:
            logging.error("Attempted to save the model before training.")
            raise ValueError("Model has not been trained yet.")
        
        logging.info("Saving the model to %s", file_path)
        joblib.dump(self.model, file_path)
        logging.info("Model saved successfully to %s", file_path)


class ModelManager:

    model = None
    model_path = None

    @staticmethod
    def load_model(model_path: str):
        if ModelManager.model is None or ModelManager.model_path != model_path:
            try:
                ModelManager.model = joblib.load(model_path)
                ModelManager.model_path = model_path
                logging.info("Model loaded successfully from %s", model_path)
            except Exception as e:
                logging.error("Error loading model: %s", e)
                raise HTTPException(status_code=500, detail="Model could not be loaded")
        return ModelManager.model, ModelManager.model.feature_name()
