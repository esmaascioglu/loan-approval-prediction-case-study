import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, APIRouter, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError

import pandas as pd
from pydantic import BaseModel
from joblib import load
import logging
import os
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from pathlib import Path
from src.config import Settings
from src.utils.lightgbm_model import LightGBMModel, ModelManager
from src.utils.data_loader import DataLoader
from src.utils.preprocessing import DataPreprocessor
from src.datamodel import Data 

config = Settings()

log_file_path = config.logs_dir
log_file_path.parent.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logging.info("Logging setup complete.")

app = FastAPI(title="Kocfinans Loan Approval Prediction API", version="0.1")
router = APIRouter()

model, cols = ModelManager.load_model(config.model_path)

@router.get("/", status_code=200, summary="Returns 200 for healthcheck.", tags=["Root"])
def index():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder(
            {
                "status": status.HTTP_200_OK,
                "success": True,
                "still": "alive",
            }
        ),
    )

@router.post("/api/V1/predict", tags=["predict"])
async def predict(input_data: Data):
    try:
        model, cols = ModelManager.load_model(config.model_path)
        
        input_df = pd.DataFrame([input_data.dict()])[cols]
        prediction = model.predict(input_df)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(
                {
                    "loan_approval_score": round(prediction[0], 4),
                    "loan_approval_status": "Approved" if prediction[0] > 0.5 else "Rejected",
                }
            ),
        )
    except Exception as e:
        logging.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction could not be made")

@router.post("/api/V1/retrain", tags=["retrain"])
async def retrain():
    from datetime import datetime
    import matplotlib.pyplot as plt

    try:
        data_loader = DataLoader(config.data_path)
        data_loader.load_data()
        preprocessor = DataPreprocessor(data_loader.data, config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess()

        lgbm_model = LightGBMModel(params=config.params)
        model, feature_importances = lgbm_model.train(X_train, y_train, X_val, y_val)
        
        # Save the new model
        today_date = datetime.today().strftime('%Y-%m-%d')
        model_path = config.model_dir / f"retrained_model_{today_date}.pkl"
        lgbm_model.save_model(model_path)

        # Reload the model
        ModelManager.load_model(model_path)

        # Save feature importance graphs
        assets_dir = config.assets_dir
        assets_dir.mkdir(parents=True, exist_ok=True)

        split_plot_path = assets_dir / f"feature_importance_split_{today_date}.png"
        feature_importances.reset_index().sort_values("split").plot(
            kind="barh", x="feature", y="split", title="Feature Importance (split)", figsize=(10, 6)
        )
        plt.savefig(split_plot_path)
        plt.close()

        gain_plot_path = assets_dir / f"feature_importance_gain_{today_date}.png"
        feature_importances.reset_index().sort_values("gain").plot(
            kind="barh", x="feature", y="gain", title="Feature Importance (gain)", figsize=(10, 6)
        )
        plt.savefig(gain_plot_path)
        plt.close()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder(
                {
                    "success": True,
                    "message": "Model retrained and saved successfully",
                    "model_path": str(model_path),
                    "split_plot_path": str(split_plot_path),
                    "gain_plot_path": str(gain_plot_path),
                }
            ),
        )
    except Exception as e:
        logging.error("Retraining error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Model retraining failed")

app.include_router(router)

def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=jsonable_encoder({"detail": "Not Found"}),
    )

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(404, not_found_exception_handler)

if __name__ == "__main__":
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
