# Kocfinans Loan Approval Prediction API

This project provides a FastAPI-based web service for predicting loan approval and retraining a machine learning model. The API uses a LightGBM model to predict whether a loan application will be approved based on various input features.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Folder Structure](#folder-structure)
- [Running with Docker](#running-with-docker)
- [Configuration](#configuration)
- [Logging](#logging)
- [Deliverables](#deliverables)
- [License](#license)

## Introduction

For financial companies, it is of utmost importance to check their customers for eligibility to loan. In order to speed up the decision-making process, utilization of machine learning models is much needed.

This project is aimed at training and deploying a machine learning model to predict loan approval status based on clients’ financial data. The project consists of two main tasks:
1. Train and evaluate a machine learning model.
2. Develop a REST API to serve the model for production use.

## Features

- Predict loan approval status.
- Retrain the model with new data.
- Save feature importance graphs for model interpretation.

## Setup

### Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)
- Docker (for packaging the application)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/esmaascioglu/loan-approval-prediction-case-study.git
    cd loan-approval-prediction-case-study
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure the data file `loan_approval_dataset_updated.csv` is in the `data` directory.

## Usage

### Running the API

To run the FastAPI server, use the following command:
```sh
uvicorn api:app --reload
```

The server will start at `http://0.0.0.0:8000`.

## API Endpoints

### Health Check

- **GET /**
    - Summary: Returns 200 for health check.
    - Response: 
        ```json
        {
            "status": 200,
            "success": True,
            "still": "alive"
        }
        ```

### Predict Loan Approval

- **POST /api/V1/predict**
    - Summary: Predict loan approval status based on input data.
    - Request Body:
        ```json
        {
            "no_of_dependents": 2,
            "education": 1,
            "self_employed": 0,
            "income_annum": 9600000,
            "loan_amount": 29900000,
            "loan_term": 12,
            "cibil_score": 778,
            "residential_assets_value": 2400000,
            "commercial_assets_value": 17600000,
            "luxury_assets_value": 22700000,
            "bank_asset_value": 8000000
        }
        ```
    - Response: 
        ```json
        {
            "loan_approval_score": 0.87,
            "loan_approval_status": "Approved"
        }
        ```

### Retrain Model

- **POST /api/V1/retrain**
    - Summary: Retrain the model with new data.
    - Response:
        ```json
        {
            "success": True,
            "message": "Model retrained and saved successfully",
            "model_path": "models/retrained_model_2024-08-05.pkl",
            "split_plot_path": "assets/feature_importance_split_2024-08-05.png",
            "gain_plot_path": "assets/feature_importance_gain_2024-08-05.png"
        }
        ```

## Folder Structure

```
loan-approval-prediction-case-study/
│
├── data/
│   └── loan_approval_dataset_updated.csv
│
├── logs/
│   └── app.log
│
├── models/
│   └── trained_model.pkl
│
├── src/
│   ├── api.py
│   ├── config.py
│   ├── datamodel.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── lightgbm_model.py
│   │   ├── preprocessing.py
│   └── assets/
│       ├── feature_importance_split_2024-08-05.png
│       └── feature_importance_gain_2024-08-05.png
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## Running with Docker

You can also run the application using Docker.

### Pull the Docker Image

```sh
docker pull esmaascioglu/loan-prediction-api:latest
```

### Run the Docker Container

```sh
docker run -d -p 8000:8000 esmaascioglu/loan-prediction-api:latest
```

The server will start at `http://0.0.0.0:8000`.

## Configuration

Configuration settings can be found in `config.py`. Key settings include:

- Data paths
- Model parameters
- Asset directory for saving feature importance graphs

## Logging

Logging is configured to provide detailed information about the API's operation. Logs include timestamps, log levels, and messages. Logs are printed to the console and saved to `logs/app.log`.

## Deliverables

### Task 1: Model Training and Evaluation

1. **Jupyter Notebook**: A notebook (.ipynb) that clearly shows the data analysis, feature engineering, model training, and evaluation steps.
2. **Model Dump**: The final trained model saved using Pickle or Joblib for deployment.

### Task 2: Model Deployment

1. **Python Application**: A Python application that includes the model dump, Dockerfile, and `requirements.txt` files as well as a `README.md` file.
2. **Docker**: Package the application using Docker for deployment on a production server.

All work should be committed to a GitHub repository and the repository link should be provided.

## License

This project is licensed under the MIT License.
```
