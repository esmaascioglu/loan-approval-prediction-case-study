# Kocfinans Loan Approval Prediction API

This case study project provides a FastAPI-based web service for predicting loan approval of a client and retraining a machine learning model. The API uses a LightGBM model to predict whether a loan application will be approved based on various input features.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
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

- Predict loan approval status of a client. 
- Retrain the model with new data.

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
