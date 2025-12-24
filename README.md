# Auto Insurance Claims Prediction & Risk Analysis

## Problem Description
Auto insurance companies face a critical challenge: accurately assessing claim risk to balance profitability, customer fairness, and fraud prevention. Incorrect risk estimation can lead to:

- Underpricing policies, resulting in financial losses
- Overpricing low-risk customers, leading to churn
- Inefficient claim handling, increasing operational costs

This project focuses on analyzing historical auto insurance policy and claim data to predict the likelihood of insurance claims and identify key risk drivers behind claims.

Using the Auto Insurance Claims dataset, the goal is to build an end-to-end machine learning pipeline that transforms raw insurance data into a deployable predictive service that can support underwriting and risk management decisions.

## Project Objectives

1. Understand customer and policy characteristics that contribute to insurance claims
2. Perform exploratory data analysis (EDA) to identify patterns and anomalies
3. Build and compare multiple classification models
4. Select and tune the best-performing model
5. Deploy the model as a web-based prediction service

# Data
The data was obtained from [Kaggle](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)

For a project, you repository/folder should contain the following:

- `README.md` with
    - Description of the problem
    - Instructions on how to run the project
- Data
    - You should either commit the dataset you used or have clear instructions how to download the dataset
- Notebook (suggested name - `notebook.ipynb`) with
    - Data preparation and data cleaning
    - EDA, feature importance analysis
    - Model selection process and parameter tuning
- Script `train.py` (suggested name)
    - Training the final model
    - Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML)
- Script `predict.py` (suggested name)
    - Loading the model
    - Serving it via a web service (with Flask or specialized software - BentoML, KServe, etc)
- Files with dependencies
    - `Pipenv` and `Pipenv.lock` if you use Pipenv
    - or equivalents: conda environment file, requirements.txt or pyproject.toml
- `Dockerfile` for running the service
- Deployment
    - URL to the service you deployed or
    - Video or image of how you interact with the deployed service