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

The project includes a fully reproducible training pipeline with dependency management, model persistence, and clear execution instructions.

## Data
The data was obtained from [Kaggle](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data). You can download directly from the website, or follow this script:

```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("buntyshah/auto-insurance-claims-data")

print("Path to dataset files:", path)
```


## Solution Approach

1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Model Training & Comparison
4. Model Evaluation & Overfitting Detection
5. Hyperparameter Tuning (GridSearchCV, 10-fold CV)
6. Model Explainability (SHAP)
7. Model Serialization
8. REST API Deployment
9. Containerization using Docker

## Project Structure
``` kotlin
project/
│
├── data/
│   └── insurance_claims.csv
│
├── models/
│   └── fraud_model.pkl
│
├── train.py
├── predict.py
├── notebook.ipynb
├── Dockerfile
├── requirements.txt
└── README.md
```

Model interpretability was conducted using SHAP with TreeExplainer for ensemble models to ensure stable and consistent feature attribution.

Feature importance was computed using SHAP directly on the transformed feature space extracted from the fitted pipeline, ensuring alignment with model inputs.

The final fraud detection model was trained using a LightGBM classifier within a preprocessing pipeline and persisted using pickle for deployment and reproducibility.

Deployed a fraud detection model as a RESTful API using Flask, enabling real-time predictions with probabilistic outputs.

