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

## Installation

### Clone the Repository
```bash
git https://github.com/Fuady/fraud_detection_car_insurance.git
cd insurance-fraud-detection
```

### Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Model Training

The training script:
- Preprocesses numerical & categorical features
- Trains multiple classifiers
- Evaluates performance
- Selects the best model
- Saves the trained model to disk

Model interpretability was conducted using SHAP with TreeExplainer for ensemble models to ensure stable and consistent feature attribution.

Feature importance was computed using SHAP directly on the transformed feature space extracted from the fitted pipeline, ensuring alignment with model inputs.

The final fraud detection model was trained using a LightGBM classifier within a preprocessing pipeline and persisted using pickle for deployment and reproducibility.

Deployed a fraud detection model as a RESTful API using Flask, enabling real-time predictions with probabilistic outputs.

### Run Training
```bash
python train.py
```

### Output
```bash
models/fraud_model.pkl
```

## Model Serving (REST API)
The trained model is exposed via a **Flask REST API**.

### Start the API
```
python predict.py
```
then you will see the following, which indicates that the API is ready
```
* Serving Flask app 'predict'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.7:5000
Press CTRL+C to quit
 * Restarting with watchdog (windowsapi)
 * Debugger is active!
 * Debugger PIN: 307-138-162
```

Then you can run this python notebook: `05-fraud-test.ipynb`.

With single example `data_1`, the results are the following
```
{'Fraud': True, 'Fraud Probability': 0.6519783313945177}
```




## Containerization with Docker

### Build Docker Image
```
docker build -t insurance-fraud-api .
```

### Run the Container
```
docker run -p 5000:5000 insurance-fraud-api
```
API will be available at:
```
http://localhost:5000
```


## Test the API (Docker)
```
curl http://localhost:5000/
```
```
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "data": [
             {
               "age": 50,
               "incident_severity": "Minor Damage",
               "total_claim_amount": 3000
             }
           ]
         }'
```