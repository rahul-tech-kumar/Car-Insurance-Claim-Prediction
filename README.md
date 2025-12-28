# 🚗 Car Insurance Claim Prediction – End-to-End ML System

## 📌 Project Overview
This project builds an end-to-end production-grade MLOps system that predicts whether a customer will raise a car insurance claim.
It covers data preprocessing, model training, hyperparameter tuning, MLflow experiment tracking, model registry, and Streamlit deployment.

## 🎯 Business Problem
Insurance companies lose millions due to fraudulent or high-risk claims.
The goal is to predict claim probability so the company can:

-  Detect risky customers
-  Optimize premiums
-  Reduce financial loss
-  Improve underwriting decisions


## ⚙️ Tech Stack

- Language :       Python
- ML :             Scikit-Learn, XGBoost
- Data :	           Pandas, NumPy
- MLflow 
- Visualization :	   Seaborn, Plotly
- Deployment :	     Streamlit
- Model Registry :	 MLflow Model Registry

## 🏗 Architecture
train.csv ──► Preprocessing ──► ML Models ──► MLflow Tracking
                                      │
                                      ▼
                               Model Registry
                                      │
                                      ▼
                                Streamlit App

## 🚀 Key Features
| Feature                  | Description                               |
| ------------------------ | ----------------------------------------- |
| MLflow Tracking          | Logs metrics, parameters, artifacts       |
| Model Registry           | Versioned production models               |
| Hyperparameter Tuning    | RandomizedSearchCV                        |
| Class Imbalance Handling | Balanced class weights & scale_pos_weight |
| Streamlit UI             | Prediction + Model Monitoring dashboard   |
| Model Comparison         | ROC-AUC / F1 / Accuracy tracking          |

## 🌐 Live Deployment
🚀 **Streamlit Cloud App:**  
https://car-insurance-claim-prediction-hxky7nm5tvmt2s4ngxpjxa.streamlit.app/

# Screenshots
## Model monitor
<img width="1920" height="1080" alt="Screenshot (185)" src="https://github.com/user-attachments/assets/0cc23c34-e87b-4cf3-85d0-271e0efad2b6" />
## Data Explorer
<img width="1920" height="1080" alt="Screenshot (186)" src="https://github.com/user-attachments/assets/cc9dfc44-cf1f-4cc7-a4a1-5920dfae9ee7" />
## Prediction
<img width="1920" height="1080" alt="Screenshot (187)" src="https://github.com/user-attachments/assets/02df713b-9a85-4f2f-84f3-0d95cba23e50" />


## 👨‍💻 Author
- Rahul Kumar
- B.Tech IT – IIEST Shibpur
- Aspiring Data Scientist / ML Engineer

