# 📊 Customer Churn Prediction (End-to-End ML Project)
## 🚀 Project Overview
Customer churn is one of the most important problems for subscription-based businesses.
This project predicts whether a customer will churn (leave the service) or not using Logistic Regression.

## 🗂️ Dataset
Telco Customer Churn Dataset from Kaggle
Target variable: Churn (Yes/No)

## ⚙️Tech Stack

Python 🐍
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
Model: Logistic Regression
Deployment: Streamlit

## 📌Workflow

### Data Preprocessing
Handle missing values
Encode categorical variables
Normalize numerical features
Train-test split
### Model Training
Logistic Regression model
Evaluation metrics: Accuracy, Precision, Recall, F1-score
### Model Visualization
ROC Curve & AUC Score
Feature Importance from Logistic Regression coefficients
### Model Saving
Save trained model (.pkl)
Save feature list for deployment
### Deployment
Streamlit app for real-time prediction
User inputs customer details → Model predicts churn

# 📊 Results
Logistic Regression Accuracy: ~78%
ROC AUC Score: ~0.82
Key features influencing churn: Tenure, Contract Type, Internet Service, Monthly Charges, Payment Method
