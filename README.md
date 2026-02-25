## Final_projectCS439

###Fraudulent Transaction Detection System

A Real-World Machine Learning Project for Financial Fraud Analytics

###Overview

This project focuses on building an intelligent machine learning system that can detect fraudulent e-commerce transactions using behavioral and transactional data. In today’s digital world, online payments are everywhere, and fraud detection has become a critical cybersecurity and business problem. The goal of this project was to experiment with multiple models and identify the most reliable approach for detecting fraud with strong recall and balanced performance.
Instead of just aiming for high accuracy, I focused on something more meaningful: correctly identifying fraudulent cases without overwhelming businesses with false alarms. This mindset shift helped me approach the problem more realistically, like a real data science task.

###Project Goals

- Build a predictive ML pipeline to classify transactions as fraudulent or legitimate
- Compare multiple supervised learning models
- Address real-world challenges such as class imbalance and noisy data
- Perform feature engineering to capture behavioral fraud patterns
- Deploy the model inside an interactive web application for real-time predictions
- This project reflects a full end-to-end data science workflow, from data preprocessing to model deployment.

###Dataset

I used a real-world Fraudulent E-Commerce Transaction Dataset from Kaggle and later merged it with a credit card fraud dataset to improve fraud representation and reduce imbalance. This increased the fraud ratio from ~5% to around 6.5%, allowing models to better learn minority class patterns. This merging step became a key turning point in the project because it significantly improved recall and model stability.

###Models Implemented

I experimented with multiple supervised learning models to understand how each handles fraud detection:

- Logistic Regression (baseline interpretable model)
- Decision Tree (nonlinear interpretable model)
- Random Forest (ensemble learning approach)
- XGBoost (advanced gradient boosting model)

All models shared the same preprocessing pipeline using a ColumnTransformer with one-hot encoding for categorical features and numerical passthrough for consistency and fair comparison.

###Experimental Design

- 80/20 stratified train-test split
- Handling class imbalance with class_weight="balanced"
- Feature engineering including:
 - Log-transformed transaction amount
 - Weekend and day-of-week indicators
 - Customer age buckets
 - Account age flags
- Hyperparameter tuning for ensemble models

I evaluated performance mainly using Recall and ROC-AUC, since missing fraudulent transactions is more costly than incorrectly flagging legitimate ones.
