## Real-Time Fraudulent Transaction Detection using Ensemble Machine Learning Models

### Fraudulent Transaction Detection System

A Real-World Machine Learning Project for Financial Fraud Analytics

### Overview

This project focuses on building an intelligent machine learning system that can detect fraudulent e-commerce transactions using behavioral and transactional data. In today’s digital world, online payments are everywhere, and fraud detection has become a critical cybersecurity and business problem. The goal of this project was to experiment with multiple models and identify the most reliable approach for detecting fraud with strong recall and balanced performance.
Instead of just aiming for high accuracy, I focused on something more meaningful: correctly identifying fraudulent cases without overwhelming businesses with false alarms. This mindset shift helped me approach the problem more realistically, like a real data science task.

### Project Goals

- Build a predictive ML pipeline to classify transactions as fraudulent or legitimate
- Compare multiple supervised learning models
- Address real-world challenges such as class imbalance and noisy data
- Perform feature engineering to capture behavioral fraud patterns
- Deploy the model inside an interactive web application for real-time predictions
- This project reflects a full end-to-end data science workflow, from data preprocessing to model deployment.

### Tech Stack

- Python (Pandas, NumPy, Scikit-learn, XGBoost)
- Data preprocessing & feature engineering
- Machine Learning model evaluation
- Flask (for deployment)
- Jupyter Notebook & VS Code

### Dataset

I used a real-world Fraudulent E-Commerce Transaction Dataset from Kaggle and later merged it with a credit card fraud dataset to improve fraud representation and reduce imbalance. This increased the fraud ratio from ~5% to around 6.5%, allowing models to better learn minority class patterns. This merging step became a key turning point in the project because it significantly improved recall and model stability.

### Models Implemented

I experimented with multiple supervised learning models to understand how each handles fraud detection:

- Logistic Regression (baseline interpretable model)
- Decision Tree (nonlinear interpretable model)
- Random Forest (ensemble learning approach)
- XGBoost (advanced gradient boosting model)

All models shared the same preprocessing pipeline using a ColumnTransformer with one-hot encoding for categorical features and numerical passthrough for consistency and fair comparison.

### Experimental Design

- 80/20 stratified train-test split
- Handling class imbalance with class_weight="balanced"
- Feature engineering including:
 - Log-transformed transaction amount
 - Weekend and day-of-week indicators
 - Customer age buckets
 - Account age flags
- Hyperparameter tuning for ensemble models

I evaluated performance mainly using Recall and ROC-AUC, since missing fraudulent transactions is more costly than incorrectly flagging legitimate ones.

### Key Results & Insights

- Logistic Regression achieved strong recall but produced more false positives
- Random Forest gave the most balanced performance
- XGBoost achieved the highest accuracy and stable ROC-AUC
- Dataset balancing significantly improved recall and F1-scores across models
Overall, Random Forest and XGBoost demonstrated the best trade-off between fraud detection power and false positive control.
This comparison helped me understand how model choice depends on business priorities (catching more fraud vs. minimizing unnecessary alerts).

### Web App Deployment

To make the project practical and interactive, I built a Flask-based web application that allows users to input transaction details and instantly receive a fraud prediction with probability score.

Users can enter:

- Transaction amount
- Customer age and account age
- Payment method
- Product category
- Device type and location

The model then predicts whether the transaction is likely fraudulent in real time, bridging the gap between theoretical modeling and real-world usability. I also experimented with an AI chatbot integration to explain predictions conversationally, which gave me valuable exposure to model interpretability and user-centered deployment. 

### What I Learned

This project was more than just training models. It taught me:
- How real-world datasets are messy and imbalanced
- Why accuracy alone is misleading in fraud detection
- The importance of feature engineering in capturing behavioral patterns
- How ensemble methods improve generalization
- The practical challenges of large-scale data processing and hyperparameter tuning

Most importantly, I learned to think like a data scientist: focusing on meaningful metrics, experimenting iteratively, and translating model outputs into actionable insights.
