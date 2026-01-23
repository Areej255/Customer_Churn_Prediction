# Customer_Churn_Prediction
End-to-end customer churn prediction using machine learning with a Streamlit dashboard, focusing on high-recall modeling and business-driven decision metrics.


Customer Churn Prediction & Retention Analysis

📌 Project Overview

Customer churn directly impacts business revenue.
This project focuses on predicting customers who are likely to churn using machine learning, with a strong emphasis on high recall to minimize missed churners.
A Streamlit dashboard is built to make predictions interpretable and actionable from a business perspective.
This is an end-to-end project covering data preprocessing, model training, evaluation, and deployment.

🎯 Business Objective

Identify customers at high risk of churn

Prioritize recall over accuracy to avoid missing churners

Support retention campaigns while considering campaign and retention costs

Enable non-technical users to interact with model outputs via a dashboard

🧠 Key Concepts Applied

Churn prediction as a cost-sensitive classification problem
Precision vs Recall tradeoff
Retention cost and campaign cost awareness
Threshold-based decision making

📊 Dataset

Structured customer-level dataset containing behavioral, demographic, and service usage features
Target variable: Churn (Yes / No)

Data preprocessing included:
Handling missing values
Encoding categorical variables
Feature scaling

⚙️ Machine Learning Workflow

Exploratory Data Analysis (EDA)
Data preprocessing and feature scaling
Model training using supervised learning algorithms

Model evaluation using:
Recall
Precision
Confusion Matrix
ROC-AUC
Threshold tuning to improve churn detection
Feature importance analysis
Models Used
Logistic Regression
Random Forest
The final model was selected based on business-relevant metrics, not just accuracy.

📈 Feature Importance

Feature importance was analyzed to:
Understand key drivers of churn
Provide interpretability for business stakeholders
Support retention strategy design

🖥️ Streamlit Dashboard

The Streamlit application allows users to:
Upload customer data
View churn predictions
Analyze feature importance
Explore confusion matrix and recall-focused metrics
Understand model behavior through visualizations

💰 Business Insight

Missing a churner (False Negative) is more costly than targeting a non-churner (False Positive)
Model decisions are aligned with campaign cost vs customer value
High recall ensures better retention coverage, even at the cost of lower precision

🛠️ Tech Stack

Python
Pandas, NumPy
Scikit-learn
Matplotlib / Seaborn
Streamlit

📌 Results

Achieved high recall for churn detection
Improved business interpretability through dashboard-based insights
Demonstrated end-to-end ML deployment skills

🔮 Future Improvements

Cost-sensitive learning using custom loss functions
Integration with real-time customer data
A/B testing for retention strategies
Model monitoring and drift detection

📬 Contact

Areeg Naeem
Bachelor’s in Software Engineering

Aspiring Data Scientist / Data Analyst

