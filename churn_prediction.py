# importing libs
import pandas as pd
import numpy as np
# visualization libs
from pyparsing import col
import seaborn as sns
import matplotlib.pyplot as plt
# ml tools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# loading dataset
df=pd.read_csv('C:\\Users\\PMLS\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
# print(df.head())
# print(df.info())

# basic data cleaning
# removing the duplicate rows
df.drop_duplicates(inplace=True)

# now checking for missing values
# print(df.isnull().sum())

# handling missing values
# numerical columns -> fill with median(robust against outliers)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# categorical columns -> fill with mode(most frequent value)
cat_cols = df.select_dtypes(include=["object"]).columns
cat_cols = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
# print(cat_cols)

# encode target variable
# convert "churn" column to binary values: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes":1, "No" :0})
# print(df["Churn"])


# print(x_train.dtypes)
# dropping non informative customer id column
# x_train.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
cat_cols = df.select_dtypes(include=["object"]).columns
# print(cat_cols)

# converting all categorical features to numerical using label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
# print(df.select_dtypes(include=["object"]).columns)

# feature and target separation
x = df.drop("Churn", axis=1)
y = df["Churn"]
# splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)

# verifying all features are numeric
# print(x_train.select_dtypes(include=["object"]).columns)

# feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# model training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)

# making predictions
y_pred = model.predict(x_test_scaled)

# evaluating the model
# accuracy
logis_reg_score = accuracy_score(y_test,y_pred)
print(f"Accuracy: {logis_reg_score:.4f}")
# confusion matrix
c_m = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n", c_m)
# classification report
class_report = classification_report(y_test,y_pred)
print("\nClassification Report:\n", class_report)

# visualizing confusion matrix
sns.heatmap(c_m, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Visualization")
plt.show()

# adding business intelligence
# df["Predicted_Churn"] = model.predict(scaler.transform(x))
df["Churn_Probability"] = model.predict_proba(scaler.transform(x))[:, 1]
# high risk customers checking churn probability > 0.7
high_risk_customers = df[df["Churn_Probability"] > 0.7]
# print(high_risk_customers[["customerID", "Churn_Probability"]])

# # feature importance using coefficients from logistic regression
feature_importance = pd.DataFrame({
    "feature": x.columns,
    "importance": model.coef_[0]
}).sort_values(by="importance", ascending=False)
print("\nFeature Importance:\n", feature_importance)

# # visualizing feature importance
plt.figure(figsize=(10,6))
sns.barplot(x="importance", y="feature", data=feature_importance)
plt.title("Feature Importance Visualization")
plt.show()


# comparing multiple models
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train_scaled, y_train)
rf_y_pred = rf_model.predict(x_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # comparing accuracies
print(f"Logistic Regression Accuracy: {logis_reg_score:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # visualizing model comparison
model_comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [logis_reg_score, rf_accuracy]
})
plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="Accuracy", data=model_comparison)
plt.title("Model Accuracy Comparison")
plt.ylim(0,1)
plt.show()

# threshold optimization
# setting threshold to 0.8

from sklearn.metrics import precision_recall_curve
# getting churn probabilities
y_scores = model.predict_proba(x_test_scaled)[:,1]
# optimizing threshold for better precision-recall trade-off
precion, recall, thresholds = precision_recall_curve(y_test, y_scores)
# converting to dataframe for easy ananlysis
pr_df = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precion[:-1],
    "Recall": recall[:-1]
})
# now choosing threshold where recall > 80% and precision is maximized
optimal_threshold = pr_df[(pr_df["Recall"]>=0.8) & (pr_df["Precision"]>=0.40)].iloc[0] 
# print("optimal_threshold:", round(optimal_threshold,4))
# by doing this “By lowering the decision threshold to 9.4%, the model catches almost all churners (95%) but only 40% of flagged customers actually churn.”

# business impact simulation
# assuming cost of retention offer is $100 per customer
retention_cost_per_customer = 100
# assuming average revenue per customer is $500
# revenue saved per retained customer
revenue_per_customer = 500
# true churners caught by the model(true positives)
true_positives = ((y_test == 1)&(y_scores >= optimal_threshold["Threshold"])).sum()
# financial calculations
revenue_saved = true_positives * revenue_per_customer
campaign_cost = ((y_scores >= optimal_threshold["Threshold"]).sum()) * retention_cost_per_customer
net_profit = revenue_saved - campaign_cost
print(f"Revenue Saved: ${revenue_saved}")
print(f"Campaign Cost: ${campaign_cost}")
print(f"Net Profit gain from Retention Campaign: ${net_profit}")



# now saving the trained model and scaler for future use
import joblib
joblib.dump(model, "churn_prediction_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(x.columns.tolist(), "feature_columns.pkl")

# import joblib

# Load the model
# model = joblib.load("churn_prediction_model.pkl")

# Test it (for example, with sample data)
# sample_data = [[1, 0, 3, 5,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19]] # example input, match your feature structure
# prediction = model.predict(sample_data)
# print(prediction)

# # Extract feature names (all columns except target)
# feature_names = list(data.drop("Churn", axis=1).columns)  # replace 'Churn' with your target column

# # Save them
# joblib.dump(feature_names, "feature_names.pkl")






