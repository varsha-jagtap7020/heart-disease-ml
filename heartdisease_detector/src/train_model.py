
"""
Machine Learning-Based Heart Disease Prediction

Author: Varsha Rajkumar Jagtap

Description:
This script implements machine learning models to predict the risk of
heart disease using clinical patient data obtained from the
UCI Machine Learning Repository.

Models Implemented:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Evaluation Metric:
- Accuracy Score

Dataset:
Heart Disease Dataset (UCI Repository)

This implementation was used as part of an independent
undergraduate research project.
"""
numeric_features = ['age','trestbps','chol','thalch','oldpeak','ca']

categorical_features = ['sex','cp','fbs','restecg','exang','slope','thal']
 
 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



import joblib



import pandas as pd

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "heart_disease_uci.csv")

df = pd.read_csv(data_path)


# 1. Check missing values
print(df.isnull().sum())

# 2. Check target column
print(df['num'].value_counts())

# 3. Basic info
df.info()


# Create binary target
df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

# Check new target distribution
print(df['target'].value_counts())

df = df.drop(columns=['id', 'dataset', 'num'])
print(df.columns)

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill numeric missing values with median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical missing values with mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Check again
print(df.isnull().sum())

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])


# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Convert categorical columns to numeric using one-hot encoding

# Check shape of final feature matrix
print(X.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1: Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)

lr_pred = lr_pipeline.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", lr_acc)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

# ---------------- RANDOM FOREST PIPELINE ----------------

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
rf_pipeline.fit(X_train, y_train)

# Save model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "results", "heart_model.pkl")
joblib.dump(rf_pipeline, model_path)
print("Pipeline model saved successfully!")

# Predict
rf_pred = rf_pipeline.predict(X_test)

# Accuracy
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_acc)

# --------- Plot Accuracy ---------

import matplotlib.pyplot as plt

plt.figure()
plt.bar(["Random Forest"], [rf_acc])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy")
plt.savefig(os.path.join(current_dir, "..", "results", "accuracy_plot.png"))
plt.close()
