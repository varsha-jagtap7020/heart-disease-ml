import joblib
import pandas as pd
import os

# Load saved model pipeline
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "results", "heart_model.pkl")

model = joblib.load(model_path)

print("Model loaded successfully!")

# Sample patient data (RAW FORMAT)
sample_data = pd.DataFrame([{
    'age': 55,
    'sex': 'Male',
    'cp': 'asymptomatic',
    'trestbps': 140,
    'chol': 240,
    'fbs': 'False',
    'restecg': 'normal',
    'thalch': 150,
    'exang': 'True',
    'oldpeak': 2.3,
    'slope': 'flat',
    'ca': 0,
    'thal': 'normal'
}])

# Predict
prediction = model.predict(sample_data)

if prediction[0] == 1:
    print("High Risk of Heart Disease")
else:
    print("Low Risk of Heart Disease")
