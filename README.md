# ❤️ Machine Learning-Based Heart Disease Prediction

## 📌 Project Overview
This project predicts the risk of heart disease using machine learning algorithms based on clinical patient data from the UCI Machine Learning Repository.

The objective of this study is to assist in early risk assessment by developing a predictive system using commonly available medical attributes.

---

## 📂 Dataset
Heart Disease Dataset – UCI Machine Learning Repository

Clinical parameters used:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Rest ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Slope
- CA
- Thal

---

## ⚙️ Models Implemented
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 79.8% |
| Random Forest | 86.9% |
| SVM | 71.1% |

---

## 🛠 Technologies Used
- Python
- Pandas
- Scikit-Learn
- Matplotlib
- NumPy

---

## 📈 Results
The Random Forest classifier achieved the highest accuracy of **86.9%** in predicting heart disease risk.

---

## 📎 Project Structure

```
heart-disease-ml/
│
├── README.md
├── requirements.txt
│
└── heartdisease_detector/
    ├── data/
    │   └── heart_disease_uci.csv
    │
    ├── src/
    │   ├── train_model.py
    │   └── predict.py
    │
    └── results/
        ├── heart_model.pkl
        └── accuracy_plot.png
```


