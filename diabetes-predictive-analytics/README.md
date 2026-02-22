#  Diabetes Predictive Analytics Project

## About This Project

This project analyzes a healthcare dataset to understand the factors that influence diabetes and builds predictive models to classify whether a patient has diabetes or not.

The goal is to apply data analysis techniques and machine learning to extract insights and evaluate prediction performance.

---

## What This Project Does

1. Loads and cleans the dataset (handles invalid zero values)
2. Performs Exploratory Data Analysis (EDA)
3. Generates a correlation heatmap
4. Splits data into training and testing sets
5. Applies feature scaling
6. Trains two models:
   - Logistic Regression
   - Random Forest
7. Evaluates models using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
   - ROC Curve
   - AUC Score

---

## Dataset Information

- Dataset: Pima Indians Diabetes Dataset
- Total Records: 768 patients
- Target Variable: Outcome (0 = No Diabetes, 1 = Diabetes)

Key Features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

---

## Model Results

Logistic Regression:
- Accuracy: ~77%
- ROC-AUC Score: 0.8229

Random Forest:
- Accuracy: ~74%
- ROC-AUC Score: 0.8334

Random Forest achieved the highest AUC score, indicating strong predictive capability.

---

##  Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## ▶How to Run

1. Install required libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn

2. Run the script:
   python diabetes_analysis.py

---

##  Conclusion

This project demonstrates how healthcare data can be analyzed to identify important health indicators and build predictive models. The results show that glucose level is one of the strongest predictors of diabetes.

The project highlights data cleaning, visualization, model building, and performance evaluation skills.