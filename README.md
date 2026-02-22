#  Diabetes Predictive Analytics Project

##  Project Overview
This project performs Exploratory Data Analysis (EDA) and predictive analytics on a healthcare dataset to identify patterns and predict diabetes outcomes.

The objective is to analyze patient health indicators and build models to classify whether a patient is likely to have diabetes.

---

## Dataset Information
- Dataset: Pima Indians Diabetes Dataset
- Total Records: 768 patients
- Features: 8 medical attributes
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

##  Exploratory Data Analysis (EDA)
- Handled missing and invalid values
- Replaced medically impossible zero values
- Performed correlation analysis
- Generated heatmap to identify important predictors

Key Insight:
Glucose level showed the highest correlation with diabetes outcome.

---

##  Models Used

### 1️ Logistic Regression
- Accuracy: ~77%
- ROC-AUC Score: 0.8229

### 2️ Random Forest Classifier
- Accuracy: ~74%
- ROC-AUC Score: 0.8334

Random Forest performed slightly better in terms of AUC score.

---

##  Model Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC Curve
- AUC Score

---

##  Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

##  Project Workflow
1. Data Cleaning
2. Feature Scaling
3. Exploratory Data Analysis
4. Model Training
5. Model Comparison
6. Performance Evaluation

---

##  Conclusion
The project demonstrates how healthcare data can be analyzed to extract meaningful insights and build predictive models.

Random Forest achieved the highest performance with an AUC score of 0.83, indicating good classification capability.

---

## 
Sameer Birla  
B.Tech (AI & ML)  
Aspiring Data Analyst
