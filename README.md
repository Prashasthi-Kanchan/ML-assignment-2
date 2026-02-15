a. **Problem Statement**
The objective of this project is to build and evaluate multiple Machine Learning models to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.
This is a binary classification problem where:
•	Target Variable: Income (<=50K or >50K)
•	Goal: Compare different ML models and identify the best-performing model using multiple evaluation metrics.

b. **Dataset Description** [1 Mark]
The dataset used in this project was obtained from
Kaggle
Dataset Name: Adult Census Income Dataset
Dataset Overview:
  Total instances: ~48,842 records
  Number of features: 14 input features + 1 target variable
  Target variable: income
  Type: Structured tabular dataset
  Problem Type: Binary Classification
Key Features:
•	Age
•	Workclass
•	Education
•	Marital Status
•	Occupation
•	Race
•	Sex
•	Capital Gain
•	Capital Loss
•	Hours per Week
•	Native Country
Preprocessing Performed:
•	Handling missing values
•	Label encoding / one-hot encoding
•	Feature scaling (for Logistic Regression and KNN)
•	Train-Test Split
•	Model training and evaluation

c. **Models Used and Evaluation Metrics** [6 Marks]
The following models were trained and evaluated:
•	Logistic Regression
•	Decision Tree
•	K-Nearest Neighbors (KNN)
•	Naive Bayes
•	Random Forest (Ensemble)
•	XGBoost (Ensemble)
Evaluation metrics used:
•	Accuracy
•	Precision
•	Recall
•	F1 Score
•	AUC (Area Under ROC Curve)
•	MCC (Matthews Correlation Coefficient)
________________________________________
Model Comparison Table
**| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |**
|---------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.851078 | 0.903775 | 0.739374 | 0.600091 | 0.662491 | 0.573368 |
| Decision Tree | 0.813599 | 0.746341 | 0.617983 | 0.614616 | 0.616295 | 0.493198 |
| KNN | 0.822001 | 0.844168 | 0.654830 | 0.569224 | 0.609034 | 0.496566 |
| Naive Bayes | 0.812161 | 0.884299 | 0.592784 | 0.730822 | 0.654605 | 0.532980 |
| Random Forest (Ensemble) | 0.851299 | 0.897473 | 0.734170 | 0.610531 | 0.666667 | 0.576038 |
| XGBoost (Ensemble) | 0.875954 | 0.929606 | 0.797468 | 0.657739 | 0.720896 |
________________________________________

d. **Observations on Model Performance** [3 Marks]
**| ML Model Name | Observation about Model Performance |**
|---------------|--------------------------------------|
| Logistic Regression | Performed well with high AUC (0.90). Balanced and stable model with good generalization. Lower recall indicates difficulty capturing all high-income cases. |
| Decision Tree | Moderate performance. Prone to overfitting and shows lower AUC compared to other models. |
| KNN | Performed better than Decision Tree but slightly lower recall. Sensitive to scaling and feature distribution. |
| Naive Bayes | High recall (0.73), meaning it captures more high-income individuals, but lower precision leads to more false positives. |
| Random Forest (Ensemble) | Improved performance over single Decision Tree. Better stability and balanced metrics. Good generalization capability. |
| XGBoost (Ensemble) | Best performing model overall. Highest Accuracy, AUC, F1, and MCC. Demonstrates strong predictive power and superior handling of feature interactions. |
