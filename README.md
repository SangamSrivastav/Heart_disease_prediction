# Heart Disease Prediction

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg) ![Language](https://img.shields.io/github/languages/top/sangamsrivastav/Heart_Disease_Prediction) ![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-92.86%25-brightgreen)

This repository contains a machine learning project that predicts the presence of heart disease in individuals based on various medical parameters. The project explores several classification algorithms, performs hyperparameter tuning, and evaluates model performance to identify the best-performing model for accurate heart disease prediction.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Results and Conclusions](#results-and-conclusions)
- [Usage](#usage)
- [Contact](#contact)

---

## Project Overview

The goal of this project is to build a robust machine learning model that can assist in the early detection of heart disease. Using a dataset of patient medical records, we train and evaluate different classification models to find the one that provides the highest accuracy. The final model, a fine-tuned Random Forest Classifier, can be used to make predictions on new, unseen data.

*Key Features:*
- *Machine Learning-Based Diagnosis:* Utilizes several popular classification algorithms.
- *Optimized with GridSearchCV:* Employs hyperparameter tuning to maximize model performance.
- *Real-Time Prediction:* Includes a script to make predictions on user-provided input data.
- *Performance Metrics Evaluation:* Analyzes model accuracy, confusion matrices, and ROC-AUC scores.

---

## Dataset

The dataset used in this project is named heart_dataset.csv. It contains 1190 rows and 12 columns, with no missing values. The dataset is a clean, well-structured resource for a classification task.

### Verifying the Dataset
- *Shape:* (1190, 12)
- *Data Types:* All features are numerical, with a mix of int64 and float64.
- *Missing Values:* No null values were found, ensuring data quality for model training.

---

## Features

The dataset includes the following medical parameters:

- age: Age of the patient.
- sex: Gender of the patient (1 = male, 0 = female).
- cp: Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic).
- tresbps: Resting blood pressure.
- chol: Serum cholesterol in mg/dl.
- fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
- restecg: Resting electrocardiographic results (values 0, 1, 2).
- thalach: Maximum heart rate achieved.
- exang: Exercise-induced angina (1 = yes, 0 = no).
- oldpeak: ST depression induced by exercise relative to rest.
- slope: The slope of the peak exercise ST segment.
- target: The target variable indicating the presence of heart disease (1 = positive, 0 = negative).

### Exploratory Data Analysis (EDA) Insights:
- *Sex vs. Target:* Females appear to be more likely to have heart problems than males in this dataset.
- *Chest Pain Type vs. Target:* Individuals with typical angina (cp = 1) are significantly less likely to have heart problems compared to those with other chest pain types.
- *Resting ECG vs. Target:* Patients with resting ECG results of '1' or '2' are more prone to heart disease than those with a result of '0'.
- *Slope vs. Target:* A slope of '2' in the peak exercise ST segment is strongly associated with a higher likelihood of heart disease.

---

## Methodology

The project follows a standard machine learning workflow:
1. *Data Preprocessing:* The dataset was loaded and checked for missing values and duplicates.
2. *Train-Test Split:* The data was split into training (80%) and testing (20%) sets to ensure the model's performance could be evaluated on unseen data.
3. *Model Training:* The following algorithms were trained and their hyperparameters were tuned using GridSearchCV or RandomizedSearchCV for optimal performance.
    - Logistic Regression
    - Naive Bayes
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Decision Tree
    - Random Forest
4. *Evaluation:* Each model was evaluated based on its accuracy score and confusion matrix on the test set.

---

## Results and Conclusions

The accuracy scores achieved by the different models are as follows:

| Algorithm               | Accuracy Score (%) |
| ----------------------- | ------------------ |
| Logistic Regression     | 80.67              |
| Naive Bayes             | 85.29              |
| Support Vector Machine  | 82.77              |
| K-Nearest Neighbors     | 85.29              |
| Decision Tree           | 89.50              |
| Random Forest           | *92.86* |

Based on these results, the *Random Forest Classifier* demonstrated the highest accuracy, making it the most suitable model for this heart disease prediction task. Its ability to handle complex, non-linear relationships in the data, combined with effective hyperparameter tuning, resulted in a highly reliable predictive model.

---

## Usage

You can use the Jupyter Notebook Heart_disease_prediction_(1)(1).ipynb to replicate the results. The notebook also includes a section where you can input new data to get a real-time heart disease prediction.

Example input for prediction:
python
# input_data = (age, sex, cp, tresbps, chol, fbs, restecg, thalach, exang, oldpeak, slope)
input_data = (37, 1, 2, 130, 283, 0, 1, 98, 0, 0, 1)

# The Person does not have a Heart Disease


---

## Contact
- *Name:* Sangam Srivastav
- *Email:* sangamsrivastav2002@gmail.com
