# Diabetes Prediction using Machine Learning

## Project Overview

This project predicts whether a patient is likely to have **Type 2 Diabetes** using Machine Learning.
The model is trained on the **Pima Indians Diabetes Dataset** and deployed as a simple web application.

The project demonstrates:

* Data preprocessing
* Machine learning model training
* Model evaluation
* Web application deployment

## Dataset

The model is trained using the **Pima Indians Diabetes Dataset**.
It contains medical predictor variables and one target variable indicating whether a patient has diabetes.

Features used in the dataset:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI (Body Mass Index)
* Diabetes Pedigree Function
* Age

Target variable:

* Outcome (0 = No Diabetes, 1 = Diabetes)

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

## Machine Learning Model

A **Random Forest Classifier** is used to train the model.

Steps followed:

1. Load dataset
2. Split dataset into training and testing sets
3. Apply feature scaling using StandardScaler
4. Train Random Forest model
5. Evaluate model accuracy
6. Save the trained model and scaler

## Project Structure

diabetes_prediction/
│
├── diabetes.csv                # Dataset
├── train_model.py              # Script to train the model
├── predict.py                  # Script for prediction
├── app.py                      # Streamlit web application
├── diabetes_model.pkl          # Saved ML model
├── scaler.pkl                  # Saved scaler
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

## How to Run the Project

### 1 Install Dependencies

pip install -r requirements.txt

### 2 Train the Model

python train_model.py

### 3 Run the Streamlit App

python -m streamlit run app.py

### 4 Open the Web App

live link: https://diabetes-prediction-ml-cs6lmwzhgoyjspc2wygybm.streamlit.app/

## Example Prediction Input

Pregnancies: 6
Glucose: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Diabetes Pedigree Function: 0.627
Age: 50
