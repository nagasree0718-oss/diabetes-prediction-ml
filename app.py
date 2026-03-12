import streamlit as st
import numpy as np
import joblib
# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("Diabetes Prediction System")
st.write("Enter patient medical details")
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)
if st.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    data = scaler.transform(data)
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("Patient likely has Diabetes")
    else:
        st.success("Patient likely does NOT have Diabetes")
