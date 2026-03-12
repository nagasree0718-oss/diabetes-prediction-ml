import numpy as np
import joblib

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# example patient data
data = np.array([[2,120,70,20,80,30.5,0.5,35]])

data = scaler.transform(data)

prediction = model.predict(data)

if prediction[0] == 1:
    print("Patient likely has Diabetes")
else:
    print("Patient likely does NOT have Diabetes")