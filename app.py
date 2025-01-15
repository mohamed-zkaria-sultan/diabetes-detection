import streamlit as st
import numpy as np
import pickle  # or joblib
import pandas as pd

# Load the pre-trained model
with open('lg.pkl', 'rb') as file:
    lg = pickle.load(file)

# User Inputs
st.title("Diabetes Prediction App")

Pregnancies = st.number_input("Enter Pregnancies", min_value=0, max_value=17, value=0)
Glucose = st.number_input("Enter Glucose", min_value=0, max_value=199, value=100)
BloodPressure = st.number_input("Enter BloodPressure", min_value=0, max_value=122, value=80)
SkinThickness = st.number_input("Enter SkinThickness", min_value=0, max_value=99, value=20)
Insulin = st.number_input("Enter Insulin", min_value=0, max_value=846, value=80)
BMI = st.number_input("Enter BMI", min_value=0.0, max_value=67.0, value=25.0, step=0.1)
DiabetesPedigreeFunction = st.number_input("Enter Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.5, step=0.01)
Age = st.number_input("Enter Age", min_value=21, max_value=81, value=30)

# Prepare input data
input_model = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

# Predict button
state = st.button("Predict")
if state:
    pred = lg.predict([input_model])  # Predicting
    result = "Diabetic" if pred[0] == 1 else "Non-Diabetic"
    st.text(f"The prediction is: {result}")
