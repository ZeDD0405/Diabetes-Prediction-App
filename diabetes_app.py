import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")

st.markdown("Enter the details below to check if the person is diabetic:")

# User input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
insulin = st.number_input("Insulin Level", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=1, step=1)

if st.button("Predict"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                 insulin, bmi, dpf, age]],
                               columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Standardize
    std_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(std_input)

    if prediction[0] == 1:
        st.error("⚠️ The person is **diabetic**.")
    else:
        st.success("✅ The person is **not diabetic**.")
