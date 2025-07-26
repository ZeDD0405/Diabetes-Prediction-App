import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# ✅ Custom CSS
st.markdown("""
    <style>
    .main {
        animation: fadeIn ease 1s;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .title-box {
        background-color: #111827;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 0 15px rgba(0,0,0,0.3);
    }

    .title-box h1 {
        color: #f1f1f1;
        text-align: center;
        margin-bottom: 5px;
    }

    .title-box p {
        color: #cccccc;
        text-align: center;
        margin-top: 0;
        font-size: 16px;
    }

    /* Input fields and hover */
    .stNumberInput > div {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-radius: 8px;
        padding: 4px;
        background-color: #1e1e1e;
        transition: box-shadow 0.3s ease-in-out;
    }

    .stNumberInput > div:hover {
        box-shadow: 0 0 10px #3b82f6;
    }

    input[type="number"] {
        background-color: #2a2a2a;
        color: white;
        border: 1px solid #444;
        padding: 6px 10px;
        border-radius: 5px;
    }

    /* Button + and - fix */
    button[title="decrement"], button[title="increment"] {
        background-color: #2d2d2d;
        border: 1px solid #444;
        color: white;
        border-radius: 6px !important;
        width: 32px;
        height: 32px;
        font-size: 20px;
        font-weight: bold;
        transition: background-color 0.2s ease;
    }

    button[title="decrement"]:hover, button[title="increment"]:hover {
        background-color: #3b82f6;
        color: white;
        border-color: #2563eb;
    }

    /* Predict button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 6px;
        border: none;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #2563eb;
        transform: scale(1.03);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }

    </style>
""", unsafe_allow_html=True)

# 🩺 Title & Author
st.markdown(
    """
    <div class="title-box">
        <h1>🩺 Diabetes Prediction App</h1>
        <p>Created and deployed by <b>Sagar Kallimani</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# 📋 Inputs
st.markdown("📋 **Enter the details below:**")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
insulin = st.number_input("Insulin Level", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=1, step=1)

# 🔍 Predict Button
if st.button("Predict"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                 insulin, bmi, dpf, age]],
                               columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    std_input = scaler.transform(input_data)
    prediction = model.predict(std_input)

    if prediction[0] == 1:
        st.error("⚠️ The person is **diabetic**.")
    else:
        st.success("✅ The person is **not diabetic**.")
