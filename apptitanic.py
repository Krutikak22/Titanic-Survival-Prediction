import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered", page_icon="🚢")

# --- Load the trained model from .pkl file ---
model_path = "random_forest_model.pkl"

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Styling ---
st.markdown("""
    <style>
        .stButton>button {
            color: white;
            background-color: #007bff;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# --- UI Title ---
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival on the Titanic.")

# --- Sidebar Inputs ---
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 25)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# --- Feature Encoding ---
sex_encoded = 1 if sex == "male" else 0
embarked_dict = {"S": 0, "C": 1, "Q": 2}
embarked_encoded = embarked_dict[embarked]

# --- Prepare input for prediction ---
input_data = pd.DataFrame([[
    pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded
]], columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# --- Predict Button ---
if st.button("Predict Survival"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"🎉 The passenger is likely to **Survive**! (Probability: {probability:.2f})")
        else:
            st.error(f"💀 The passenger is likely **Not to Survive**. (Probability: {probability:.2f})")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ Kalvankar Krutika")
