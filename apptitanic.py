import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from streamlit_lottie import st_lottie

# --- Streamlit Page Config ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered", page_icon="🚢")

# --- Clean Styling Without GIF Background ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }

        .block-container {
            background-color: rgba(255, 255, 255, 0.88);
            padding: 2rem;
            border-radius: 10px;
        }

        .stButton>button {
            color: white;
            background-color: #007bff;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load the trained model ---
model_path = "random_forest2.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Load Lottie Animations Safely ---
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"Lottie animation failed to load: HTTP {r.status_code}")
    except Exception as e:
        st.warning(f"Error loading Lottie animation: {e}")
    return None

welcome_anim = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_tfb3estd.json")
survive_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_ydo1amjm.json")
not_survive_anim = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json")

# --- UI Title and Animation ---
st.title("🚢 Titanic Survival Predictor")
if welcome_anim:
    st_lottie(welcome_anim, height=200, key="welcome")

st.markdown("Enter passenger details to predict survival on the Titanic.")

# --- Sidebar Inputs ---
st.sidebar.header("🧾 Passenger Details")

pclass = st.sidebar.selectbox("🎫 Ticket Class", [1, 2, 3])
sex = st.sidebar.selectbox("🧑 Sex", ["male", "female"])
age = st.sidebar.slider("🎂 Age", 1, 80, 25)
sibsp = st.sidebar.slider("👨‍👩‍👧‍👦 Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("👶 Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("💰 Fare Paid", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("🛳️ Port of Embarkation", ["S", "C", "Q"])

# --- Show Avatar (Gender-specific logic) ---
if st.checkbox("🎭 Show Avatar (For Fun)"):
    avatar_style = "micah" if sex == "male" else "notionists"
    seed = f"{sex}-{age}"
    avatar_url = f"https://api.dicebear.com/7.x/{avatar_style}/png?seed={seed}"
    st.image(avatar_url, width=150, caption=f"{sex.capitalize()} Avatar")

# --- Feature Encoding ---
sex_encoded = 1 if sex == "male" else 0
embarked_dict = {"S": 0, "C": 1, "Q": 2}
embarked_encoded = embarked_dict[embarked]

# --- Prepare input for prediction ---
input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

# --- Predict Button ---
if st.button("Predict Survival"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Summary
        st.markdown("### 👤 Passenger Summary")
        st.info(f"""
        - *Class:* {pclass}  
        - *Sex:* {sex.capitalize()}  
        - *Age:* {age}  
        - *Siblings/Spouses:* {sibsp}  
        - *Parents/Children:* {parch}  
        - *Fare:* ${fare:.2f}  
        - *Embarked from:* {embarked}
        """)

        # Prediction
        st.markdown(f"<h4 style='color: {'green' if prediction==1 else 'red'};'>Survival Probability: {probability:.2%}</h4>", unsafe_allow_html=True)
        st.progress(int(probability * 100))

        if prediction == 1:
            st.success(f"🎉 The passenger is likely to Survive! (Probability: {probability:.2f})")
            if survive_anim:
                st_lottie(survive_anim, height=300, key="survive")
        else:
            st.error(f"💀 The passenger is likely Not to Survive. (Probability: {probability:.2f})")
            if not_survive_anim:
                st_lottie(not_survive_anim, height=300, key="not_survive")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# --- Model Explanation ---
with st.expander("ℹ️ How the prediction works"):
    st.write("""
    This app uses a *Random Forest classifier* trained on Titanic passenger data.
    It considers factors like *age, sex, travel class, fare, and embarkation port*
    to estimate survival probability.
    """)

# --- Dataset Insights (optional) ---
if st.checkbox("📊 Show Dataset Insights"):
    try:
        df = pd.read_csv("train.csv")
        st.write(df.describe())
        st.bar_chart(df["Age"])
    except Exception as e:
        st.warning("⚠️ Could not load dataset insights. Ensure 'train.csv' is available.")

# --- Footer ---
st.markdown("---")
st.caption("Built by Krutika")
