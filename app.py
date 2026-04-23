import streamlit as st
import numpy as np
import pickle

# Load model + scaler safely
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="Liver Disease Predictor")

st.title("🩺 Liver Disease Prediction App")
st.markdown("Enter patient medical details below:")

# ---------------- INPUTS ---------------- #

age = st.number_input("Age", min_value=1, max_value=120, value=30)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0

tb = st.number_input("Total Bilirubin (TB)", min_value=0.0, value=1.0)
db = st.number_input("Direct Bilirubin (DB)", min_value=0.0, value=0.3)
alkphos = st.number_input("Alkaline Phosphotase", min_value=0.0, value=200.0)
sgpt = st.number_input("SGPT", min_value=0.0, value=30.0)
sgot = st.number_input("SGOT", min_value=0.0, value=30.0)
tp = st.number_input("Total Proteins (TP)", min_value=0.0, value=6.5)
alb = st.number_input("Albumin (ALB)", min_value=0.0, value=3.5)
ag_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, value=1.0)

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    try:
        # IMPORTANT: order must match training data EXACTLY
        input_data = np.array([[ 
            age,
            gender_val,
            tb,
            db,
            alkphos,
            sgpt,
            sgot,
            tp,
            alb,
            ag_ratio
        ]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # ---------------- OUTPUT ---------------- #

        st.subheader("Result")

        if prediction == 1:
            st.error(f"⚠️ High chance of Liver Disease")
        else:
            st.success(f"✅ Low chance of Liver Disease")

        st.write(f"**Prediction Confidence:** {probability:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")