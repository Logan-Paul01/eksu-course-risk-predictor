import streamlit as st
import joblib
import numpy as np
import os

st.title("EKSU Course Risk Predictor")

# Load model safely
MODEL_PATH = "eksu_risk_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please upload eksu_risk_model.joblib to the repository.")
else:
    model = joblib.load(MODEL_PATH)

    study = st.slider("Study hours per week", 0, 40, 10)
    sleep = st.slider("Sleep hours per day", 3, 10, 6)
    phone = st.slider("Phone usage hours per day", 0, 12, 5)

    attendance = st.selectbox("Attendance level", [0,1,2])
    difficulty = st.selectbox("Course difficulty", [0,1,2])
    stress = st.selectbox("Financial stress", [0,1,2])

    if st.button("Analyze Risk"):

        X = np.array([[study, sleep, phone, attendance, difficulty, stress]])

        prob = model.predict_proba(X)[0][1]

        st.subheader("Failure Probability")

        st.write(prob)

        if prob < 0.3:
            st.success("Low Risk")
        elif prob < 0.6:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")
