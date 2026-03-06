import streamlit as st
import joblib
import numpy as np
import os

st.title("Eksu Student Course Risk Predictor")

# Load model safely
MODEL_PATH = "eksu_risk_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please upload eksu_risk_model.joblib to the repository.")
else:
    model = joblib.load(MODEL_PATH)

    study = st.slider("Study hours per week",0,40,10)
    
    sleep = st.slider("Avg Sleep hours",4,10,6)
    
    phone = st.slider("Phone usage hours",0,12,5)
    
    attendance = st.slider("Attendance level (1=poor, 5=excellent)",1,5,3)
    
    difficulty = st.slider("Course difficulty (1=easy, 5=hard)",1,5,3)
    
    stress = st.slider("Financial stress (1=none, 5=extreme)",1,5,3)

    gpa = st.input("Last semester GPA")

    st.body("If you are a 100l students kindly input 3.00 as your last semester gpa as to not skew your prediction")
    
    if st.button("Analyze Risk"):

        X = np.array([[study, sleep, phone, attendance, difficulty, stress]])

        prob = model.predict_proba(X)[0][1]

        st.subheader("Failure Probability")

        proba = (prob * 100)
        st.write(proba)

        if prob < 0.3:
            st.success("Low Risk")
        elif prob < 0.6:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

st.subtitle("Dear Students Your inputed data is used to improve the model anonymously, this if for educational purposes only and should not be heavily relied upon")
