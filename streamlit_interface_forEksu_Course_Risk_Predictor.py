
import streamlit as st
import joblib
import numpy as np

model = joblib.load("eksu_risk_model.joblib")

st.title("EKSU Course Risk Predictor")

study = st.slider("Study hours per week",0,40,10)
sleep = st.slider("Sleep hours",3,10,6)
phone = st.slider("Phone usage hours",0,12,5)

attendance = st.selectbox("Attendance level",[0,1,2])
difficulty = st.selectbox("Course difficulty",[0,1,2])
stress = st.selectbox("Financial stress",[0,1,2])

if st.button("Analyze Risk"):

    X = np.array([[study,sleep,phone,attendance,difficulty,stress]])

    prob = model.predict_proba(X)[0][1]

    st.write("Failure Probability:",prob)