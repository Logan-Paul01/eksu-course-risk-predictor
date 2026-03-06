
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import uuid  # for unique ID generation

st.set_page_config(
    page_title="EKSU Course Risk Predictor",
    page_icon="📊"
)

st.title("📊 EKSU Course Risk Predictor")

MODEL_PATH = "model/eksu_risk_model.joblib"
LOG_FILE = "data/prediction_log.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load the model
if not os.path.exists(MODEL_PATH):
    st.error("Model file missing. Upload eksu_risk_model.joblib to the correct path.")
else:
    model = load_model()

    st.header("📘 Course Info")
    course = st.text_input("Which course are you predicting? (e.g., MTH103)")
    gpa = st.number_input("Current GPA (enter 3.0 if no GPA yet)", 0.0, 5.0, 3.0, step=0.01)

    st.header("📚 Study Behaviour")
    study = st.slider("Study hours per week for this course", 0, 40, 10)
    sleep = st.slider("Average sleep hours per night", 4, 10, 6)
    phone = st.slider("Daily phone usage hours", 0, 12, 5)

    st.header("🎓 Academic Context")
    attendance = st.slider("Lecture attendance (1=poor, 5=excellent)", 1, 5, 3)
    difficulty = st.slider("Course difficulty perception (1=easy, 5=very difficult)", 1, 5, 3)
    stress = st.slider("Financial stress level (1=none, 5=extreme)", 1, 5, 3)

    if st.button("Predict Risk"):

        # Prepare features for prediction
        X = np.array([[study, sleep, phone, attendance, difficulty, stress]])
        prob = model.predict_proba(X)[0][1]
        percentage = round(prob * 100, 2)

        # Show result
        st.subheader("📉 Probability of Failing This Course")
        st.metric("Failure Probability", f"{percentage}%")
        st.progress(prob)

        if prob < 0.30:
            st.success("Low Risk")
        elif prob < 0.60:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

        # Generate unique ID for linking later
        prediction_id = str(uuid.uuid4())
        st.info(f"Your prediction ID (keep this to submit your result later): {prediction_id}")

        # Save prediction data with all features
        row = {
            "id": prediction_id,
            "timestamp": datetime.now(),
            "course": course,
            "gpa": gpa,
            "study_hours": study,
            "sleep_hours": sleep,
            "phone_hours": phone,
            "attendance": attendance,
            "difficulty": difficulty,
            "financial_stress": stress,
            "predicted_failure_prob": prob
        }

        df = pd.DataFrame([row])
        if os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)

        st.success("Your data was saved anonymously to improve the model.")


