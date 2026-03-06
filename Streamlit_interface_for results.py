
import streamlit as st
import pandas as pd
import os
from datetime import datetime

RESULT_LOG = "data/results_log.csv"
PREDICTION_LOG = "data/prediction_log.csv"

st.set_page_config(
    page_title="EKSU Course Result Submission",
    page_icon="📝"
)

st.title("📝 Submit Your Course Result")

st.write(
    "Enter the **Prediction ID** you received when you predicted your course risk, "
    "then submit your actual grade to help improve the model."
)

# Input fields
prediction_id = st.text_input("Enter your Prediction ID")
grade = st.selectbox("Actual Grade", ["Select", "A", "B", "C", "D", "E", "F"])

if st.button("Submit Result"):
    if not prediction_id:
        st.error("Please enter your Prediction ID.")
    elif grade == "Select":
        st.error("Please select a grade.")
    else:
        # Check if ID exists in prediction log
        if not os.path.exists(PREDICTION_LOG):
            st.error("No predictions have been logged yet. Please make a prediction first.")
        else:
            pred_df = pd.read_csv(PREDICTION_LOG)
            if prediction_id not in pred_df['id'].values:
                st.error("Prediction ID not found. Please check your ID and try again.")
            else:
                # Save the result linked to the prediction ID
                row = {
                    "id": prediction_id,
                    "timestamp": datetime.now(),
                    "grade": grade
                }
                df = pd.DataFrame([row])
                if os.path.exists(RESULT_LOG):
                    df.to_csv(RESULT_LOG, mode="a", header=False, index=False)
                else:
                    df.to_csv(RESULT_LOG, index=False)

                st.success(f"Your result for Prediction ID {prediction_id} has been submitted successfully.")

