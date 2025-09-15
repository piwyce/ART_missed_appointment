import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("RandomForest_missed_appointment.joblib")

model = load_model()

st.title("📊 Missed ART Appointment Predictor")
st.markdown(
    """
    Enter patient details below to estimate the probability that
    a patient will **miss their next ART appointment**.
    """
)

# Collect user inputs
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=35)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

regimen_group = st.sidebar.selectbox(
    "Regimen Group", ["1st Line", "2nd Line", "3rd Line", "Other"]
)

months_prescription = st.sidebar.slider("Months of Prescription", 1, 6, 3)

days_since_last = st.sidebar.number_input("Days Since Last Visit", min_value=0, value=30)

appointment_gap = st.sidebar.number_input("Appointment Gap (days)", min_value=0, value=30)

fast_track = st.sidebar.selectbox("Differentiated care model", ["Fast Track-Facility ART distribution group", "Standard Care-Facility ART distribution group", "Standard Care-Community ART distribution group", "Fast Track-Community ART distribution group"])

has_ncd = st.sidebar.selectbox("Has NCD (comorbidity)?", ["Yes", "No"])

# Create a dictionary from inputs, matching the feature names used during training
input_dict = {
    "Age_final": age,
    "Sex": gender[0], # Use 'M' or 'F' based on the selectbox
    "Regimen_group": regimen_group[0], # Use the first character as in training
    "Months of Prescription": months_prescription,
    "Days_since_last_visit": days_since_last,
    "Appointment_gap": appointment_gap,
    "Differentiated care model": fast_track,
    "Has_NCD": 1 if has_ncd == "Yes" else 0, # Convert Yes/No to 1/0
    "Age_group": "Unknown", # Placeholder - Age_group is derived in preprocessing
    "Current Regimen": "Unknown" # Placeholder - Current Regimen is not directly used after feature engineering
}

input_df = pd.DataFrame([input_dict])

# Reorder columns to match the training data's feature order
# This is crucial for the ColumnTransformer
feature_order = ['Sex', 'Age_final', 'Age_group', 'Current Regimen',
                 'Months of Prescription', 'Differentiated care model',
                 'Days_since_last_visit', 'Appointment_gap',
                 'Regimen_group', 'Has_NCD']

input_df = input_df[feature_order]


# Make Predictions
if st.button("Predict Risk"):
    # The loaded pipeline handles preprocessing internally
    probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"⚠️ High risk of missed appointment. Probability: {probability:.2%}")
    else:
        st.success(f"✅ Likely to attend. Probability of missing: {probability:.2%}")

    st.caption("Probability is the model-estimated chance of missing the next appointment.")
