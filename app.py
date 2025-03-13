import streamlit as st
import pandas as pd
import joblib

# Streamlit app title
st.title("Doctor Survey Prediction")

# Input for usage time
time = st.number_input("Enter the usage time in minutes (e.g., 18.0):")

# Load the model
model = joblib.load("model.pkl")

if st.button("Predict"):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        "Usage Time (mins)": [time],
        "Count of Survey Attempts": [0]  # Assuming 0 attempts initially for prediction
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display the prediction result
    if prediction == 1:
        st.success("The doctor is likely to attend the survey!")
    else:
        st.warning("The doctor is not likely to attend the survey.")

    # Export input and prediction to CSV
    input_data["Prediction"] = prediction
    input_data.to_csv("predicted_doctor.csv", index=False)
    st.download_button(
        "Download Prediction CSV",
        data=input_data.to_csv(index=False),
        file_name="predicted_doctor.csv"
    )
