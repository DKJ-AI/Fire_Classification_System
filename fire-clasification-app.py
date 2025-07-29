import streamlit as st
import numpy as np
import joblib

model = joblib.load("best_fire_detection_model.pkl")    # Load model and scaler
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fire Type Classifier", layout="centered")    # Set page title

st.title(":red[Fire Type] _Classification_  :fire:")    # App title 
st.markdown("### LCS Model for Predicting Fire Type ")      # info
st.markdown("Built using  *MODIS satellite data*  to accurately classify and predict fire types.") 

brightness = st.number_input("Brightness", value=300.0)   # User input
bright_t31 = st.number_input("Bright T31", value=290.0)
frp = st.number_input("Fire Radiative Power (FRP)", value=15.0)
scan = st.slider("Scan", 0.0, 6.0, 1.0)
track = st.slider("Track", 0.0, 10.0, 1.0)
confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"])


confidence_map = {"low": 0, "nominal": 1, "high": 2}    # Map confidence to numeric
confidence_val = confidence_map[confidence]


input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])   # Combine and scale input
scaled_input = scaler.transform(input_data)


if st.button("Predict Fire Type"):    # Predict and display
    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: "Vegetation Fire",
        2: "Other Static Land Source",
        3: "Offshore Fire"
    }

    result = fire_types.get(prediction, "Unknown")
    st.success(f"**Predicted Fire Type:** {result}")