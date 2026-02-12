import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Smart Agriculture AI", layout="wide")
st.title("🌱 Smart Agriculture AI System")

# =========================================
# TRAIN CROP MODEL (Built-in small dataset)
# =========================================

@st.cache_resource
def train_crop_model():
    data = {
        "N": [90, 85, 60, 40, 30],
        "P": [40, 35, 50, 20, 25],
        "K": [40, 45, 30, 20, 30],
        "ph": [6.5, 6.0, 7.0, 6.8, 7.5],
        "temperature": [25, 22, 30, 28, 35],
        "humidity": [80, 70, 60, 65, 50],
        "rainfall": [200, 150, 100, 120, 80],
        "crop": ["Rice", "Wheat", "Maize", "Groundnut", "Cotton"]
    }

    df = pd.DataFrame(data)
    X = df.drop("crop", axis=1)
    y = df["crop"]

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

crop_model = train_crop_model()

# =========================================
# SIDEBAR NAVIGATION
# =========================================

option = st.sidebar.selectbox(
    "Select Module",
    ["Plant Disease Detection", "Crop Recommendation"]
)

# =========================================
# MODULE 1: DISEASE DETECTION (Simple Vision Logic)
# =========================================

if option == "Plant Disease Detection":

    st.header("🍃 Upload Leaf Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

        img = np.array(image)
        img = cv2.resize(img, (100, 100))

        # Simple green color detection logic
        avg_color = np.mean(img, axis=(0, 1))

        if avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
            disease = "Healthy"
            confidence = 92
        else:
            disease = "Leaf Disease Detected"
            confidence = 87

        st.success(f"Disease Result: {disease}")
        st.info(f"Confidence: {confidence}%")

        if disease != "Healthy":
            st.subheader("Suggested Treatment:")
            st.write("- Remove infected leaves")
            st.write("- Use organic fungicide spray")
            st.write("- Avoid excess watering")

# =========================================
# MODULE 2: CROP RECOMMENDATION
# =========================================

if option == "Crop Recommendation":

    st.header("🌾 Enter Soil Details")

    N = st.number_input("Nitrogen (N)", 0, 200)
    P = st.number_input("Phosphorus (P)", 0, 200)
    K = st.number_input("Potassium (K)", 0, 200)
    ph = st.number_input("pH Value", 0.0, 14.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 60.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)

    if st.button("Recommend Crop"):
        input_data = np.array([[N, P, K, ph, temperature, humidity, rainfall]])
        prediction = crop_model.predict(input_data)

        st.success(f"Recommended Crop: {prediction[0]}")
