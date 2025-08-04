import streamlit as st
import numpy as np
from PIL import Image

# --- Mock Prediction Function ---
def predict_image(image):
    """Replace this with your actual model logic if needed"""
    # Dummy "model" - returns random results
    gender = "Female" if np.random.rand() > 0.5 else "Male"
    age = int(np.random.uniform(18, 80))
    return gender, age

# --- Streamlit UI ---
st.title("👨‍🦱👩‍🦰 Age and Gender Prediction (Demo)")
st.warning("⚠️ Using mock predictions - add your real model later")

uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    if st.button('Predict'):
        gender, age = predict_image(image)
        st.success(f"Gender: **{gender}** | Age: **{age} years**")
        st.info("Tip: To add real predictions, convert your models to ONNX/Joblib")
