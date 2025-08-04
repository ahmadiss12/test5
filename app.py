import streamlit as st
import numpy as np
from PIL import Image
from joblib import load

# --- Load Models ---
@st.cache_resource
def load_model(path):
    return load(path)

gender_model = load_model('gender_model.joblib') 
age_model = load_model('age_model.joblib')

# --- Prediction ---
def predict_image(image):
    img_array = np.array(image.resize((64, 64))).flatten()  # Adjust shape as needed
    
    gender = "Female" if gender_model.predict([img_array])[0] == 1 else "Male"
    age = int(age_model.predict([img_array])[0])
    
    return gender, age

# (Keep the same Streamlit UI code from your working version)
