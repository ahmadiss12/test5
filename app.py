import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# --- Load ONNX Models ---
@st.cache_resource
def load_onnx_model(model_path):
    return ort.InferenceSession(model_path)

gender_model = load_onnx_model('gender_model.onnx')  # Converted from .h5
age_model = load_onnx_model('age_model.onnx')       # Converted from .h5

# --- Image Preprocessing ---
def preprocess_image(image):
    img = image.resize((64, 64))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(img_array, axis=0)        # Add batch dimension

def predict_image(image):
    input_data = preprocess_image(image)
    
    # Gender prediction
    gender_input = {gender_model.get_inputs()[0].name: input_data}
    gender_pred = gender_model.run(None, gender_input)[0]
    gender = "Female" if np.argmax(gender_pred) == 1 else "Male"
    
    # Age prediction
    age_input = {age_model.get_inputs()[0].name: input_data}
    age_pred = age_model.run(None, age_input)[0][0][0]
    age = int(age_pred)
    
    return gender, age

# --- Streamlit UI ---
st.title("👨‍🦱👩‍🦰 Age and Gender Prediction (ONNX)")
uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    if st.button('Predict'):
        gender, age = predict_image(image)
        st.success(f"Gender: **{gender}** | Age: **{age} years**")
