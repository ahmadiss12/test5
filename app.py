import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# --- PyTorch Model Loading ---
def load_pytorch_model(model_path):
    # Load model architecture and weights
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set to evaluation mode
    return model

# Load models (replace with your actual PyTorch models)
gender_model = load_pytorch_model('gender_model.pth')
age_model = load_pytorch_model('age_model.pth')

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Gender prediction (assuming binary classification)
    with torch.no_grad():
        gender_output = gender_model(img_tensor)
        gender = "Female" if torch.argmax(gender_output).item() == 1 else "Male"
        
        # Age prediction (assuming regression)
        age_output = age_model(img_tensor)
        age = int(age_output.item())
    
    return gender, age

# --- Streamlit UI ---
st.title("👨‍🦱👩‍🦰 Age and Gender Prediction App (PyTorch)")
st.write("Upload a face photo and get predicted age and gender instantly!")

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        gender, age = predict_image(image)
        st.success(f"Predicted Gender: **{gender}**")
        st.success(f"Predicted Age: **{age} years**")
