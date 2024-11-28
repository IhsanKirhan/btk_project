import torch
import streamlit as st
from PIL import Image

# Load your trained model
model = torch.load('best_model.pt')
model.eval()

# Streamlit UI
st.title("Model Inference with Streamlit")
st.write("Upload a JPG image to get predictions from the model.")

# File upload
uploaded_file = st.file_uploader("Choose a JPG image", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Convert image to tensor (if needed for your model)
    image_tensor = torch.tensor(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    # Display the image and the result
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Class: {predicted.item()}")
