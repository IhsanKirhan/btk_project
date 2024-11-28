import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# Function to predict emotion
def predict_emotion(image, model):

    # Run the model on the image
    with torch.no_grad():
        results = model.predict(image)
    
    # Assuming the model's output is a tensor of predictions
    predicted_class = results[0].argmax(dim=1).item()  # Get the class with the highest score

    # Map class index to emotion label
    emotion_labels = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']  # Adjust as per your model's labels
    return emotion_labels[predicted_class]

# Streamlit UI
st.title("Emotion Prediction from Face")
st.write("Upload an image to predict the emotion:")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model = load_model()

    # Predict the emotion
    emotion = predict_emotion(image, model)
    
    # Display the predicted emotion
    st.write(f"Predicted Emotion: {emotion}")
