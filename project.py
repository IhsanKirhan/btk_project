import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best_model.pt")  # Load your trained model (best_model.pt)

# Streamlit UI
st.title("YOLO Model Inference with Streamlit")
st.write("Upload a JPG image to get predictions from the YOLO model.")

# File upload
uploaded_file = st.file_uploader("Choose a JPG image", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Use YOLO to make predictions
    results = model(image)  # Perform inference

    # Render the results
    st.image(results.render()[0], caption="Predictions", use_column_width=True)  # Render and display the output image

    # Display predictions (bounding boxes, labels, etc.)
    st.write("Predictions:")
    st.write(f"Labels: {results.names}")
    st.write(f"Boxes: {results.xywh[0]}")  # This is the bounding box information (x, y, width, height)
    st.write(f"Confidence: {results.conf[0]}")  # Confidence scores for each detected object
