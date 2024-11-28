import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best_model.pt")  

def predict_image(image):
    results = model(image)

    return results[0].plot()  

st.title("YOLO Image Prediction")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Prediction Result:")
    result_image = predict_image(image)
    st.image(result_image, caption="Predicted Image with Bounding Boxes", use_column_width=True)

