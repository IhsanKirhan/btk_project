import streamlit as st
from PIL import Image
from ultralytics import YOLO


model = YOLO("best_model.pt")


uploaded_file = st.file_uploader("Upload a face photo (PNG)", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image', use_column_width=True)
    

    st.write("Emotion detection...")
    results = model.predict(image)

    result_image = results[0].plot()  
    st.image(result_image, caption='Result:', use_column_width=True)
