import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("YOLO Object Detection Application")



@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)


model = load_model(model_files["best_model.pt"])


uploaded_file = st.file_uploader("Upload a PNG file", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    

    st.write("### Performing Object Detection...")
    results = model.predict(image)
    
  
    result_image = results[0].plot()  
    st.image(result_image, caption='Detection Result', use_column_width=True)
