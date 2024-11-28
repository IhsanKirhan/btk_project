import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model = YOLO("best_model.pt")  # Replace with the path to your YOLOv8 model
    return model

model = load_model()

# App title and description
st.title("YOLOv8 Classification App")
st.write("Upload an image to classify objects using YOLOv8.")

# Confidence slider
confidence = st.slider("Set Confidence Threshold", 0.0, 1.0, 0.5)

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 classification
    st.write("Running YOLOv8 model...")
    results = model.predict(image, conf=confidence)

    # Process and display results
    st.write("Detection Results:")
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes (xmin, ymin, xmax, ymax)
        scores = result.boxes.conf  # Confidence scores
        classes = result.boxes.cls  # Class indices
        class_names = result.names  # Class names

        for i in range(len(boxes)):
            st.write(
                f"Object: {class_names[int(classes[i])]} | Confidence: {scores[i]:.2f} | Box: {boxes[i].tolist()}"
            )

    # Render and display the annotated image
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Model Output", use_column_width=True)

    # Download button for the annotated image
    st.write("Download the annotated image below:")
    annotated_image_pil = Image.fromarray(annotated_image)
    annotated_image_pil.save("output.jpg")
    with open("output.jpg", "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name="annotated_image.jpg",
            mime="image/jpeg",
        )
