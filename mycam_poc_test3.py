import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from threading import Thread
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Get a list of model files in the current folder
current_folder = os.getcwd()
model_files = [f for f in os.listdir(current_folder) if f.endswith(".h5")]

# Default model
default_model = "keras_poc.h5"

# Load the initial model
model = load_model(default_model, compile=False)

# Create the array of the right shape to feed into the Keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Global variable for webcam frame and prediction text
frame = None
prediction_text = ""

# Function to preprocess the frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1
    data[0] = normalized_frame
    return data

# Function to continuously capture frames from the webcam
def webcam_capture_function(selected_model):
    global frame, prediction_text

    # Load the selected model
    model_path = os.path.join(current_folder, selected_model)
    model = load_model(model_path, compile=False)

    # Generate label file path based on the selected model
    label_file = os.path.splitext(selected_model)[0] + ".txt"
    label_file_path = os.path.join(current_folder, label_file)
    class_names = open(label_file_path, "r").readlines()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to open the webcam.")
        return

    while True:
        ret, current_frame = cap.read()
        if ret:
            # Resize and preprocess the frame
            data = preprocess_frame(current_frame)

            # Predict the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Update the frame and prediction text
            frame = current_frame.copy()
            prediction_text = f"Prediction: {class_name[2:]} | Confidence: {confidence_score:.4f}"

# Streamlit app
def main():
    global frame, prediction_text

    st.title("Webcam Image Classifier")

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.selectbox("Select Model", model_files, index=model_files.index(default_model))

    # Start the webcam capture thread with selected model
    webcam_thread = Thread(target=webcam_capture_function, args=(selected_model,))
    webcam_thread.start()

    # Placeholder for displaying webcam feed
    webcam_placeholder = st.empty()

    # Placeholder for displaying prediction text
    prediction_placeholder = st.empty()

    while True:
        if frame is not None:
            # Encode the frame to JPEG format
            _, encoded_frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            # Display the webcam feed using st.image
            webcam_placeholder.image(encoded_frame.tobytes(), channels="BGR", use_column_width=True)

            # Display the prediction text
            prediction_placeholder.text(prediction_text)

if __name__ == "__main__":
    main()
