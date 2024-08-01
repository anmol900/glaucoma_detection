import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle
import tensorflow as tf
import os

# Load the pre-trained CNN model from a pickle file
@st.cache(allow_output_mutation=True)
def load_model(file_path):
    if not os.path.exists(file_path):
        st.error(f"Model file not found at {file_path}")
        return None
    with open(file_path, "rb") as file:
        model = pickle.load(file)
    return model

model_path = "/home/skr/Downloads/model.pkl"  # Update with the correct path
model = load_model(model_path)

# Ensure the model is loaded correctly
if model is not None:
    # Get the input shape expected by the model
    input_shape = model.input_shape[1:3]
else:
    input_shape = (256, 256)  # Default shape, update this if needed

# Define the class labels
class_labels = ['Glaucomaous', 'Non Glaucomaous']  # Update with your actual class labels

# Function to preprocess the image
def preprocess_image(image, target_size):
    img = np.array(image)
    img = cv2.resize(img, target_size)  # Resize to the input size your model expects
    img = img / 255.0  # Normalize to [0, 1] range
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app
st.title("Image Classification with CNN")

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    

    if model is not None:
        preprocessed_image = preprocess_image(image, input_shape)
        prediction = model.predict(preprocessed_image)
        predicted_class = class_labels[np.argmax(prediction)]
        st.write(prediction)
        st.write(f"Predicted class: {predicted_class}")
    else:
        st.write("Model could not be loaded.")




