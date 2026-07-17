import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mnist.h5")

# Class names
class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

st.title("MNIST Digit Classifier")

st.write("Upload an image of a handwritten digit (0-9).")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=200)

    # Resize image
    image = image.resize((28, 28))

    # Convert to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Invert colors (optional)
    img_array = 1 - img_array

    # Reshape
    img_array = img_array.reshape(1, 28, 28)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(f"Predicted Digit: {predicted_class}")

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )