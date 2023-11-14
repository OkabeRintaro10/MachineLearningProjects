import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input
import numpy as np

# Load your pretrained model
model = keras.models.load_model("Models/MedicalClassification_ResNet50V2")


def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0, class_index]

    return class_index, confidence


def main():
    st.title("Image Classification App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make prediction
        class_index, confidence = predict_image(uploaded_file)
        labels = {
            0: 'Normal',
            1: 'Affected'
        }
        st.write(f"Prediction: {labels[class_index]}")
        st.write(f"Accuracy: {confidence*100:.2f}%")


if __name__ == "__main__":
    main()
