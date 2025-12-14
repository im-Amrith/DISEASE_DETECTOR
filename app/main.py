import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))   # it provide absolute path for app directory
model_path = f"{working_dir}/trained_model/Disease.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def predict_class(model,path,class_indices):
   img1 = Image.open(path)
   img1 = img1.resize((224,224))
   img1 = img1.convert('RGB')
   img1 = np.array(img1)
   img1 = img1 / 255.0  # optional depending on model preprocessing

# Add batch dimension ---> important step (1,224,224,3)
   img1 = np.expand_dims(img1, axis=0)
# Predict
   y_p = model.predict(img1)
   y_predicted = y_p.argmax(axis=1)
   return class_indices[str(y_predicted[0])]


# Streamlit App
st.title('🔍Livestock Disease Predictor🔍')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((250, 250))
        st.image(resized_img)

    with col2:
        if st.button('Predict'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_class(model,uploaded_image,class_indices)
            st.success(f'Prediction: {str(prediction)}')