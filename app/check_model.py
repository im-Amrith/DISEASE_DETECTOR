import os
import tensorflow as tf
import numpy as np
from PIL import Image
import json

def check_model():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, "trained_model", "Disease.h5")
    class_indices_path = os.path.join(working_dir, "class_indices.json")

    print(f"Checking paths:")
    print(f"Model path: {model_path}")
    print(f"Class indices path: {class_indices_path}")

    if not os.path.exists(model_path):
        print("ERROR: Model file not found!")
        return
    
    if not os.path.exists(class_indices_path):
        print("ERROR: Class indices file not found!")
        return

    print("\nLoading class indices...")
    try:
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        print(f"Success! Loaded {len(class_indices)} classes.")
    except Exception as e:
        print(f"ERROR loading class indices: {e}")
        return

    print("\nLoading model (this might take a while)...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Success! Model loaded.")
        model.summary()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    print("\nCreating dummy image for prediction test...")
    try:
        # Create a black image
        img = Image.new('RGB', (224, 224), color = 'black')
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        print("Running prediction...")
        y_p = model.predict(img)
        print(f"Prediction raw output shape: {y_p.shape}")
        y_predicted = y_p.argmax(axis=1)
        print(f"Predicted class index: {y_predicted[0]}")
        print(f"Predicted label: {class_indices[str(y_predicted[0])]}")
        print("\nVERIFICATION SUCCESSFUL: Model and prediction logic are working.")
    except Exception as e:
        print(f"ERROR during prediction: {e}")

if __name__ == "__main__":
    check_model()
