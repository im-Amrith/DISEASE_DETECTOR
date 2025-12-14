import os
import json
import sys
import traceback
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template

# Configure TensorFlow to avoid some warnings/errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CRITICAL FIX: Use tf_keras (Legacy Keras 2) to load older .h5 models
try:
    import tf_keras as keras
    print("SUCCESS: Imported tf_keras (Legacy Keras 2) for model compatibility.")
except ImportError:
    try:
        import tensorflow.keras as keras
        print("WARNING: Could not import tf_keras. Falling back to tensorflow.keras.")
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not import keras: {e}", file=sys.stderr)
        sys.exit(1)

try:
    import tensorflow as tf
    print("TensorFlow imported successfully")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import TensorFlow: {e}", file=sys.stderr)
    tf = None

# Fix for "Unrecognized keyword arguments: ['batch_shape']"
# This custom layer intercepts the 'batch_shape' argument which causes errors in newer Keras versions
class FixedInputLayer(keras.layers.InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            # Ensure batch_shape is a tuple/list, not a string
            if isinstance(batch_shape, str):
                # If it's a string, it might be serialized JSON or similar, but usually it should be a list
                # For now, let's try to ignore it if it's a string or handle it if we can parse it
                pass 
            else:
                kwargs['batch_input_shape'] = batch_shape
        super(FixedInputLayer, self).__init__(**kwargs)

# Fix for "Unknown dtype policy: 'DTypePolicy'"
# This mock class handles the Keras 3 DTypePolicy object found in the model config
# and converts it back to a Keras 2 mixed_precision.Policy object
class MockDTypePolicy:
    @classmethod
    def from_config(cls, config):
        policy_name = config.get('name', 'float32')
        return keras.mixed_precision.Policy(policy_name)

app = Flask(__name__)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "Disease.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

model = None
class_indices = {}

def load_resources():
    global model, class_indices
    print(f"Loading resources from {working_dir}...")
    
    # Load Class Indices
    if os.path.exists(class_indices_path):
        try:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            print(f"Class indices loaded: {len(class_indices)} classes")
        except Exception as e:
            print(f"Error loading class indices: {e}", file=sys.stderr)
    else:
        print(f"Error: Class indices file not found at {class_indices_path}", file=sys.stderr)

    # Load Model
    if os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}...")
            # Use the compatible keras loader with custom objects to fix compatibility issues
            custom_objects = {
                'InputLayer': FixedInputLayer,
                'DTypePolicy': MockDTypePolicy
            }
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully")
        except Exception as e:
            print(f"ERROR loading model with load_model: {e}", file=sys.stderr)
            print("Attempting to reconstruct model architecture and load weights...", file=sys.stderr)
            try:
                # Reconstruct the model based on inspection:
                # Sequential:
                # - InputLayer (224, 224, 3)
                # - MobileNetV2 (Functional)
                # - GlobalAveragePooling2D
                # - Dense (dense_4)
                # - Dropout
                # - Dense (dense_5) - Output (7 classes)
                
                base_model = keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights=None # We will load weights from the h5 file
                )
                base_model.trainable = False # Usually base model is frozen
                
                model = keras.Sequential([
                    base_model,
                    keras.layers.GlobalAveragePooling2D(),
                    keras.layers.Dense(128, activation='relu'), # Corrected units based on weight shape (1280, 128)
                    keras.layers.Dropout(0.2), # Guessing dropout rate
                    keras.layers.Dense(len(class_indices), activation='softmax') # Output layer
                ])
                
                # Build the model to initialize weights
                model.build((None, 224, 224, 3))
                
                # Load weights
                model.load_weights(model_path)
                print("SUCCESS: Model reconstructed and weights loaded.")
                
            except Exception as reconstruction_error:
                print(f"CRITICAL ERROR reconstructing model: {reconstruction_error}", file=sys.stderr)
                traceback.print_exc()
    else:
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)

# Load resources on startup
load_resources()

def predict_class(image_file):
    if model is None:
        raise Exception("Model is not loaded")
        
    try:
        img = Image.open(image_file)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        y_p = model.predict(img)
        y_predicted = y_p.argmax(axis=1)
        
        predicted_index = str(y_predicted[0])
        if predicted_index in class_indices:
            return class_indices[predicted_index]
        else:
            return f"Unknown Class ({predicted_index})"
            
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        traceback.print_exc()
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        try:
            prediction = predict_class(file)
            return jsonify({'prediction': prediction})
        except Exception as e:
            error_msg = str(e)
            print(f"Request failed: {error_msg}", file=sys.stderr)
            return jsonify({'error': error_msg}), 500
    
    return jsonify({'error': 'Unknown error'}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
