from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the models
model_path = os.path.join(os.path.dirname(__file__), 'crop_recommend.joblib')
model_path2 = os.path.join(os.path.dirname(__file__), 'crop_yield.joblib')
h5_model_path = os.path.join(os.path.dirname(__file__), 'plant.h5')

# Initialize models
model = None
model2 = None
h5_model = None

try:
    model = joblib.load(model_path)
    print("Crop recommendation model loaded successfully")
except Exception as e:
    model = None
    print(f"Error loading crop recommendation model: {e}")

try:
    model2 = joblib.load(model_path2)
    print("Crop yield model loaded successfully")
except Exception as e:
    model2 = None
    print(f"Error loading crop yield model: {e}")

try:
    h5_model = load_model(h5_model_path)
    print("H5 model loaded successfully")
except Exception as e:
    h5_model = None
    print(f"Error loading H5 model: {e}")

@app.route('/')
def home():
    return "Welcome to the Flask API!"

@app.route('/api/crop_recommendation/', methods=['POST'])
def get_crop_recommendation():
    if model is None:
        return jsonify({"error": "Crop recommendation model not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    try:
        input_values = [data[feature] for feature in required_features]
        input_values = [float(value) for value in input_values]
        df = pd.DataFrame([input_values], columns=required_features)
        prediction = model.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/yield/', methods=['POST'])
def get_crop_yield():
    if model2 is None:
        return jsonify({"error": "Crop yield model not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    crop_label_encoding = {}  # Define or import your label encoding mappings
    season_label_encoding = {}
    state_label_encoding = {}

    try:
        encoded_data = [
            crop_label_encoding[data["Crop"]],
            season_label_encoding[data["Season"]],
            state_label_encoding[data["State"]],
            float(data["Area"]),
            float(data["Production"]),
            float(data["Fertilizer"]),
            float(data["Pesticide"])
        ]
        df = pd.DataFrame([encoded_data], columns=['Crop', 'Season', 'State', 'Area', 'Production', 'Fertilizer', 'Pesticide'])
        prediction = model2.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/h5_model/', methods=['POST'])
def get_h5_model_prediction():
    if h5_model is None:
        return jsonify({"error": "H5 model not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    image_data = data.get('image')
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode base64 image
        image_data = image_data.split(",")[1]  # Remove 'data:image/jpeg;base64,' part
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image (resize to 64x64 to match the model's input requirements)
        image = image.resize((64, 64))  # Adjust size as needed
        image_array = np.array(image) / 255.0  # Normalize if required
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using the h5 model
        prediction = h5_model.predict(image_array).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
