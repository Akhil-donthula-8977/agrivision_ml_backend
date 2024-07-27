# # from flask import Flask, jsonify, request
# # import joblib
# # import os
# # import pandas as pd
# # from label_encodings import crop_label_encoding, season_label_encoding, state_label_encoding
# # from flask_cors import CORS

# # app = Flask(__name__)
# # CORS(app)  # Enable CORS for all routes

# # # Load the models
# # model_path = os.path.join(os.path.dirname(__file__), 'crop_recommend.joblib')
# # model_path2 = os.path.join(os.path.dirname(__file__), 'crop_yield.joblib')

# # # Initialize models
# # model = None
# # model2 = None

# # try:
# #     model = joblib.load(model_path)
# #     print("Crop recommendation model loaded successfully")
# # except Exception as e:
# #     print(f"Error loading crop recommendation model: {e}")

# # try:
# #     model2 = joblib.load(model_path2)
# #     print("Crop yield model loaded successfully")
# # except Exception as e:
# #     print(f"Error loading crop yield model: {e}")

# # @app.route('/')
# # def home():
# #     return "Welcome to the Flask API!"

# # @app.route('/api/crop_recommendation/', methods=['POST'])
# # def get_crop_recommendation():
# #     if model is None:
# #         return jsonify({"error": "Model not loaded properly"}), 500

# #     all_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
# #     data_columns = ['N', 'P', 'K']
# #     float_columns = ['temperature', 'humidity', 'ph', 'rainfall']
    
# #     input_values = []
# #     data = request.get_json()

# #     if not data:
# #         return jsonify({"error": "Invalid input"}), 400

# #     try:
# #         for feature in data_columns:
# #             input_values.append(data[feature])
# #         for feature in float_columns:
# #             input_values.append(float(data[feature]))
        
# #         ans = model.predict(pd.DataFrame([input_values], columns=all_columns))
# #         ans_list = ans.tolist()
# #         return jsonify({"prediction": ans_list})
# #     except KeyError as e:
# #         return jsonify({"error": f"Missing feature: {str(e)}"}), 400
# #     except ValueError as e:
# #         return jsonify({"error": f"Invalid value type: {str(e)}"}), 400
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # @app.route('/api/yield/', methods=['POST'])
# # def get_crop_yield():
# #     if model2 is None:
# #         return jsonify({"error": "Model not loaded properly"}), 500

# #     all_columns = ['Crop', 'Season', 'State', 'Area', 'Production', 'Fertilizer', 'Pesticide']
# #     int_columns = ['Crop', 'Season', 'State']
# #     float_columns = ['Area', 'Production', 'Fertilizer', 'Pesticide']
    
# #     input_values = []
# #     data = request.get_json()
# #     data["Crop"] = crop_label_encoding[data["Crop"]]
# #     data["Season"] = season_label_encoding[data["Season"]]
# #     data["State"] = state_label_encoding[data["State"]]

# #     if not data:
# #         return jsonify({"error": "Invalid input"}), 400

# #     try:
# #         # Process integer columns
# #         for feature in int_columns:
# #             input_values.append(int(data[feature]))
# #         # Process float columns
# #         for feature in float_columns:
# #             input_values.append(float(data[feature]))
        
# #         # Make prediction
# #         ans = model2.predict(pd.DataFrame([input_values], columns=all_columns))
# #         ans_list = ans.tolist()
# #         return jsonify({"prediction": ans_list})
# #     except KeyError as e:
# #         return jsonify({"error": f"Missing feature: {str(e)}"}), 400
# #     except ValueError as e:
# #         return jsonify({"error": f"Invalid value type: {str(e)}"}), 400
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500


# # if __name__ == '__main__':
# #     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
# from flask import Flask, jsonify, request
# import joblib
# import os
# import pandas as pd
# from label_encodings import crop_label_encoding, season_label_encoding, state_label_encoding
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Model file paths
# model_path = os.path.join(os.path.dirname(__file__), 'crop_recommendation.joblib')
# model_path2 = os.path.join(os.path.dirname(__file__), 'crop_yield.joblib')

# # Initialize models
# model = None
# model2 = None

# # Load models
# try:
#     if os.path.exists(model_path):
#         model = joblib.load(model_path)
#         print("Crop recommendation model loaded successfully")
#     else:
#         print(f"Model file not found: {model_path}")
# except Exception as e:
#     print(f"Error loading crop recommendation model: {e}")

# try:
#     if os.path.exists(model_path2):
#         model2 = joblib.load(model_path2)
#         print("Crop yield model loaded successfully")
#     else:
#         print(f"Model file not found: {model_path2}")
# except Exception as e:
#     print(f"Error loading crop yield model: {e}")

# @app.route('/')
# def home():
#     return "Welcome to the Flask API!"

# @app.route('/api/crop_recommendation/', methods=['POST'])
# def get_crop_recommendation():
#     if model is None:
#         return jsonify({"error": "Model not loaded properly"}), 500

#     all_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#     data_columns = ['N', 'P', 'K']
#     float_columns = ['temperature', 'humidity', 'ph', 'rainfall']
    
#     input_values = []
#     data = request.get_json()

#     if not data:
#         return jsonify({"error": "Invalid input"}), 400

#     try:
#         for feature in data_columns:
#             input_values.append(data[feature])
#         for feature in float_columns:
#             input_values.append(float(data[feature]))
        
#         ans = model.predict(pd.DataFrame([input_values], columns=all_columns))
#         ans_list = ans.tolist()
#         return jsonify({"prediction": ans_list})
#     except KeyError as e:
#         return jsonify({"error": f"Missing feature: {str(e)}"}), 400
#     except ValueError as e:
#         return jsonify({"error": f"Invalid value type: {str(e)}"}), 400
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/api/yield/', methods=['POST'])
# def get_crop_yield():
#     if model2 is None:
#         return jsonify({"error": "Model not loaded properly"}), 500

#     all_columns = ['Crop', 'Season', 'State', 'Area', 'Production', 'Fertilizer', 'Pesticide']
#     int_columns = ['Crop', 'Season', 'State']
#     float_columns = ['Area', 'Production', 'Fertilizer', 'Pesticide']
    
#     input_values = []
#     data = request.get_json()
#     data["Crop"] = crop_label_encoding.get(data["Crop"], -1)
#     data["Season"] = season_label_encoding.get(data["Season"], -1)
#     data["State"] = state_label_encoding.get(data["State"], -1)

#     if not data:
#         return jsonify({"error": "Invalid input"}), 400

#     try:
#         # Process integer columns
#         for feature in int_columns:
#             input_values.append(int(data[feature]))
#         # Process float columns
#         for feature in float_columns:
#             input_values.append(float(data[feature]))
        
#         # Make prediction
#         ans = model2.predict(pd.DataFrame([input_values], columns=all_columns))
#         ans_list = ans.tolist()
#         return jsonify({"prediction": ans_list})
#     except KeyError as e:
#         return jsonify({"error": f"Missing feature: {str(e)}"}), 400
#     except ValueError as e:
#         return jsonify({"error": f"Invalid value type: {str(e)}"}), 400
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
from flask import Flask, jsonify, request
import joblib
import os
import pandas as pd
import requests
from label_encodings import crop_label_encoding, season_label_encoding, state_label_encoding
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model file paths
model_path = os.path.join(os.path.dirname(__file__), 'crop_recommendation.joblib')
model_path2 = os.path.join(os.path.dirname(__file__), 'crop_yield.joblib')

# Function to download files from Google Drive
def download_file_from_google_drive(gdrive_id, local_filename):
    URL = f"https://drive.google.com/uc?export=download&id={gdrive_id}"
    response = requests.get(URL, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Google Drive file IDs (replace with your actual file IDs)
model_gdrive_id ='1c7vXu8OhoNNOvsLp8lsNoIaR0uCkCVm3'
model_gdrive_id2 = '1jRCjhEgHC70DVEs48YR3pfEmovxliJcU'

# Initialize models
model = None
model2 = None

# Attempt to load models from local files, otherwise download
try:
    if not os.path.exists(model_path):
        print("Model file not found locally, downloading...")
        download_file_from_google_drive(model_gdrive_id, model_path)
    model = joblib.load(model_path)
    print("Crop recommendation model loaded successfully")
except Exception as e:
    print(f"Error loading crop recommendation model: {e}")
    model = None

try:
    if not os.path.exists(model_path2):
        print("Model file not found locally, downloading...")
        download_file_from_google_drive(model_gdrive_id2, model_path2)
    model2 = joblib.load(model_path2)
    print("Crop yield model loaded successfully")
except Exception as e:
    print(f"Error loading crop yield model: {e}")
    model2 = None

@app.route('/')
def home():
    return "Welcome to the Flask API!"

@app.route('/api/files', methods=['GET'])
def list_files():
    files = os.listdir('.')
    return jsonify({"files": files})

@app.route('/api/crop_recommendation/', methods=['POST'])
def get_crop_recommendation():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    all_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    data_columns = ['N', 'P', 'K']
    float_columns = ['temperature', 'humidity', 'ph', 'rainfall']
    
    input_values = []
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        for feature in data_columns:
            input_values.append(data[feature])
        for feature in float_columns:
            input_values.append(float(data[feature]))
        
        ans = model.predict(pd.DataFrame([input_values], columns=all_columns))
        ans_list = ans.tolist()
        return jsonify({"prediction": ans_list})
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value type: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/yield/', methods=['POST'])
def get_crop_yield():
    if model2 is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    all_columns = ['Crop', 'Season', 'State', 'Area', 'Production', 'Fertilizer', 'Pesticide']
    int_columns = ['Crop', 'Season', 'State']
    float_columns = ['Area', 'Production', 'Fertilizer', 'Pesticide']
    
    input_values = []
    data = request.get_json()
    
    # Ensure the crop, season, and state values are encoded
    data["Crop"] = crop_label_encoding.get(data["Crop"], -1)
    data["Season"] = season_label_encoding.get(data["Season"], -1)
    data["State"] = state_label_encoding.get(data["State"], -1)

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        # Process integer columns
        for feature in int_columns:
            input_values.append(int(data[feature]))
        # Process float columns
        for feature in float_columns:
            input_values.append(float(data[feature]))
        
        # Make prediction
        ans = model2.predict(pd.DataFrame([input_values], columns=all_columns))
        ans_list = ans.tolist()
        return jsonify({"prediction": ans_list})
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value type: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
