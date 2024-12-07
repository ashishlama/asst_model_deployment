from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import json
import logging
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Function to load configuration file
def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def get_model_paths(config):
    adaboost_pickle = config.get('adaboost-pickle', '')
    oneclasssvm_pickle = config.get('oneclasssvm-pickle', '')
    ann_pickle = config.get('ann-tanh-pickle', '')
    log_path = config.get('log_path', '')
    return adaboost_pickle, oneclasssvm_pickle, ann_pickle, log_path

config = load_config()

adaboost_pickle, oneclasssvm_pickle, ann_pickle, log_path = get_model_paths(config)

log_filename = 'app.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Write logs to a file
        logging.StreamHandler()  # Also log to the console (optional)
    ]
)

logger = logging.getLogger(__name__)

def load_adaboost_pickle_model():
    logger.info("Loading Adaboost Pickle model from .pkl file...")
    with open(adaboost_pickle, 'rb') as pickle_file:
        pickle_model = pickle.load(pickle_file)
    logger.info("Pickle model loaded successfully.")
    return pickle_model

def load_oneclasssvm_pickle_model():
    logger.info("Loading One-class SVM Pickle model from .pkl file...")
    with open(oneclasssvm_pickle, 'rb') as pickle_file:
        pickle_model = pickle.load(pickle_file)
    logger.info("Pickle model loaded successfully.")
    return pickle_model

def load_ann_h5_model():
    logger.info("Loading ANN with tanh h5 model from .h5 file...")
    pickle_model = load_model(ann_pickle)
    logger.info("Pickle model loaded successfully.")
    return pickle_model

adaboost_pickle_model = load_adaboost_pickle_model()
ocsvm_pickle_model = load_oneclasssvm_pickle_model()
ann_pickle_model = load_ann_h5_model()

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/one_class_svm', methods=['GET'])
def one_class_svm():
    # Render the form page
    return render_template('one_class_svm.html')

@app.route('/ann_tanh', methods=['GET'])
def ann_tanh():
    # Render the form page
    return render_template('ann_tanh.html')

@app.route('/predict_adaboost', methods=['POST'])
def predict_adaboost():
    try:
        # data = request.get_json()

        # features = np.array([data['features']])

        # Accessing form data using request.form
        features = []
        
        # List of form field names for the features (you will need to adjust this based on your actual form fields)
        form_fields = [
            'useful', 'funny', 'cool', 'latitude', 'longitude', 'review_count', 'Alcohol',
            'OutdoorSeating', 'WiFi', 'NoiseLevel', 'Ambience', 'Shopping', 'Beauty & Spas',
            'Food', 'Restaurants', 'Home & Garden', 'Local Services', 'Fashion', 'Home Services',
            'Hair Salons', 'Nail Salons', 'Event Planning & Services', 'Fast Food', 'Health & Medical',
            'Flowers & Gifts', 'Pizza', "Women's Clothing", 'Home Decor', 'Electronics', 'Sandwiches',
            'Furniture Stores', 'Jewelry', 'Accessories', 'Specialty Food', 'IT Services & Computer Repair',
            'Hair Removal', 'Cosmetics & Beauty Supply', 'Florists', 'Sporting Goods', 'Mobile Phones',
            'Arts & Entertainment', 'Eyewear & Opticians', 'Barbers', 'Coffee & Tea', 'Optometrists',
            'Bakeries', "Men's Clothing", 'Arts & Crafts', 'Professional Services', 'Hotels & Travel',
            'Hair Stylists', 'Burgers', 'Waxing', 'Skin Care', 'Grocery', 'Convenience Stores',
            'Mobile Phone Accessories', 'Other', 'Monday_Open', 'Tuesday_Open', 'Wednesday_Open',
            'Thursday_Open', 'Friday_Open', 'Saturday_Open', 'Sunday_Open', 'stars'
        ]

        # Retrieve the form data for each feature
        for field in form_fields:
            # Check if the field exists in the form data, if not set it to 0 or another default value
            value = request.form.get(field, type=float)  # Using float since features are numeric
            features.append(value if value is not None else 0.0)  # Default to 0 if missing or invalid

        # Convert the features list to a numpy array
        features = np.array([features])

        prediction = adaboost_pickle_model.predict(features)

        prediction = int(prediction[0])
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/predict_oneclass_svm', methods=['POST'])
def predict_oneclass_svm():
    try:
        # Retrieving the form data
        features = []
        form_fields = [
            'latitude', 'longitude', 'stars', 'review_count', 'is_open',
            'ByAppointmentOnly', 'BusinessAcceptsCreditCards', 'BikeParking',
            'RestaurantsPriceRange2', 'CoatCheck', 'RestaurantsTakeOut',
            'RestaurantsDelivery', 'Caters', 'WheelchairAccessible', 'HappyHour',
            'OutdoorSeating', 'HasTV', 'RestaurantsReservations', 'DogsAllowed',
            'GoodForKids', 'AcceptsInsurance', 'BYOB', 'Open24Hours',
            'RestaurantsCounterService', 'BusinessParking_garage',
            'BusinessParking_street', 'BusinessParking_lot',
            'BusinessParking_valet'
        ]

        for field in form_fields:
            value = request.form.get(field, type=float)
            features.append(value if value is not None else 0.0)  # Default to 0 if missing or invalid

        # Convert the list to a numpy array
        features = np.array([features])

        # Use the model to make a prediction
        prediction = ocsvm_pickle_model.predict(features)

        # Return the result
        if prediction == -1:
            return jsonify({'status': 'anomaly', 'prediction': prediction.tolist()})
        else:
            return jsonify({'status': 'normal', 'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/predict_ann_tanh', methods=['POST'])
def predict_ann_tanh():
    try:
        features = ['useful', 'funny', 'cool', 'latitude', 'longitude', 'review_count',
            'Alcohol', 'OutdoorSeating', 'WiFi', 'NoiseLevel', 'Ambience',
            'Shopping', 'Beauty & Spas', 'Food', 'Restaurants', 'Home & Garden',
            'Local Services', 'Fashion', 'Home Services', 'Hair Salons',
            'Nail Salons', 'Event Planning & Services', 'Fast Food',
            'Health & Medical', 'Flowers & Gifts', 'Pizza', "Women's Clothing",
            'Home Decor', 'Electronics', 'Sandwiches', 'Furniture Stores',
            'Jewelry', 'Accessories', 'Specialty Food',
            'IT Services & Computer Repair', 'Hair Removal',
            'Cosmetics & Beauty Supply', 'Florists', 'Sporting Goods',
            'Mobile Phones', 'Arts & Entertainment', 'Eyewear & Opticians',
            'Barbers', 'Coffee & Tea', 'Optometrists', 'Bakeries', "Men's Clothing",
            'Arts & Crafts', 'Professional Services', 'Hotels & Travel',
            'Hair Stylists', 'Burgers', 'Waxing', 'Skin Care', 'Grocery',
            'Convenience Stores', 'Mobile Phone Accessories', 'Other',
            'Monday_Open', 'Tuesday_Open', 'Wednesday_Open', 'Thursday_Open',
            'Friday_Open', 'Saturday_Open', 'Sunday_Open']
        
        input_data = {feature: request.form.get(feature) for feature in features}

        # Convert checkbox values (which will be 'on' or None) to 1 or 0
        for feature in input_data:
            if input_data[feature] == 'on':
                input_data[feature] = 1
            else:
                # For non-checkbox fields (e.g., numeric features), keep them as they are
                if input_data[feature] is None:
                    input_data[feature] = 0
                else:
                    input_data[feature] = float(input_data[feature])

        # Prepare the input for the model
        input_array = np.array([list(input_data.values())])

        prediction = ann_pickle_model.predict(input_array)
        
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True, port=8000)
