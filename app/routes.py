# app/routes.py
from flask import Blueprint, current_app, request, jsonify
import numpy as np
import pandas as pd
from app.database import get_values_from_table


main_bp = Blueprint('main', __name__)


@main_bp.route('/predict_churn', methods=['POST'])
def predict_churn():
    # Get JSON data from the request
    
    model = current_app.config['MODEL']
    
    data = request.get_json()
    features = pd.DataFrame.from_dict([data])
    print(features)
    
    # Check if data is None or empty
    if data is None:
        return jsonify({'error': 'No data provided'}), 400
    
    # Convert JSON data to numpy array (assuming input is a list of dictionaries)
    #input_data = np.array([list(item.values()) for item in data])
    
    # Get predictions from the model
    probabilities = model.predict_heart_disease(features)
    #probability_churn = probabilities.tolist()[0][1]

    
    # Prepare response
    response = {
        'probabilities': probabilities.tolist()[0]
    }
    
    # JSON RESPONSE
    response = {
            'status': 'success',
            'message': 'Prediction successful',
            'probability': f"{probabilities.tolist()[0][1]*100:.2f}%",
        }

    return jsonify(response), 200

@main_bp.route('/model_evaluation', methods=['POST'])
def model_evaluation():
    model = current_app.config['MODEL']
    
    data = request.get_json()
    df = pd.DataFrame.from_dict(data)

    X_eval = df.drop('target', axis = 1)
    y_eval = df['target']
    model.evaluate_model(X_eval, y_eval)
    
    response = {
            'status': 'success',
            'message': 'Evaluation finished with success',
        }

    return jsonify(response), 200