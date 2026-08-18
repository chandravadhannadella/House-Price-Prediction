"""
House Price Prediction - Flask REST API Deployment

This Flask application loads the trained Linear Regression model and provides
a REST API for making house price predictions.
"""

import os
import sys
import json
import traceback
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template


# Initialize Flask app
app = Flask(__name__)

# Global variables for model and feature info
MODEL = None
FEATURE_NAMES = None
PROCESSED_DF = None
REQUIRED_FEATURES = ['Area', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft']
CATEGORICAL_FEATURES = ['Furnishing', 'Locality', 'Status', 'Transaction', 'Type']


def load_model_and_features():
    """
    Load the trained model and feature information from the processed dataset.
    This is called when the Flask app starts.
    
    Returns:
        Tuple of (model, feature_names, processed_df)
        
    Raises:
        FileNotFoundError: If model or dataset files don't exist
        Exception: If loading fails
    """
    global MODEL, FEATURE_NAMES, PROCESSED_DF
    
    model_path = "models/house_price_model.pkl"
    processed_data_path = "data/processed/cleaned_house_prices.csv"
    
    try:
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        MODEL = joblib.load(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
        print(f"  Model type: {type(MODEL).__name__}")
        print(f"  Model features: {len(MODEL.coef_)}")
        
        # Load processed dataset to get feature names
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed dataset not found at: {processed_data_path}")
        
        PROCESSED_DF = pd.read_csv(processed_data_path)
        print(f"✓ Processed dataset loaded: {PROCESSED_DF.shape}")
        
        # Extract feature names (excluding target 'Price')
        FEATURE_NAMES = [col for col in PROCESSED_DF.columns if col != 'Price']
        print(f"✓ Feature names extracted: {len(FEATURE_NAMES)} features")
        print(f"  First 5 features: {FEATURE_NAMES[:5]}")
        print(f"  Last 5 features: {FEATURE_NAMES[-5:]}")
        
        return MODEL, FEATURE_NAMES, PROCESSED_DF
    
    except Exception as e:
        print(f"❌ Error loading model/features: {str(e)}", file=sys.stderr)
        raise


def create_feature_vector(input_data: dict) -> np.ndarray:
    """
    Convert user input to the exact feature vector expected by the model.
    
    The model expects all 374 features in the exact order from training.
    This function takes basic house parameters and creates the feature vector
    through one-hot encoding of categorical features.
    
    Args:
        input_data: Dictionary with house parameters
        
    Returns:
        NumPy array of features in the correct order for the model
        
    Raises:
        ValueError: If required features are missing or invalid
    """
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])
    
    # Start with the basic numerical features
    feature_vector = []
    
    # Add numerical features in the correct order
    numerical_features = ['Area', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft']
    for feat in numerical_features:
        if feat not in input_data:
            raise ValueError(f"Missing required numerical feature: {feat}")
        
        value = input_data[feat]
        try:
            feature_vector.append(float(value))
        except (ValueError, TypeError):
            raise ValueError(f"{feat} must be a valid number, got: {value}")
    
    # Create a binary feature matrix for all known categorical combinations
    # We need to match the exact features from the trained model
    
    # For categorical features, we need to check against the processed dataset
    # to determine which binary features should be set to 1
    
    # Extract categorical inputs
    furnishing = input_data.get('Furnishing', '').strip()
    locality = input_data.get('Locality', '').strip()
    status = input_data.get('Status', '').strip()
    transaction = input_data.get('Transaction', '').strip()
    property_type = input_data.get('Type', '').strip()
    
    # Build categorical feature flags
    categorical_flags = {}
    
    # Furnishing flags
    if furnishing:
        categorical_flags['Furnishing_Semi-Furnished'] = 1 if furnishing.lower() == 'semi-furnished' else 0
        categorical_flags['Furnishing_Unfurnished'] = 1 if furnishing.lower() == 'unfurnished' else 0
    else:
        categorical_flags['Furnishing_Semi-Furnished'] = 0
        categorical_flags['Furnishing_Unfurnished'] = 0
    
    # Locality flag
    if locality:
        locality_key = f'Locality_{locality}'
    else:
        locality_key = None
    
    # Status flag
    if status:
        status_key = f'Status_{status.replace(" ", "_")}'
    else:
        status_key = None
    
    # Transaction flag
    if transaction:
        transaction_key = f'Transaction_{transaction.replace(" ", "_")}'
    else:
        transaction_key = None
    
    # Type flag
    if property_type:
        type_key = f'Type_{property_type.replace(" ", "_")}'
    else:
        type_key = None
    
    # Now add all remaining features in the correct order
    for feature in FEATURE_NAMES[5:]:  # Skip the first 5 numerical features already added
        if feature in categorical_flags:
            feature_vector.append(categorical_flags[feature])
        elif feature == locality_key:
            feature_vector.append(1)
        elif feature == status_key:
            feature_vector.append(1)
        elif feature == transaction_key:
            feature_vector.append(1)
        elif feature == type_key:
            feature_vector.append(1)
        else:
            # All other categorical features are 0
            feature_vector.append(0)
    
    return np.array([feature_vector])


@app.route('/', methods=['GET'])
def home():
    """Render the house price prediction web interface."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Accepts POST request with house parameters as JSON and returns predicted price.
    
    Expected JSON format:
    {
        "Area": 1500,
        "BHK": 3,
        "Bathroom": 2,
        "Parking": 1,
        "Per_Sqft": 8500,
        "Furnishing": "Semi-Furnished",
        "Locality": "Dwarka",
        "Status": "Ready_to_move",
        "Transaction": "Resale",
        "Type": "Apartment"
    }
    
    Returns:
        JSON response with predicted price or error message
    """
    try:
        # Check if model is loaded
        if MODEL is None or FEATURE_NAMES is None:
            return jsonify({
                "status": "error",
                "message": "Model not loaded. Server initialization failed."
            }), 500
        
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Request must be JSON"
            }), 400
        
        input_data = request.get_json()
        
        # Validate required numerical features
        required_fields = ['Area', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft']
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "required_fields": required_fields,
                "optional_fields": CATEGORICAL_FEATURES
            }), 400
        
        # Validate that numerical values are numeric
        for field in required_fields:
            try:
                float(input_data[field])
            except (ValueError, TypeError):
                return jsonify({
                    "status": "error",
                    "message": f"{field} must be a valid number"
                }), 400
        
        # Create feature vector
        try:
            features = create_feature_vector(input_data)
        except ValueError as e:
            return jsonify({
                "status": "error",
                "message": f"Invalid input: {str(e)}"
            }), 400
        
        # Make prediction
        predicted_price = MODEL.predict(features)[0]
        
        # Ensure prediction is a Python native type (not numpy)
        predicted_price = float(predicted_price)
        
        return jsonify({
            "status": "success",
            "predicted_price": predicted_price,
            "predicted_price_formatted": f"₹{predicted_price:,.2f}",
            "input_summary": {
                "Area": input_data.get('Area'),
                "BHK": input_data.get('BHK'),
                "Bathroom": input_data.get('Bathroom'),
                "Parking": input_data.get('Parking'),
                "Per_Sqft": input_data.get('Per_Sqft')
            }
        }), 200
    
    except Exception as e:
        # Log the full error for debugging
        print(f"❌ Prediction error: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        
        return jsonify({
            "status": "error",
            "message": "An error occurred during prediction",
            "error_type": type(e).__name__
        }), 500


@app.route('/info', methods=['GET'])
def get_info():
    """
    Get information about the model and expected input format.
    
    Returns:
        JSON with model info and API documentation
    """
    return jsonify({
        "status": "success",
        "model_info": {
            "type": "Linear Regression",
            "features": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
            "loaded": MODEL is not None
        },
        "api_endpoints": {
            "health_check": "GET /",
            "prediction": "POST /predict",
            "info": "GET /info"
        },
        "required_fields": {
            "Area": "float or int (in square feet)",
            "BHK": "integer (number of bedrooms)",
            "Bathroom": "float or int",
            "Parking": "integer (number of parking spaces)",
            "Per_Sqft": "float or int (price per square foot)"
        },
        "optional_fields": {
            "Furnishing": "string (e.g., 'Semi-Furnished', 'Unfurnished')",
            "Locality": "string (e.g., 'Dwarka')",
            "Status": "string (e.g., 'Ready_to_move')",
            "Transaction": "string (e.g., 'Resale')",
            "Type": "string (e.g., 'Apartment')"
        },
        "example_request": {
            "Area": 1200,
            "BHK": 3,
            "Bathroom": 2,
            "Parking": 1,
            "Per_Sqft": 8500,
            "Furnishing": "Semi-Furnished",
            "Locality": "Dwarka",
            "Status": "Ready_to_move",
            "Transaction": "Resale",
            "Type": "Apartment"
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": ["/", "/predict", "/info"]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "status": "error",
        "message": "Method not allowed"
    }), 405


if __name__ == '__main__':
    print("="*60)
    print("HOUSE PRICE PREDICTION - FLASK API")
    print("="*60)
    
    try:
        # Load model and features before starting server
        print("\nLoading model and features...")
        load_model_and_features()
        
        print("\n" + "="*60)
        print("✓ INITIALIZATION COMPLETE")
        print("="*60)
        print("\nAPI Endpoints:")
        print("  - Health Check: GET http://127.0.0.1:5000/")
        print("  - Predict:      POST http://127.0.0.1:5000/predict")
        print("  - Info:         GET http://127.0.0.1:5000/info")
        print("\nStarting Flask server...")
        print("="*60 + "\n")
        
        # Run Flask app in development mode
        app.run(host='127.0.0.1', port=5000, debug=True)
    
    except Exception as e:
        print(f"\n❌ Failed to start API: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
