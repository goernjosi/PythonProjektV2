# Import necessary libraries
from flask import Flask, request, jsonify, send_file  # Flask framework for web application
import pandas as pd  # For data manipulation
import joblib  # For loading saved models

# Initialize Flask application
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    """
    Serve the main HTML page when users access the root URL.
    """
    return send_file('templates/index.html')

def load_models():
    """
    Load all trained models from disk.
    This function is separated to make it reusable and keep code organized.
    
    Returns:
        Tuple of loaded models (logistic regression, decision tree, random forest, xgboost)
    """
    logreg = joblib.load('models/logistic_regression.joblib')
    dt = joblib.load('models/decision_tree.joblib')
    rf = joblib.load('models/random_forest.joblib')
    xgb = joblib.load('models/xgboost.joblib')
    return logreg, dt, rf, xgb

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST requests for fraud predictions.
    Accepts transaction data and returns predictions from all models.
    """
    try:
        # Load all models
        logreg, dt, rf, xgb = load_models()
        
        # Get JSON data from request
        data = request.json
        
        # Get feature names from original dataset
        # Excludes 'Fraud' and 'Index' as they're not used for prediction
        feature_names = list(pd.read_csv('creditcard.csv').drop(['Fraud', 'Index'], axis=1).columns)
        
        # Create a complete data dictionary
        # Uses 0 as default value for any missing features
        complete_data = {col: data.get(col, 0) for col in feature_names}
        
        # Convert dictionary to DataFrame for model input
        df = pd.DataFrame([complete_data])
        
        # Generate predictions from all models
        # For each model:
        # - Calculate probability of fraud
        # - Make binary prediction (Fraud/Not Fraud) using 0.5 threshold
        predictions = {
            'logistic_regression': {
                'probability': float(logreg.predict_proba(df)[0][1]),
                'prediction': 'Fraud' if logreg.predict_proba(df)[0][1] > 0.5 else 'Not Fraud'
            },
            'decision_tree': {
                'probability': float(dt.predict_proba(df)[0][1]),
                'prediction': 'Fraud' if dt.predict_proba(df)[0][1] > 0.5 else 'Not Fraud'
            },
            'random_forest': {
                'probability': float(rf.predict_proba(df)[0][1]),
                'prediction': 'Fraud' if rf.predict_proba(df)[0][1] > 0.5 else 'Not Fraud'
            },
            'xgboost': {
                'probability': float(xgb.predict_proba(df)[0][1]),
                'prediction': 'Fraud' if xgb.predict_proba(df)[0][1] > 0.5 else 'Not Fraud'
            }
        }
        
        # Return predictions as JSON
        return jsonify(predictions)
    
    except Exception as e:
        # Handle any errors and return them with 500 status code
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

# Run the application
if __name__ == '__main__':
    app.run(debug=True)