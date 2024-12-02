from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return send_file('templates/index.html')

def load_models():
    logreg = joblib.load('models/logistic_regression.joblib')
    dt = joblib.load('models/decision_tree.joblib')
    rf = joblib.load('models/random_forest.joblib')
    xgb = joblib.load('models/xgboost.joblib')
    return logreg, dt, rf, xgb

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logreg, dt, rf, xgb = load_models()
        data = request.json
        
        # Get features from CSV
        feature_names = list(pd.read_csv('creditcard.csv').drop(['Fraud', 'Index'], axis=1).columns)
        
        # Create data dict
        complete_data = {col: data.get(col, 0) for col in feature_names}
        
        # Create DataFrame
        df = pd.DataFrame([complete_data])
        
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
        
        return jsonify(predictions)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)