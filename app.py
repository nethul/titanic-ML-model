# app.py - Your main Flask application
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model when the app starts
try:
    model = joblib.load('titanic_model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({
        'message': 'Titanic Survival Prediction API',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        },
        'usage': 'Send POST request to /predict with JSON data'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expected JSON format:
    {
        "pclass": 1,
        "sex": "female", 
        "age": 25,
        "sibsp": 0,
        "parch": 0
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['pclass', 'sex', 'age', 'sibsp', 'parch']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Convert and validate data types
        try:
            pclass = int(data['pclass'])
            age = float(data['age'])
            sibsp = int(data['sibsp'])
            parch = int(data['parch'])
            sex = str(data['sex']).lower()
            
            # Validate sex input
            if sex not in ['male', 'female']:
                return jsonify({'error': "sex must be 'male' or 'female'"}), 400
                
            # Validate pclass
            if pclass not in [1, 2, 3]:
                return jsonify({'error': 'pclass must be 1, 2, or 3'}), 400
                
        except ValueError as e:
            return jsonify({'error': f'Invalid data type: {str(e)}'}), 400
        
        # Prepare input for model (same format as training)
        input_data = [[pclass, age, sibsp, parch, 1 if sex == 'male' else 0]]
        
        # Make prediction
        probability = model.predict_proba(input_data)[0][1]
        survival_chance = round(probability * 100, 2)
        
        # Determine survival status
        status = "Survive" if survival_chance > 50 else "Not Survive"
        
        # Return prediction
        return jsonify({
            'survival_probability': survival_chance,
            'survival_status': status,
            'input_data': {
                'pclass': pclass,
                'sex': sex,
                'age': age,
                'sibsp': sibsp,
                'parch': parch
            },
            'message': 'Success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)