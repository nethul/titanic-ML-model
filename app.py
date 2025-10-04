from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Or specify your frontend URL: ["http://localhost:5173", "https://your-frontend.com"]
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def train_new_model():
    """Always train a fresh model to avoid version conflicts"""
    logger.info("🔄 Training new model...")
    try:
        # Load Titanic dataset
        titanic = fetch_openml('titanic', version=1, as_frame=True)
        X, y = titanic.data, titanic.target
        
        # Simple preprocessing
        X = X[['pclass', 'sex', 'age', 'sibsp', 'parch']]
        X = pd.get_dummies(X, columns=['sex'], drop_first=True)
        X = X.fillna(X.mean())
        
        # Train model with optimized parameters
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=1,
            max_depth=5
        )
        model.fit(X, y)
        
        logger.info("✅ New model trained successfully!")
        return model
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        return None

# Train a fresh model on startup
model = train_new_model()

@app.route('/')
def home():
    return jsonify({
        'message': 'Titanic Survival Prediction API',
        'model_loaded': model is not None,
        'status': 'ready' if model is not None else 'training_failed'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    if model is None:
        return jsonify({'error': 'Model not available. Service unavailable.'}), 503
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['pclass', 'sex', 'age', 'sibsp', 'parch']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Convert and validate data
        try:
            pclass = int(data['pclass'])
            age = float(data['age'])
            sibsp = int(data['sibsp'])
            parch = int(data['parch'])
            sex = str(data['sex']).lower()
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid data types: {str(e)}'}), 400
        
        if sex not in ['male', 'female']:
            return jsonify({'error': "sex must be 'male' or 'female'"}), 400
        if pclass not in [1, 2, 3]:
            return jsonify({'error': "pclass must be 1, 2, or 3"}), 400
        
        # Prepare input for model
        input_data = [[pclass, age, sibsp, parch, 1 if sex == 'male' else 0]]
        
        # Make prediction
        probability = model.predict_proba(input_data)[0][1]
        survival_chance = round(probability * 100, 2)
        status = "Survive" if survival_chance > 50 else "Not Survive"
        
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
            'message': 'Prediction successful'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
