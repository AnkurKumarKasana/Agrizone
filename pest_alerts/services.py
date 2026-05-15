"""
Pest Alert Prediction Service
===============================
ML-based pest risk prediction using RF + GBM ensemble.
Integrates with weather data for real-time predictions.
"""

import numpy as np
import pickle
import json
import os
import logging
import requests

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models', 'saved_models')

# Cached instances
_rf_model = None
_gb_model = None
_scaler = None
_crop_encoder = None
_pest_encoder = None
_severity_encoder = None
_pest_info_db = None


def _load_models():
    """Load and cache pest prediction models."""
    global _rf_model, _gb_model, _scaler, _crop_encoder, _pest_encoder, _severity_encoder, _pest_info_db

    if _rf_model is not None:
        return True

    paths = {
        'rf': os.path.join(MODEL_DIR, 'pest_rf_model.pkl'),
        'gb': os.path.join(MODEL_DIR, 'pest_gb_severity_model.pkl'),
        'scaler': os.path.join(MODEL_DIR, 'pest_scaler.pkl'),
        'crop_enc': os.path.join(MODEL_DIR, 'pest_crop_encoder.pkl'),
        'pest_enc': os.path.join(MODEL_DIR, 'pest_name_encoder.pkl'),
        'sev_enc': os.path.join(MODEL_DIR, 'pest_severity_encoder.pkl'),
        'info': os.path.join(MODEL_DIR, 'pest_info_database.json'),
    }

    # Attempt 1: Load existing models
    try:
        if not os.path.exists(paths['rf']):
            raise FileNotFoundError("Pest model not found")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(paths['rf'], 'rb') as f:
                _rf_model = pickle.load(f)
            with open(paths['gb'], 'rb') as f:
                _gb_model = pickle.load(f)
            with open(paths['scaler'], 'rb') as f:
                _scaler = pickle.load(f)
            with open(paths['crop_enc'], 'rb') as f:
                _crop_encoder = pickle.load(f)
            with open(paths['pest_enc'], 'rb') as f:
                _pest_encoder = pickle.load(f)
            with open(paths['sev_enc'], 'rb') as f:
                _severity_encoder = pickle.load(f)

        if os.path.exists(paths['info']):
            with open(paths['info'], 'r') as f:
                _pest_info_db = json.load(f)
        else:
            _pest_info_db = {}

        logger.info("Pest prediction models loaded successfully.")
        return True

    except Exception as e:
        logger.warning("Existing pest model incompatible (%s). Retraining...", e)
        _rf_model = None
        _gb_model = None

    # Attempt 2: Retrain models with current sklearn version
    try:
        logger.info("Retraining pest models for sklearn compatibility...")
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'ml_models'))
        from train_pest_model import train_pest_models
        train_pest_models()

        # Reload freshly trained models
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(paths['rf'], 'rb') as f:
                _rf_model = pickle.load(f)
            with open(paths['gb'], 'rb') as f:
                _gb_model = pickle.load(f)
            with open(paths['scaler'], 'rb') as f:
                _scaler = pickle.load(f)
            with open(paths['crop_enc'], 'rb') as f:
                _crop_encoder = pickle.load(f)
            with open(paths['pest_enc'], 'rb') as f:
                _pest_encoder = pickle.load(f)
            with open(paths['sev_enc'], 'rb') as f:
                _severity_encoder = pickle.load(f)

        if os.path.exists(paths['info']):
            with open(paths['info'], 'r') as f:
                _pest_info_db = json.load(f)
        else:
            _pest_info_db = {}

        logger.info("Pest models retrained and loaded successfully.")
        return True

    except Exception as e:
        logger.error("Failed to retrain pest models: %s", e)
        _rf_model = None
        return False


def get_weather_data(location, api_key=None):
    """Fetch real weather data from OpenWeatherMap API."""
    try:
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('OPENWEATHER_API_KEY', '')

        if not api_key:
            return None

        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return {
                'temp': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data.get('wind', {}).get('speed', 10),
                'rainfall': data.get('rain', {}).get('1h', 0) * 24,  # Estimated daily
            }
    except Exception as e:
        logger.warning("Weather API error: %s", e)
    return None


def predict_pest_risk(crop, weather_data):
    """
    Predict pest risks for a crop given weather conditions.
    
    Returns:
        dict with success, alerts list, weather data, and risk summary
    """
    if not _load_models():
        return {'success': False, 'error': 'Pest models not loaded. Run training first.'}

    try:
        crop_lower = crop.lower().strip()

        # Check if crop is supported
        if crop_lower not in _crop_encoder.classes_:
            supported = ', '.join(c.title() for c in _crop_encoder.classes_ if c != 'None')
            return {
                'success': False,
                'error': f'Crop "{crop}" not supported. Supported: {supported}'
            }

        temp = weather_data.get('temp', 25)
        humidity = weather_data.get('humidity', 70)
        rainfall = weather_data.get('rainfall', 50)
        wind_speed = weather_data.get('wind_speed', 10)

        import datetime
        month = datetime.datetime.now().month

        # Encode crop
        crop_encoded = _crop_encoder.transform([crop_lower])[0]

        # Prepare feature vector
        features = np.array([[crop_encoded, temp, humidity, rainfall, wind_speed, month]])
        features_scaled = _scaler.transform(features)

        # Predict pest probabilities
        pest_probs = _rf_model.predict_proba(features_scaled)[0]
        severity_pred = _gb_model.predict(features_scaled)[0]
        severity_probs = _gb_model.predict_proba(features_scaled)[0]

        # Get top pest predictions (exclude 'None')
        alerts = []
        pest_indices = np.argsort(pest_probs)[::-1]

        for idx in pest_indices[:5]:
            pest_name = _pest_encoder.classes_[idx]
            probability = float(pest_probs[idx]) * 100

            if pest_name == 'None' or probability < 5:
                continue

            # Get pest info from database
            pest_details = _get_pest_details(crop_lower, pest_name)

            # Determine severity
            severity_name = _severity_encoder.classes_[severity_pred]

            # Calculate risk score (0-100)
            risk_score = min(100, probability * (1 + {'low': 0, 'medium': 0.3, 'high': 0.6, 'critical': 1.0}.get(severity_name, 0)))

            alerts.append({
                'pest_name': pest_name,
                'probability': round(probability, 1),
                'severity': severity_name,
                'risk_score': round(risk_score, 1),
                'symptoms': pest_details.get('symptoms', 'Monitor crop for unusual signs'),
                'prevention': pest_details.get('prevention', 'Follow integrated pest management practices'),
                'treatment': pest_details.get('treatment', 'Consult agricultural extension officer'),
            })

        # Sort by risk score
        alerts.sort(key=lambda x: x['risk_score'], reverse=True)

        # Overall risk level
        max_risk = max([a['risk_score'] for a in alerts], default=0)
        if max_risk >= 70:
            overall_risk = 'HIGH'
        elif max_risk >= 40:
            overall_risk = 'MODERATE'
        elif max_risk >= 15:
            overall_risk = 'LOW'
        else:
            overall_risk = 'MINIMAL'

        return {
            'success': True,
            'alerts': alerts,
            'weather': weather_data,
            'risk_summary': {
                'overall_risk': overall_risk,
                'max_risk_score': round(max_risk, 1),
                'threats_detected': len(alerts),
            }
        }

    except Exception as e:
        logger.error("Pest prediction error: %s", e)
        return {'success': False, 'error': str(e)}


def _get_pest_details(crop, pest_name):
    """Get detailed pest information from database."""
    if _pest_info_db and crop in _pest_info_db:
        for pest in _pest_info_db[crop]:
            if pest['pest_name'] == pest_name:
                return pest
    return {}


def get_supported_crops():
    """Return list of supported crops for pest prediction."""
    if _load_models():
        return [c.title() for c in _crop_encoder.classes_ if c != 'None']
    return ['Rice', 'Wheat', 'Maize', 'Cotton', 'Tomato', 'Potato', 'Sugarcane']
