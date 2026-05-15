"""
Crop Recommendation Service
============================
Handles model loading, prediction, and result formatting.
Uses cached model loading for efficient inference.
"""

import numpy as np
import pickle
import json
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models', 'saved_models')

# Cached model instances
_model = None
_scaler = None
_label_encoder = None
_crop_info = None


def _load_models():
    """Load and cache ML models. Called once on first prediction."""
    global _model, _scaler, _label_encoder, _crop_info

    if _model is not None:
        return True

    model_path = os.path.join(MODEL_DIR, 'crop_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'crop_scaler.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'crop_label_encoder.pkl')
    info_path = os.path.join(MODEL_DIR, 'crop_info.json')

    # Attempt 1: Load existing models
    try:
        if os.path.exists(model_path):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(model_path, 'rb') as f:
                    _model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    _scaler = pickle.load(f)
                with open(encoder_path, 'rb') as f:
                    _label_encoder = pickle.load(f)

            # Quick sanity check: can the model predict?
            test_input = np.array([[50, 50, 50, 25, 70, 6.5, 100]])
            _scaler.transform(test_input)
            _model.predict(test_input if not hasattr(_scaler, 'transform') else _scaler.transform(test_input))

            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    _crop_info = json.load(f)
            else:
                _crop_info = {}

            logger.info("Crop recommendation models loaded successfully.")
            return True

    except Exception as e:
        logger.warning("Existing crop model incompatible (%s). Retraining...", e)
        _model = None
        _scaler = None
        _label_encoder = None

    # Attempt 2: Retrain models with current sklearn version
    try:
        logger.info("Retraining crop model for sklearn compatibility...")
        import sys
        sys.path.insert(0, os.path.join(BASE_DIR, 'ml_models'))
        from train_crop_model import train_and_evaluate
        _model, _scaler, _label_encoder = train_and_evaluate()

        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                _crop_info = json.load(f)
        else:
            _crop_info = {}

        logger.info("Crop model retrained and loaded successfully.")
        return True

    except Exception as e:
        logger.error("Failed to retrain crop models: %s", e)
        _model = None
        return False


def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """
    Predict top 3 recommended crops with confidence scores.
    
    Returns:
        dict with keys: success, predictions (list of top 3), or error message
    """
    if not _load_models():
        return {'success': False, 'error': 'ML models not loaded. Please run model training first.'}

    try:
        features = np.array([[
            float(nitrogen), float(phosphorus), float(potassium),
            float(temperature), float(humidity), float(ph), float(rainfall)
        ]])

        # Scale features
        features_scaled = _scaler.transform(features)

        # Get probability predictions
        if hasattr(_model, 'predict_proba'):
            probabilities = _model.predict_proba(features_scaled)[0]
        else:
            # Fallback for models without predict_proba
            prediction = _model.predict(features_scaled)[0]
            probabilities = np.zeros(len(_label_encoder.classes_))
            probabilities[prediction] = 1.0

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        predictions = []

        for rank, idx in enumerate(top_indices, 1):
            crop_name = _label_encoder.classes_[idx]
            confidence = float(probabilities[idx]) * 100
            crop_details = _crop_info.get(crop_name, {})

            predictions.append({
                'rank': rank,
                'crop': crop_name.title(),
                'crop_key': crop_name,
                'confidence': round(confidence, 1),
                'image': f"crops/{crop_name.lower()}.jpg",
                'season': crop_details.get('season', 'Varies by region'),
                'water_needs': crop_details.get('water_needs', 'Moderate'),
                'growth_duration': crop_details.get('growth_duration', '90-120 days'),
                'ideal_conditions': crop_details.get('ideal_conditions', {}),
            })

        return {
            'success': True,
            'predictions': predictions,
            'model_info': {
                'total_crops': len(_label_encoder.classes_),
                'model_type': type(_model).__name__,
            }
        }

    except Exception as e:
        logger.error("Crop prediction error: %s", e)
        return {'success': False, 'error': str(e)}


def get_supported_crops():
    """Return list of all supported crops."""
    if _load_models():
        return [c.title() for c in _label_encoder.classes_]
    return []
