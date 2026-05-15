#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

# Retrain ML models with the installed scikit-learn version to avoid version mismatch
echo "Retraining ML models for scikit-learn compatibility..."
python ml_models/train_crop_model.py || echo "WARNING: Crop model training failed (non-fatal)"
python ml_models/train_pest_model.py || echo "WARNING: Pest model training failed (non-fatal)"

python manage.py collectstatic --noinput
python manage.py migrate
