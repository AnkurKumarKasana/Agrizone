"""
Crop Recommendation Model Training Script
==========================================
Trains RF, GBM, SVM on Kaggle Crop Recommendation dataset format.
22 crops, 7 features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

CROP_PROFILES = {
    'rice':        {'N': (60,100), 'P': (35,65), 'K': (35,55), 'temp': (20,27), 'humidity': (80,95), 'pH': (5.0,7.0), 'rainfall': (200,300)},
    'maize':       {'N': (60,100), 'P': (35,65), 'K': (15,35), 'temp': (18,27), 'humidity': (55,75), 'pH': (5.5,7.5), 'rainfall': (60,110)},
    'jute':        {'N': (60,100), 'P': (35,60), 'K': (35,45), 'temp': (23,27), 'humidity': (78,92), 'pH': (6.0,7.5), 'rainfall': (150,200)},
    'cotton':      {'N': (100,140),'P': (40,70), 'K': (15,25), 'temp': (22,30), 'humidity': (75,85), 'pH': (6.0,8.0), 'rainfall': (60,110)},
    'coconut':     {'N': (15,30),  'P': (10,20), 'K': (25,40), 'temp': (25,30), 'humidity': (90,98), 'pH': (5.5,7.0), 'rainfall': (130,200)},
    'papaya':      {'N': (35,60),  'P': (45,70), 'K': (45,60), 'temp': (25,35), 'humidity': (90,98), 'pH': (6.0,7.0), 'rainfall': (100,180)},
    'orange':      {'N': (15,30),  'P': (10,20), 'K': (5,15),  'temp': (10,35), 'humidity': (90,98), 'pH': (6.0,8.0), 'rainfall': (100,120)},
    'apple':       {'N': (15,30),  'P': (120,145),'K': (195,205),'temp': (21,24),'humidity': (90,95), 'pH': (5.5,6.5), 'rainfall': (100,130)},
    'muskmelon':   {'N': (95,110), 'P': (5,15),  'K': (45,55), 'temp': (27,32), 'humidity': (90,95), 'pH': (6.0,6.8), 'rainfall': (20,30)},
    'watermelon':  {'N': (80,110), 'P': (5,15),  'K': (45,55), 'temp': (24,28), 'humidity': (80,92), 'pH': (6.0,6.8), 'rainfall': (40,60)},
    'grapes':      {'N': (15,30),  'P': (120,145),'K': (195,205),'temp': (8,15), 'humidity': (78,85), 'pH': (5.5,6.5), 'rainfall': (60,80)},
    'mango':       {'N': (15,30),  'P': (15,30), 'K': (25,40), 'temp': (27,35), 'humidity': (45,65), 'pH': (5.5,7.0), 'rainfall': (90,110)},
    'banana':      {'N': (90,110), 'P': (70,85), 'K': (45,55), 'temp': (25,30), 'humidity': (75,85), 'pH': (5.5,7.0), 'rainfall': (100,130)},
    'pomegranate': {'N': (15,30),  'P': (5,15),  'K': (35,45), 'temp': (18,24), 'humidity': (85,95), 'pH': (5.5,7.5), 'rainfall': (100,120)},
    'lentil':      {'N': (15,30),  'P': (55,75), 'K': (15,25), 'temp': (18,28), 'humidity': (18,65), 'pH': (6.0,8.0), 'rainfall': (40,50)},
    'blackgram':   {'N': (30,50),  'P': (55,75), 'K': (15,25), 'temp': (25,35), 'humidity': (60,70), 'pH': (6.0,8.0), 'rainfall': (60,70)},
    'mungbean':    {'N': (15,30),  'P': (40,65), 'K': (15,25), 'temp': (27,32), 'humidity': (80,90), 'pH': (6.0,7.5), 'rainfall': (40,55)},
    'mothbeans':   {'N': (15,30),  'P': (40,65), 'K': (15,25), 'temp': (24,32), 'humidity': (40,65), 'pH': (3.5,9.0), 'rainfall': (30,60)},
    'pigeonpeas':  {'N': (15,30),  'P': (55,75), 'K': (15,25), 'temp': (18,36), 'humidity': (30,70), 'pH': (4.5,8.0), 'rainfall': (130,170)},
    'kidneybeans': {'N': (15,30),  'P': (55,75), 'K': (15,25), 'temp': (15,22), 'humidity': (15,25), 'pH': (5.5,6.5), 'rainfall': (60,80)},
    'chickpea':    {'N': (30,50),  'P': (55,80), 'K': (70,85), 'temp': (15,22), 'humidity': (14,20), 'pH': (6.0,8.0), 'rainfall': (60,90)},
    'coffee':      {'N': (90,120), 'P': (15,30), 'K': (25,40), 'temp': (23,28), 'humidity': (50,70), 'pH': (6.0,7.0), 'rainfall': (140,180)},
}

CROP_SEASONS = {
    'rice': 'Kharif (June-July)', 'maize': 'Kharif (June-July)', 'jute': 'Kharif (March-May)',
    'cotton': 'Kharif (April-May)', 'coconut': 'Year-round', 'papaya': 'Year-round',
    'orange': 'Rabi (July-August)', 'apple': 'Rabi (January-March)',
    'muskmelon': 'Zaid (Feb-Mar)', 'watermelon': 'Zaid (Feb-Mar)',
    'grapes': 'Rabi (Jan-Feb)', 'mango': 'Year-round', 'banana': 'Year-round',
    'pomegranate': 'Year-round', 'lentil': 'Rabi (Oct-Nov)', 'blackgram': 'Kharif (Jun-Jul)',
    'mungbean': 'Kharif/Zaid', 'mothbeans': 'Kharif (Jul-Aug)', 'pigeonpeas': 'Kharif (Jun-Jul)',
    'kidneybeans': 'Rabi (Oct-Nov)', 'chickpea': 'Rabi (Oct-Nov)', 'coffee': 'Year-round',
}

CROP_WATER = {
    'rice': 'High', 'maize': 'Moderate', 'jute': 'High', 'cotton': 'Moderate',
    'coconut': 'High', 'papaya': 'Moderate', 'orange': 'Moderate', 'apple': 'Moderate',
    'muskmelon': 'Low', 'watermelon': 'Moderate', 'grapes': 'Low', 'mango': 'Low-Moderate',
    'banana': 'High', 'pomegranate': 'Low', 'lentil': 'Low', 'blackgram': 'Low',
    'mungbean': 'Low', 'mothbeans': 'Very Low', 'pigeonpeas': 'Low', 'kidneybeans': 'Low',
    'chickpea': 'Low', 'coffee': 'Moderate',
}

CROP_DURATION = {
    'rice': '120-150 days', 'maize': '80-110 days', 'jute': '120-150 days',
    'cotton': '150-180 days', 'coconut': '6-10 years', 'papaya': '10-12 months',
    'orange': '3-5 years', 'apple': '4-8 years', 'muskmelon': '80-120 days',
    'watermelon': '80-110 days', 'grapes': '2-3 years', 'mango': '5-8 years',
    'banana': '10-15 months', 'pomegranate': '2-3 years', 'lentil': '90-120 days',
    'blackgram': '60-90 days', 'mungbean': '60-90 days', 'mothbeans': '75-90 days',
    'pigeonpeas': '120-180 days', 'kidneybeans': '90-120 days', 'chickpea': '90-120 days',
    'coffee': '3-4 years',
}


def generate_crop_dataset(samples_per_crop=150, seed=42):
    np.random.seed(seed)
    data = []
    for crop, profile in CROP_PROFILES.items():
        for _ in range(samples_per_crop):
            row = {
                'N': np.random.uniform(*profile['N']),
                'P': np.random.uniform(*profile['P']),
                'K': np.random.uniform(*profile['K']),
                'temperature': np.random.uniform(*profile['temp']),
                'humidity': np.random.uniform(*profile['humidity']),
                'ph': np.random.uniform(*profile['pH']),
                'rainfall': np.random.uniform(*profile['rainfall']),
                'label': crop,
            }
            for feat in ['N','P','K','temperature','humidity','ph','rainfall']:
                row[feat] += np.random.normal(0, abs(row[feat]) * 0.03)
            data.append(row)
    return pd.DataFrame(data)


def train_and_evaluate():
    print("=" * 60)
    print("AGRIZONE - Crop Recommendation Model Training")
    print("=" * 60)

    df = generate_crop_dataset()
    print(f"\nDataset: {df.shape[0]} samples, {df['label'].nunique()} crops")
    df.to_csv(os.path.join(MODEL_DIR, 'crop_dataset.csv'), index=False)

    X = df[['N','P','K','temperature','humidity','ph','rainfall']].values
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=8, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, probability=True, random_state=42),
    }

    best_name, best_acc, best_model = None, 0, None
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        cv = cross_val_score(model, X_scaled, y_enc, cv=5)
        results[name] = {'accuracy': acc, 'cv_mean': float(cv.mean()), 'cv_std': float(cv.std())}
        print(f"  Accuracy: {acc:.4f} | CV: {cv.mean():.4f} +/- {cv.std():.4f}")
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

    print(f"\nBest: {best_name} ({best_acc:.4f})")
    print(classification_report(y_test, best_model.predict(X_test), target_names=le.classes_))

    # Save artifacts
    with open(os.path.join(MODEL_DIR, 'crop_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, 'crop_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, 'crop_label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    # Save metadata
    meta = {
        'best_model': best_name, 'accuracy': best_acc,
        'features': ['N','P','K','temperature','humidity','ph','rainfall'],
        'crops': list(le.classes_), 'num_crops': len(le.classes_),
        'model_comparison': results,
    }
    with open(os.path.join(MODEL_DIR, 'crop_model_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save crop info
    crop_info = {}
    for crop in le.classes_:
        p = CROP_PROFILES[crop]
        crop_info[crop] = {
            'display_name': crop.title(),
            'ideal_conditions': {
                'N': f"{p['N'][0]}-{p['N'][1]}", 'P': f"{p['P'][0]}-{p['P'][1]}",
                'K': f"{p['K'][0]}-{p['K'][1]}", 'temperature': f"{p['temp'][0]}-{p['temp'][1]}°C",
                'humidity': f"{p['humidity'][0]}-{p['humidity'][1]}%",
                'pH': f"{p['pH'][0]}-{p['pH'][1]}", 'rainfall': f"{p['rainfall'][0]}-{p['rainfall'][1]} mm",
            },
            'season': CROP_SEASONS.get(crop, 'Varies'),
            'water_needs': CROP_WATER.get(crop, 'Moderate'),
            'growth_duration': CROP_DURATION.get(crop, '90-120 days'),
        }
    with open(os.path.join(MODEL_DIR, 'crop_info.json'), 'w') as f:
        json.dump(crop_info, f, indent=2)

    print("\nAll artifacts saved to:", MODEL_DIR)
    return best_model, scaler, le


if __name__ == '__main__':
    train_and_evaluate()
