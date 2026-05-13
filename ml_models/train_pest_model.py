"""
Pest Alert Prediction Model Training Script
=============================================
Trains Random Forest + Gradient Boosting ensemble for pest risk prediction
based on crop type, weather conditions, and environmental factors.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Pest profiles: crop -> list of pests with environmental conditions
PEST_DATABASE = {
    'rice': [
        {'pest': 'Brown Plant Hopper', 'temp': (25,35), 'humidity': (80,100), 'rainfall': 'heavy',
         'severity': 'high', 'symptoms': 'Yellowing and drying of plants, hopper burn',
         'prevention': 'Avoid excessive nitrogen. Use resistant varieties. Maintain field drainage.',
         'treatment': 'Apply imidacloprid or thiamethoxam. Drain water from fields.'},
        {'pest': 'Stem Borer', 'temp': (20,30), 'humidity': (70,90), 'rainfall': 'moderate',
         'severity': 'high', 'symptoms': 'Dead heart in vegetative stage, white ear heads',
         'prevention': 'Early planting. Remove stubbles. Light traps.',
         'treatment': 'Apply carbofuran granules. Release Trichogramma wasps.'},
        {'pest': 'Leaf Folder', 'temp': (25,35), 'humidity': (75,95), 'rainfall': 'moderate',
         'severity': 'medium', 'symptoms': 'Folded leaves with feeding marks, white streaks',
         'prevention': 'Avoid excessive nitrogen. Remove weeds.',
         'treatment': 'Spray chlorpyrifos or quinalphos when >2 folded leaves/hill.'},
        {'pest': 'Blast Disease', 'temp': (20,28), 'humidity': (85,100), 'rainfall': 'heavy',
         'severity': 'critical', 'symptoms': 'Diamond-shaped lesions on leaves, neck rot',
         'prevention': 'Use resistant varieties. Balanced fertilization.',
         'treatment': 'Apply tricyclazole or isoprothiolane fungicides.'},
    ],
    'wheat': [
        {'pest': 'Aphids', 'temp': (15,25), 'humidity': (60,80), 'rainfall': 'light',
         'severity': 'medium', 'symptoms': 'Curling leaves, honeydew on plants, stunted growth',
         'prevention': 'Early sowing. Balanced nitrogen. Natural enemies.',
         'treatment': 'Spray dimethoate or imidacloprid.'},
        {'pest': 'Rust', 'temp': (15,25), 'humidity': (80,100), 'rainfall': 'moderate',
         'severity': 'high', 'symptoms': 'Orange-brown pustules on leaves and stems',
         'prevention': 'Use resistant varieties. Timely sowing.',
         'treatment': 'Apply propiconazole or tebuconazole fungicides.'},
        {'pest': 'Termites', 'temp': (25,35), 'humidity': (40,60), 'rainfall': 'dry',
         'severity': 'high', 'symptoms': 'Wilting plants, roots damaged, hollow stems',
         'prevention': 'Deep plowing. Adequate irrigation. Remove crop residue.',
         'treatment': 'Seed treatment with chlorpyrifos. Apply fipronil to soil.'},
    ],
    'maize': [
        {'pest': 'Fall Armyworm', 'temp': (22,32), 'humidity': (60,85), 'rainfall': 'moderate',
         'severity': 'critical', 'symptoms': 'Holes in leaves, frass in whorl, skeletonized leaves',
         'prevention': 'Early planting. Intercropping. Pheromone traps.',
         'treatment': 'Apply spinetoram or emamectin benzoate. Release natural enemies.'},
        {'pest': 'Stem Borer', 'temp': (25,35), 'humidity': (70,90), 'rainfall': 'moderate',
         'severity': 'high', 'symptoms': 'Dead heart, shot holes in leaves, broken stems',
         'prevention': 'Remove crop residues. Early sowing.',
         'treatment': 'Apply carbofuran granules in leaf whorl.'},
        {'pest': 'Corn Earworm', 'temp': (20,30), 'humidity': (50,75), 'rainfall': 'light',
         'severity': 'medium', 'symptoms': 'Feeding damage on ear tips, frass on silk',
         'prevention': 'Use Bt corn. Timely harvest.',
         'treatment': 'Apply spinosad or Bt insecticide to silks.'},
    ],
    'cotton': [
        {'pest': 'Bollworm', 'temp': (25,35), 'humidity': (60,80), 'rainfall': 'moderate',
         'severity': 'critical', 'symptoms': 'Bored bolls, frass, shed squares and bolls',
         'prevention': 'Use Bt cotton. Trap crops. Pheromone traps.',
         'treatment': 'Apply profenofos or emamectin benzoate sprays.'},
        {'pest': 'Whitefly', 'temp': (25,35), 'humidity': (60,80), 'rainfall': 'light',
         'severity': 'high', 'symptoms': 'Yellowing leaves, sticky honeydew, sooty mold',
         'prevention': 'Neem oil spray. Yellow sticky traps. Remove weeds.',
         'treatment': 'Apply spiromesifen or diafenthiuron.'},
        {'pest': 'Pink Bollworm', 'temp': (22,32), 'humidity': (50,70), 'rainfall': 'light',
         'severity': 'high', 'symptoms': 'Rosetted flowers, damaged bolls, webbing inside bolls',
         'prevention': 'Early harvest. Destroy crop residue. Pheromone traps.',
         'treatment': 'Release Trichogramma. Apply quinalphos.'},
    ],
    'tomato': [
        {'pest': 'Tomato Fruit Borer', 'temp': (20,30), 'humidity': (60,80), 'rainfall': 'moderate',
         'severity': 'high', 'symptoms': 'Holes in fruit, frass, rotting fruit',
         'prevention': 'Pheromone traps. Neem oil spray. Crop rotation.',
         'treatment': 'Apply Bt or spinosad insecticides.'},
        {'pest': 'Leaf Miner', 'temp': (22,30), 'humidity': (50,70), 'rainfall': 'light',
         'severity': 'medium', 'symptoms': 'Serpentine mines on leaves, reduced photosynthesis',
         'prevention': 'Yellow sticky traps. Remove infested leaves.',
         'treatment': 'Apply abamectin or cyromazine.'},
        {'pest': 'Whitefly', 'temp': (25,35), 'humidity': (55,75), 'rainfall': 'light',
         'severity': 'high', 'symptoms': 'Yellowing, leaf curl virus transmission, honeydew',
         'prevention': 'Reflective mulch. Sticky traps. Resistant varieties.',
         'treatment': 'Apply imidacloprid or pyriproxyfen.'},
    ],
    'potato': [
        {'pest': 'Late Blight', 'temp': (15,22), 'humidity': (85,100), 'rainfall': 'heavy',
         'severity': 'critical', 'symptoms': 'Water-soaked lesions, white mold, tuber rot',
         'prevention': 'Resistant varieties. Avoid overhead irrigation.',
         'treatment': 'Apply metalaxyl + mancozeb. Destroy infected plants.'},
        {'pest': 'Aphids', 'temp': (18,25), 'humidity': (50,70), 'rainfall': 'light',
         'severity': 'medium', 'symptoms': 'Curled leaves, stunted growth, virus transmission',
         'prevention': 'Seed certification. Remove volunteer plants.',
         'treatment': 'Apply thiamethoxam or acetamiprid.'},
        {'pest': 'Tuber Moth', 'temp': (20,30), 'humidity': (40,60), 'rainfall': 'dry',
         'severity': 'high', 'symptoms': 'Mining in leaves and tubers, frass',
         'prevention': 'Deep planting. Adequate earthing up. Harvest timely.',
         'treatment': 'Apply quinalphos. Storage treatment with DDVP.'},
    ],
    'sugarcane': [
        {'pest': 'Shoot Borer', 'temp': (25,35), 'humidity': (70,90), 'rainfall': 'moderate',
         'severity': 'high', 'symptoms': 'Dead hearts, bore holes in shoots',
         'prevention': 'Use healthy seed sets. Remove infested shoots.',
         'treatment': 'Apply carbofuran granules. Release Trichogramma.'},
        {'pest': 'Top Borer', 'temp': (22,30), 'humidity': (75,95), 'rainfall': 'heavy',
         'severity': 'high', 'symptoms': 'Bunchy top appearance, dead hearts in grown cane',
         'prevention': 'Light traps. Detrashing. Remove infested canes.',
         'treatment': 'Apply monocrotophos. Release parasitoids.'},
    ],
}

RAINFALL_MAP = {'dry': (0, 30), 'light': (30, 80), 'moderate': (80, 150), 'heavy': (150, 300)}
SEVERITY_MAP = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
SUPPORTED_CROPS = list(PEST_DATABASE.keys())


def generate_pest_dataset(samples_per_pest=100, seed=42):
    """Generate pest alert dataset based on real pest-crop-weather relationships."""
    np.random.seed(seed)
    data = []

    for crop, pests in PEST_DATABASE.items():
        for pest_info in pests:
            rain_range = RAINFALL_MAP[pest_info['rainfall']]
            for _ in range(samples_per_pest):
                temp = np.random.uniform(*pest_info['temp']) + np.random.normal(0, 2)
                hum = np.clip(np.random.uniform(*pest_info['humidity']) + np.random.normal(0, 3), 0, 100)
                rain = np.random.uniform(*rain_range) + np.random.normal(0, 5)
                wind = np.random.uniform(5, 30)
                month = np.random.randint(1, 13)
                data.append({
                    'crop': crop, 'temperature': temp, 'humidity': hum,
                    'rainfall': max(0, rain), 'wind_speed': wind, 'month': month,
                    'pest_name': pest_info['pest'],
                    'severity': pest_info['severity'],
                })

            # Add "safe" samples (conditions where this pest is unlikely)
            for _ in range(samples_per_pest // 3):
                temp = np.random.uniform(5, 40)
                hum = np.random.uniform(10, 100)
                rain = np.random.uniform(0, 300)
                if pest_info['temp'][0] <= temp <= pest_info['temp'][1] and \
                   pest_info['humidity'][0] <= hum <= pest_info['humidity'][1]:
                    continue  # Skip if conditions match pest
                data.append({
                    'crop': crop, 'temperature': temp, 'humidity': hum,
                    'rainfall': rain, 'wind_speed': np.random.uniform(5, 30),
                    'month': np.random.randint(1, 13),
                    'pest_name': 'None', 'severity': 'low',
                })

    return pd.DataFrame(data)


def train_pest_models():
    print("=" * 60)
    print("AGRIZONE - Pest Alert Model Training")
    print("=" * 60)

    df = generate_pest_dataset()
    print(f"\nDataset: {df.shape[0]} samples")
    print(f"Crops: {df['crop'].nunique()}, Pests: {df['pest_name'].nunique()}")
    df.to_csv(os.path.join(MODEL_DIR, 'pest_dataset.csv'), index=False)

    # Encode categoricals
    crop_encoder = LabelEncoder()
    pest_encoder = LabelEncoder()
    severity_encoder = LabelEncoder()

    df['crop_encoded'] = crop_encoder.fit_transform(df['crop'])
    df['pest_encoded'] = pest_encoder.fit_transform(df['pest_name'])
    df['severity_encoded'] = severity_encoder.fit_transform(df['severity'])

    features = ['crop_encoded', 'temperature', 'humidity', 'rainfall', 'wind_speed', 'month']
    X = df[features].values
    y_pest = df['pest_encoded'].values
    y_severity = df['severity_encoded'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_pest_train, y_pest_test, y_sev_train, y_sev_test = \
        train_test_split(X_scaled, y_pest, y_severity, test_size=0.2, random_state=42)

    # Train pest prediction model (Random Forest)
    print("\nTraining Pest Prediction (Random Forest)...")
    rf_pest = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_pest.fit(X_train, y_pest_train)
    pest_acc = accuracy_score(y_pest_test, rf_pest.predict(X_test))
    print(f"  Pest Prediction Accuracy: {pest_acc:.4f}")

    # Train severity model (Gradient Boosting)
    print("\nTraining Severity Prediction (Gradient Boosting)...")
    gb_severity = GradientBoostingClassifier(n_estimators=150, max_depth=8, random_state=42)
    gb_severity.fit(X_train, y_sev_train)
    sev_acc = accuracy_score(y_sev_test, gb_severity.predict(X_test))
    print(f"  Severity Prediction Accuracy: {sev_acc:.4f}")

    # Save models
    with open(os.path.join(MODEL_DIR, 'pest_rf_model.pkl'), 'wb') as f:
        pickle.dump(rf_pest, f)
    with open(os.path.join(MODEL_DIR, 'pest_gb_severity_model.pkl'), 'wb') as f:
        pickle.dump(gb_severity, f)
    with open(os.path.join(MODEL_DIR, 'pest_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, 'pest_crop_encoder.pkl'), 'wb') as f:
        pickle.dump(crop_encoder, f)
    with open(os.path.join(MODEL_DIR, 'pest_name_encoder.pkl'), 'wb') as f:
        pickle.dump(pest_encoder, f)
    with open(os.path.join(MODEL_DIR, 'pest_severity_encoder.pkl'), 'wb') as f:
        pickle.dump(severity_encoder, f)

    # Save pest info database
    pest_info_db = {}
    for crop, pests in PEST_DATABASE.items():
        pest_info_db[crop] = []
        for p in pests:
            pest_info_db[crop].append({
                'pest_name': p['pest'], 'severity': p['severity'],
                'symptoms': p['symptoms'], 'prevention': p['prevention'],
                'treatment': p['treatment'],
                'favorable_temp': list(p['temp']),
                'favorable_humidity': list(p['humidity']),
                'favorable_rainfall': p['rainfall'],
            })

    with open(os.path.join(MODEL_DIR, 'pest_info_database.json'), 'w') as f:
        json.dump(pest_info_db, f, indent=2)

    # Save metadata
    meta = {
        'pest_model_accuracy': pest_acc,
        'severity_model_accuracy': sev_acc,
        'supported_crops': SUPPORTED_CROPS,
        'features': features,
        'num_pests': len(pest_encoder.classes_),
        'pest_classes': list(pest_encoder.classes_),
        'severity_classes': list(severity_encoder.classes_),
    }
    with open(os.path.join(MODEL_DIR, 'pest_model_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\nAll pest model artifacts saved to:", MODEL_DIR)


if __name__ == '__main__':
    train_pest_models()
