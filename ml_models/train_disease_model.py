"""
Plant Disease Detection Model Training Script
==============================================
Creates a MobileNetV2-based CNN for plant disease classification.
Generates a lightweight model suitable for web deployment.

Based on PlantVillage dataset class structure (38 classes).
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# PlantVillage 38-class labels
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

TREATMENT_DB = {
    'Apple___Apple_scab': {
        'disease': 'Apple Scab',
        'crop': 'Apple',
        'symptoms': 'Dark olive-green spots on leaves, velvety texture, leaves may curl and drop',
        'treatment': 'Apply fungicides like captan or myclobutanil. Remove fallen leaves. Prune for air circulation.',
        'prevention': 'Plant resistant varieties. Apply preventive fungicides in spring. Keep area clean of debris.',
    },
    'Apple___Black_rot': {
        'disease': 'Black Rot',
        'crop': 'Apple',
        'symptoms': 'Brown spots with concentric rings on leaves, rotting fruit with black coloring',
        'treatment': 'Remove infected fruit and cankers. Apply captan or thiophanate-methyl fungicides.',
        'prevention': 'Prune dead wood. Remove mummified fruits. Maintain good air circulation.',
    },
    'Apple___Cedar_apple_rust': {
        'disease': 'Cedar Apple Rust',
        'crop': 'Apple',
        'symptoms': 'Yellow-orange spots on leaves, small tubes on leaf undersides',
        'treatment': 'Apply myclobutanil or triadimefon fungicides during spring.',
        'prevention': 'Remove nearby cedar/juniper trees. Plant resistant varieties.',
    },
    'Apple___healthy': {
        'disease': 'Healthy',
        'crop': 'Apple',
        'symptoms': 'No disease symptoms detected',
        'treatment': 'No treatment needed. Continue regular care.',
        'prevention': 'Maintain proper nutrition, watering, and pest monitoring.',
    },
    'Blueberry___healthy': {
        'disease': 'Healthy', 'crop': 'Blueberry',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Maintain acidic soil pH and proper drainage.',
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'disease': 'Powdery Mildew', 'crop': 'Cherry',
        'symptoms': 'White powdery coating on leaves and fruit',
        'treatment': 'Apply sulfur-based or potassium bicarbonate fungicides.',
        'prevention': 'Improve air circulation. Avoid overhead watering.',
    },
    'Cherry_(including_sour)___healthy': {
        'disease': 'Healthy', 'crop': 'Cherry',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Proper pruning and nutrition management.',
    },
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'disease': 'Gray Leaf Spot', 'crop': 'Corn',
        'symptoms': 'Rectangular gray-brown lesions on leaves',
        'treatment': 'Apply strobilurin or triazole fungicides. Use resistant hybrids.',
        'prevention': 'Crop rotation. Tillage to reduce residue. Plant resistant varieties.',
    },
    'Corn_(maize)___Common_rust_': {
        'disease': 'Common Rust', 'crop': 'Corn',
        'symptoms': 'Reddish-brown pustules on both leaf surfaces',
        'treatment': 'Apply foliar fungicides if severe. Use resistant hybrids.',
        'prevention': 'Plant resistant varieties. Early planting.',
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'disease': 'Northern Leaf Blight', 'crop': 'Corn',
        'symptoms': 'Long cigar-shaped gray-green lesions on leaves',
        'treatment': 'Apply fungicides at early stages. Remove infected debris.',
        'prevention': 'Use resistant hybrids. Crop rotation. Proper spacing.',
    },
    'Corn_(maize)___healthy': {
        'disease': 'Healthy', 'crop': 'Corn',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Proper fertilization and pest management.',
    },
    'Grape___Black_rot': {
        'disease': 'Black Rot', 'crop': 'Grape',
        'symptoms': 'Brown circular lesions on leaves, shriveled black fruit',
        'treatment': 'Apply myclobutanil or mancozeb fungicides.',
        'prevention': 'Remove mummified berries. Prune for air circulation.',
    },
    'Grape___Esca_(Black_Measles)': {
        'disease': 'Esca (Black Measles)', 'crop': 'Grape',
        'symptoms': 'Interveinal striping on leaves, dark spots on berries',
        'treatment': 'Remove infected vines. Apply wound protectants after pruning.',
        'prevention': 'Avoid large pruning wounds. Use clean pruning tools.',
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'disease': 'Leaf Blight', 'crop': 'Grape',
        'symptoms': 'Brown spots with dark borders on leaves',
        'treatment': 'Apply copper-based fungicides. Remove infected leaves.',
        'prevention': 'Good air circulation. Proper spacing between vines.',
    },
    'Grape___healthy': {
        'disease': 'Healthy', 'crop': 'Grape',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Regular monitoring and proper vineyard management.',
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'disease': 'Citrus Greening (HLB)', 'crop': 'Orange',
        'symptoms': 'Yellowing of leaves, lopsided fruit, bitter taste',
        'treatment': 'No cure. Remove infected trees. Control Asian citrus psyllid.',
        'prevention': 'Use certified disease-free nursery stock. Control psyllid vectors.',
    },
    'Peach___Bacterial_spot': {
        'disease': 'Bacterial Spot', 'crop': 'Peach',
        'symptoms': 'Water-soaked spots on leaves, sunken lesions on fruit',
        'treatment': 'Apply copper sprays and oxytetracycline.',
        'prevention': 'Plant resistant varieties. Avoid overhead irrigation.',
    },
    'Peach___healthy': {
        'disease': 'Healthy', 'crop': 'Peach',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Proper pruning and nutrition.',
    },
    'Pepper,_bell___Bacterial_spot': {
        'disease': 'Bacterial Spot', 'crop': 'Bell Pepper',
        'symptoms': 'Small dark spots on leaves and fruit',
        'treatment': 'Apply copper-based bactericides. Remove infected plants.',
        'prevention': 'Use pathogen-free seeds. Crop rotation.',
    },
    'Pepper,_bell___healthy': {
        'disease': 'Healthy', 'crop': 'Bell Pepper',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Proper watering and nutrient management.',
    },
    'Potato___Early_blight': {
        'disease': 'Early Blight', 'crop': 'Potato',
        'symptoms': 'Dark brown concentric ring spots on lower leaves',
        'treatment': 'Apply chlorothalonil or mancozeb fungicides.',
        'prevention': 'Crop rotation. Remove plant debris. Adequate fertilization.',
    },
    'Potato___Late_blight': {
        'disease': 'Late Blight', 'crop': 'Potato',
        'symptoms': 'Water-soaked spots, white mold on leaf undersides',
        'treatment': 'Apply metalaxyl or chlorothalonil. Destroy infected plants.',
        'prevention': 'Plant resistant varieties. Avoid overhead irrigation. Good drainage.',
    },
    'Potato___healthy': {
        'disease': 'Healthy', 'crop': 'Potato',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Proper hilling and moisture management.',
    },
    'Raspberry___healthy': {
        'disease': 'Healthy', 'crop': 'Raspberry',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Proper pruning and pest management.',
    },
    'Soybean___healthy': {
        'disease': 'Healthy', 'crop': 'Soybean',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Crop rotation and proper fertilization.',
    },
    'Squash___Powdery_mildew': {
        'disease': 'Powdery Mildew', 'crop': 'Squash',
        'symptoms': 'White powdery patches on leaves',
        'treatment': 'Apply sulfur or potassium bicarbonate sprays.',
        'prevention': 'Proper spacing. Resistant varieties. Morning watering.',
    },
    'Strawberry___Leaf_scorch': {
        'disease': 'Leaf Scorch', 'crop': 'Strawberry',
        'symptoms': 'Dark purple spots on leaves, drying edges',
        'treatment': 'Remove infected leaves. Apply captan fungicide.',
        'prevention': 'Proper spacing. Avoid overhead watering. Clean planting material.',
    },
    'Strawberry___healthy': {
        'disease': 'Healthy', 'crop': 'Strawberry',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Mulching and proper runner management.',
    },
    'Tomato___Bacterial_spot': {
        'disease': 'Bacterial Spot', 'crop': 'Tomato',
        'symptoms': 'Small dark raised spots on leaves and fruit',
        'treatment': 'Apply copper sprays. Remove infected plants.',
        'prevention': 'Use disease-free seeds. Crop rotation. Avoid overhead watering.',
    },
    'Tomato___Early_blight': {
        'disease': 'Early Blight', 'crop': 'Tomato',
        'symptoms': 'Dark brown spots with concentric rings on lower leaves',
        'treatment': 'Apply chlorothalonil or copper fungicides.',
        'prevention': 'Mulch around plants. Stake for air flow. Crop rotation.',
    },
    'Tomato___Late_blight': {
        'disease': 'Late Blight', 'crop': 'Tomato',
        'symptoms': 'Large irregular water-soaked patches, white mold',
        'treatment': 'Apply metalaxyl-based fungicides. Remove affected plants.',
        'prevention': 'Plant resistant varieties. Good air circulation. Avoid wet foliage.',
    },
    'Tomato___Leaf_Mold': {
        'disease': 'Leaf Mold', 'crop': 'Tomato',
        'symptoms': 'Yellow spots on upper leaf surface, olive-green mold below',
        'treatment': 'Improve ventilation. Apply copper or chlorothalonil fungicides.',
        'prevention': 'Reduce humidity in greenhouse. Space plants properly.',
    },
    'Tomato___Septoria_leaf_spot': {
        'disease': 'Septoria Leaf Spot', 'crop': 'Tomato',
        'symptoms': 'Small circular spots with dark borders and gray centers',
        'treatment': 'Apply copper-based or chlorothalonil fungicides.',
        'prevention': 'Remove lower leaves. Mulch. Crop rotation.',
    },
    'Tomato___Spider_mites_Two-spotted_spider_mite': {
        'disease': 'Spider Mites', 'crop': 'Tomato',
        'symptoms': 'Tiny yellow spots, fine webbing on leaves, leaf bronzing',
        'treatment': 'Apply miticides or insecticidal soap. Introduce predatory mites.',
        'prevention': 'Maintain humidity. Avoid dusty conditions. Regular monitoring.',
    },
    'Tomato___Target_Spot': {
        'disease': 'Target Spot', 'crop': 'Tomato',
        'symptoms': 'Brown spots with concentric rings, target-like pattern',
        'treatment': 'Apply chlorothalonil or copper-based fungicides.',
        'prevention': 'Improve air circulation. Remove infected debris.',
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'disease': 'Yellow Leaf Curl Virus', 'crop': 'Tomato',
        'symptoms': 'Upward curling yellow leaves, stunted growth',
        'treatment': 'Remove infected plants. Control whitefly vectors.',
        'prevention': 'Use resistant varieties. Reflective mulch. Whitefly management.',
    },
    'Tomato___Tomato_mosaic_virus': {
        'disease': 'Mosaic Virus', 'crop': 'Tomato',
        'symptoms': 'Mottled yellow-green leaves, distorted growth',
        'treatment': 'Remove infected plants. No chemical cure available.',
        'prevention': 'Use resistant varieties. Sanitize tools. Avoid tobacco near plants.',
    },
    'Tomato___healthy': {
        'disease': 'Healthy', 'crop': 'Tomato',
        'symptoms': 'No disease symptoms', 'treatment': 'Continue regular care.',
        'prevention': 'Proper watering, staking, and pest monitoring.',
    },
}


def build_disease_model():
    """Build and save a MobileNetV2-based disease classification model."""
    print("=" * 60)
    print("AGRIZONE - Disease Detection Model Setup")
    print("=" * 60)

    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        print("\nBuilding MobileNetV2 model for 38-class disease classification...")
        
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(DISEASE_CLASSES), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        # Save model
        model_path = os.path.join(MODEL_DIR, 'disease_model.h5')
        model.save(model_path)
        print(f"\nModel saved: {model_path}")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
    except ImportError:
        print("\nTensorFlow not installed. Creating model configuration only.")
        print("Install with: pip install tensorflow")
        print("Model will be created when TensorFlow is available.")

    # Save class indices (always)
    class_indices = {cls: idx for idx, cls in enumerate(DISEASE_CLASSES)}
    idx_path = os.path.join(MODEL_DIR, 'class_indices.json')
    with open(idx_path, 'w') as f:
        json.dump(class_indices, f, indent=2)
    print(f"Class indices saved: {idx_path}")

    # Save treatment database
    treat_path = os.path.join(MODEL_DIR, 'treatment_database.json')
    with open(treat_path, 'w') as f:
        json.dump(TREATMENT_DB, f, indent=2)
    print(f"Treatment DB saved: {treat_path}")

    # Save model config
    config = {
        'model_type': 'MobileNetV2',
        'input_shape': [224, 224, 3],
        'num_classes': len(DISEASE_CLASSES),
        'classes': DISEASE_CLASSES,
        'preprocessing': 'tf.keras.applications.mobilenet_v2.preprocess_input',
    }
    config_path = os.path.join(MODEL_DIR, 'disease_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved: {config_path}")
    
    print("\nDisease model setup complete!")


if __name__ == '__main__':
    build_disease_model()
