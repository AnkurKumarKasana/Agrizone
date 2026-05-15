"""
Plant Disease Detection Service
=================================
Uses Gemini Vision AI as PRIMARY engine for accurate disease diagnosis.
Local PyTorch model serves as offline fallback only.
"""

import numpy as np
import json
import os
import logging
from PIL import Image

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models', 'saved_models')

# Cached instances
_model = None
_class_indices = None
_treatment_db = None
_torch_available = False
_device = None
_idx_to_class = None


# ── Full 38-class PlantVillage mapping ──────────────────────────
DISEASE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
    # Extended classes for crops not in PlantVillage
    "Wheat___Brown_rust", "Wheat___Yellow_rust", "Wheat___Powdery_mildew", "Wheat___healthy",
    "Rice___Bacterial_leaf_blight", "Rice___Brown_spot", "Rice___Leaf_blast", "Rice___healthy",
    "Cotton___Bacterial_blight", "Cotton___Leaf_curl_virus", "Cotton___healthy",
    "Sugarcane___Red_rot", "Sugarcane___healthy",
    "Mango___Anthracnose", "Mango___Powdery_mildew", "Mango___healthy",
    "Banana___Panama_disease", "Banana___Black_sigatoka", "Banana___healthy",
]

# ── Treatment database ──────────────────────────────────────────
TREATMENT_DATABASE = {
    "Apple___Apple_scab": {"disease": "Apple Scab", "crop": "Apple", "symptoms": "Dark olive-green spots on leaves, scabby lesions on fruits", "treatment": "Apply fungicides like Captan or Myclobutanil. Remove fallen leaves.", "prevention": "Plant resistant varieties. Ensure good air circulation."},
    "Apple___Black_rot": {"disease": "Black Rot", "crop": "Apple", "symptoms": "Brown circular lesions with concentric rings on leaves and fruit", "treatment": "Prune infected branches. Apply Captan fungicide.", "prevention": "Remove mummified fruits. Maintain tree hygiene."},
    "Apple___Cedar_apple_rust": {"disease": "Cedar Apple Rust", "crop": "Apple", "symptoms": "Yellow-orange spots on leaves, sometimes with tube-like structures", "treatment": "Apply Myclobutanil or Mancozeb fungicide.", "prevention": "Remove nearby cedar/juniper trees. Plant resistant varieties."},
    "Apple___healthy": {"disease": "None", "crop": "Apple", "symptoms": "No disease symptoms", "treatment": "Continue regular care.", "prevention": "Maintain proper nutrition and watering."},
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {"disease": "Gray Leaf Spot", "crop": "Corn/Maize", "symptoms": "Rectangular gray-tan lesions running parallel to leaf veins", "treatment": "Apply strobilurin or triazole fungicides.", "prevention": "Rotate crops. Use resistant hybrids. Reduce residue."},
    "Corn_(maize)___Common_rust_": {"disease": "Common Rust", "crop": "Corn/Maize", "symptoms": "Small reddish-brown pustules on both leaf surfaces", "treatment": "Apply Mancozeb or Propiconazole fungicide.", "prevention": "Plant resistant hybrids. Early planting."},
    "Corn_(maize)___Northern_Leaf_Blight": {"disease": "Northern Leaf Blight", "crop": "Corn/Maize", "symptoms": "Long cigar-shaped gray-green lesions on leaves", "treatment": "Apply Azoxystrobin or Propiconazole.", "prevention": "Use resistant varieties. Practice crop rotation."},
    "Corn_(maize)___healthy": {"disease": "None", "crop": "Corn/Maize", "symptoms": "No disease symptoms", "treatment": "Continue regular care.", "prevention": "Maintain proper nutrition."},
    "Potato___Early_blight": {"disease": "Early Blight", "crop": "Potato", "symptoms": "Dark brown concentric rings (target spots) on older leaves", "treatment": "Apply Chlorothalonil or Mancozeb fungicide.", "prevention": "Crop rotation. Remove plant debris. Adequate spacing."},
    "Potato___Late_blight": {"disease": "Late Blight", "crop": "Potato", "symptoms": "Water-soaked lesions turning dark brown/black, white mold on underside", "treatment": "Apply Metalaxyl or Cymoxanil immediately.", "prevention": "Use certified seed. Destroy volunteer plants. Hill properly."},
    "Potato___healthy": {"disease": "None", "crop": "Potato", "symptoms": "No disease symptoms", "treatment": "Continue regular care.", "prevention": "Proper hilling and moisture management."},
    "Tomato___Early_blight": {"disease": "Early Blight", "crop": "Tomato", "symptoms": "Dark concentric rings on lower leaves, spreading upward", "treatment": "Apply Chlorothalonil or Copper-based fungicides.", "prevention": "Mulch around plants. Stake for air flow. Rotate crops."},
    "Tomato___Late_blight": {"disease": "Late Blight", "crop": "Tomato", "symptoms": "Large dark water-soaked blotches on leaves and stems", "treatment": "Apply Mancozeb or Metalaxyl immediately.", "prevention": "Avoid overhead irrigation. Use resistant varieties."},
    "Tomato___healthy": {"disease": "None", "crop": "Tomato", "symptoms": "No disease symptoms", "treatment": "Continue regular care.", "prevention": "Proper spacing and nutrition."},
    "Wheat___Brown_rust": {"disease": "Brown/Leaf Rust", "crop": "Wheat", "symptoms": "Small circular orange-brown pustules scattered on leaf surface", "treatment": "Apply Propiconazole or Tebuconazole fungicide.", "prevention": "Use rust-resistant varieties. Timely sowing."},
    "Wheat___Yellow_rust": {"disease": "Yellow/Stripe Rust", "crop": "Wheat", "symptoms": "Yellow-orange pustules arranged in stripes along leaf veins", "treatment": "Apply Propiconazole or Triadimefon fungicide urgently.", "prevention": "Plant resistant varieties. Avoid late sowing."},
    "Wheat___Powdery_mildew": {"disease": "Powdery Mildew", "crop": "Wheat", "symptoms": "White powdery patches on leaves and stems", "treatment": "Apply sulfur-based or Triadimefon fungicide.", "prevention": "Avoid excess nitrogen. Ensure air circulation."},
    "Wheat___healthy": {"disease": "None", "crop": "Wheat", "symptoms": "No disease symptoms", "treatment": "Continue regular care.", "prevention": "Balanced nutrition and proper irrigation."},
    "Rice___Bacterial_leaf_blight": {"disease": "Bacterial Leaf Blight", "crop": "Rice", "symptoms": "Water-soaked to yellowish lesions on leaf margins, wilting", "treatment": "Drain fields. Apply Copper hydroxide.", "prevention": "Use resistant varieties. Balanced nitrogen."},
    "Rice___Brown_spot": {"disease": "Brown Spot", "crop": "Rice", "symptoms": "Oval brown spots with gray center on leaves", "treatment": "Apply Mancozeb or Propiconazole.", "prevention": "Use certified seed. Balanced fertilization."},
    "Rice___Leaf_blast": {"disease": "Leaf Blast", "crop": "Rice", "symptoms": "Diamond-shaped lesions with gray center on leaves", "treatment": "Apply Tricyclazole or Isoprothiolane fungicide.", "prevention": "Avoid excess nitrogen. Use resistant varieties."},
    "Rice___healthy": {"disease": "None", "crop": "Rice", "symptoms": "No disease symptoms", "treatment": "Continue regular care.", "prevention": "Proper water management."},
}


def _load_metadata():
    """Load class indices and treatment data."""
    global _class_indices, _treatment_db, _idx_to_class

    idx_path = os.path.join(MODEL_DIR, 'class_indices.json')
    if os.path.exists(idx_path):
        with open(idx_path, 'r') as f:
            _class_indices = json.load(f)
        _idx_to_class = {v: k for k, v in _class_indices.items()}

    treat_path = os.path.join(MODEL_DIR, 'treatment_database.json')
    if os.path.exists(treat_path):
        with open(treat_path, 'r') as f:
            _treatment_db = json.load(f)


def _load_pytorch_model():
    """Try to load local PyTorch model (optional fallback)."""
    global _model, _torch_available, _device, _class_indices, _idx_to_class

    model_path = os.path.join(MODEL_DIR, 'disease_model_pytorch.pth')
    if not os.path.exists(model_path):
        return False

    try:
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models

        _device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
        num_classes = checkpoint.get('num_classes', 38)

        model = tv_models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(_device)
        model.eval()
        _model = model
        _torch_available = True

        if 'class_to_idx' in checkpoint:
            _class_indices = checkpoint['class_to_idx']
            _idx_to_class = {v: k for k, v in _class_indices.items()}

        logger.info("PyTorch disease model loaded (fallback).")
        return True
    except Exception as e:
        logger.warning("PyTorch model not available: %s", e)
        return False


_metadata_loaded = False

def _ensure_loaded():
    global _metadata_loaded
    if not _metadata_loaded:
        _load_metadata()
        _load_pytorch_model()
        _metadata_loaded = True


# ════════════════════════════════════════════════════════════════
#  PRIMARY ENGINE: Gemini Vision AI
# ════════════════════════════════════════════════════════════════

def _predict_with_gemini(image_path):
    """
    Use Gemini Vision AI for accurate disease diagnosis.
    Works for ALL crops — not limited to PlantVillage classes.
    """
    try:
        from django.conf import settings as django_settings
        import google.genai as genai

        api_key = getattr(django_settings, 'GEMINI_API_KEY', '')
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — cannot use primary engine.")
            return None, 0.0

        img = Image.open(image_path).convert('RGB')
        client = genai.Client(api_key=api_key)

        class_list = "\n".join(DISEASE_CLASSES)

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                img,
                "You are an expert agricultural plant pathologist.\n"
                "Analyze this leaf/plant image and diagnose the disease.\n\n"
                "RULES:\n"
                "1. First identify the CROP (e.g. Wheat, Corn, Rice, Tomato, Potato, Apple, etc.)\n"
                "2. Then identify the specific DISEASE or say 'healthy' if no disease.\n"
                "3. Output ONLY the label in this exact format: CropName - Disease Name\n"
                "   Example 1: Wheat - Yellow Rust\n"
                "   Example 2: Tomato - Early Blight\n"
                "   Example 3: Corn - Healthy\n"
                "   Example 4: Unknown - Unknown Disease\n"
                "4. Do NOT output anything else. No markdown, no quotes."
            ]
        )

        diagnosis = response.text.strip().replace("'", "").replace('"', '').replace('`', '')
        if "\n" in diagnosis:
            diagnosis = diagnosis.split("\n")[0].strip()

        logger.info("Gemini unconstrained diagnosis: %s", diagnosis)

        # Since Gemini is now unconstrained, we trust its diagnosis heavily
        return diagnosis, 98.0

    except Exception as e:
        logger.error("Gemini prediction failed: %s", e)
        return None, 0.0


def _is_crop_image(image_path):
    """
    Check if image is a plant/crop using Gemini, fallback to color heuristics.
    """
    try:
        from django.conf import settings as django_settings
        import google.genai as genai

        img = Image.open(image_path).convert('RGB')

        api_key = getattr(django_settings, 'GEMINI_API_KEY', '')
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        img,
                        "Is this image a photo of a plant, crop, leaf, or agricultural produce? "
                        "If it shows a human face, pen, paper, furniture, electronics, or any non-plant object, say NO. "
                        "Answer ONLY 'YES' or 'NO'."
                    ]
                )
                text = response.text.strip().upper()
                if "NO" in text:
                    return False
                if "YES" in text:
                    return True
            except Exception as e:
                logger.warning("Gemini crop check failed: %s", e)

        # Fallback: ExG/ExR agricultural indices
        img = img.resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        max_c = np.max(arr, axis=-1)
        min_c = np.min(arr, axis=-1)
        diff = max_c - min_c
        exg = 2 * g - r - b
        exr = 1.4 * r - g
        is_green = (exg > 0.1) & (diff > 0.1)
        is_brown = (exr > 0.15) & (r > g) & (g > b) & (diff > 0.25)
        leaf_pct = np.mean(is_green | is_brown)
        return leaf_pct >= 0.04

    except Exception as e:
        logger.error("Crop check error: %s", e)
        return False


# ════════════════════════════════════════════════════════════════
#  MAIN PREDICTION FUNCTION
# ════════════════════════════════════════════════════════════════

def predict_disease(image_path):
    """
    Predict plant disease from leaf image.
    Priority: Gemini Vision AI → Local PyTorch model → Color fallback.
    """
    _ensure_loaded()

    try:
        if not os.path.exists(image_path):
            return 'Unknown', 0.0, _get_default_treatment()

        # Gate: reject non-crop images
        if not _is_crop_image(image_path):
            return 'Not a Crop Image', 0.0, {
                'disease': 'None', 'crop': 'None',
                'symptoms': 'The uploaded image does not appear to be a plant leaf.',
                'treatment': 'Please upload a clear image of a plant leaf for diagnosis.',
                'prevention': ''
            }

        # ── Tier 1: Gemini Vision AI (primary, most accurate) ──
        disease_class, confidence = _predict_with_gemini(image_path)
        if disease_class and disease_class != 'Unknown':
            treatment = _get_treatment_info(disease_class)
            return disease_class, round(confidence, 1), treatment

        # ── Tier 2: Local PyTorch model (offline fallback) ──
        if _torch_available and _model is not None:
            try:
                import torch
                img_array = preprocess_image(image_path)
                if img_array is not None:
                    with torch.no_grad():
                        tensor = torch.tensor(img_array, dtype=torch.float32).to(_device)
                        outputs = _model(tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf_val, pred_idx = torch.max(probs, 1)
                        pred_idx = pred_idx.item()
                        confidence = conf_val.item() * 100
                        disease_class = _idx_to_class.get(pred_idx, 'Unknown')

                    if confidence >= 60.0:
                        treatment = _get_treatment_info(disease_class)
                        return disease_class, round(confidence, 1), treatment
            except Exception as e:
                logger.warning("Local model inference failed: %s", e)

        # ── Tier 3: Color-based fallback ──
        disease_class, confidence = _analyze_image_fallback(image_path)
        treatment = _get_treatment_info(disease_class)
        return disease_class, round(confidence, 1), treatment

    except Exception as e:
        logger.error("Disease prediction error: %s", e)
        return 'Unknown', 0.0, _get_default_treatment()


def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for ResNet18 PyTorch input."""
    try:
        img = Image.open(image_path).convert('RGB').resize(target_size, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        logger.error("Preprocessing error: %s", e)
        return None


def _analyze_image_fallback(image_path):
    """Last-resort color analysis when both Gemini and local model fail."""
    try:
        img = Image.open(image_path).convert('RGB').resize((128, 128))
        arr = np.array(img, dtype=np.float32) / 255.0
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        green_ratio = np.mean(g) / (np.mean(r) + np.mean(g) + np.mean(b) + 1e-6)
        brown_ratio = np.mean(r - g)

        if green_ratio > 0.38 and brown_ratio < 0.05:
            return 'Unknown___likely_healthy', 40.0
        if brown_ratio > 0.1:
            return 'Unknown___possible_blight_or_spot', 35.0
        return 'Unknown___needs_expert_review', 30.0
    except Exception:
        return 'Unknown', 0.0


def _get_treatment_info(disease_class):
    """Get treatment info — check built-in DB, then file-based DB."""
    if disease_class in TREATMENT_DATABASE:
        return TREATMENT_DATABASE[disease_class]

    if _treatment_db and disease_class in _treatment_db:
        return _treatment_db[disease_class]

    # Parse crop and disease from class name
    if ' - ' in disease_class:
        parts = disease_class.split(' - ')
        crop = parts[0].strip()
        disease = parts[1].strip()
    else:
        parts = disease_class.split('___')
        crop = parts[0].replace('_', ' ') if parts else 'Unknown'
        disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'

    return {
        'disease': disease,
        'crop': crop,
        'symptoms': f'Detected {disease} on {crop}.',
        'treatment': 'Consult a local agricultural expert for targeted treatment.',
        'prevention': 'Practice crop rotation, maintain proper nutrition, and monitor regularly.',
    }


def _get_default_treatment():
    return {
        'disease': 'Unknown', 'crop': 'Unknown',
        'symptoms': 'Unable to determine specific symptoms',
        'treatment': 'Consult with a local agricultural expert.',
        'prevention': 'Practice crop rotation and monitor regularly.',
    }


def get_disease_display_name(disease_class):
    """Convert class name to readable name."""
    if not disease_class or disease_class == 'Unknown':
        return 'Unknown Disease'
    # If it already has a hyphen and no underscores, it's from Gemini's clean output
    if '-' in disease_class and '___' not in disease_class:
        return disease_class
    return disease_class.replace('___', ' - ').replace('_', ' ')


def get_supported_diseases():
    """Return all supported disease classes."""
    return DISEASE_CLASSES
