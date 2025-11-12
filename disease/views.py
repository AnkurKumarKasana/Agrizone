from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import DiseaseDetection
# import tensorflow as tf  # Commented out for now
import numpy as np
from PIL import Image
import os

# Load the disease detection model (placeholder - you need to add your actual model)
# model_path = os.path.join(settings.BASE_DIR, 'disease', 'model.h5')
# if os.path.exists(model_path):
#     disease_model = tf.keras.models.load_model(model_path)
# else:
#     disease_model = None

# Disease classes (example - replace with your actual classes)
DISEASE_CLASSES = [
    'Healthy',
    'Bacterial Blight',
    'Brown Spot',
    'Leaf Smut',
    'Powdery Mildew',
    'Rust',
    'Septoria Leaf Spot',
    'Yellow Leaf Curl Virus'
]

# Treatment suggestions
TREATMENT_SUGGESTIONS = {
    'Healthy': 'Your plant looks healthy! Continue with regular care and monitoring.',
    'Bacterial Blight': 'Apply copper-based fungicides. Remove infected leaves and improve air circulation.',
    'Brown Spot': 'Use fungicides containing mancozeb or chlorothalonil. Avoid overhead watering.',
    'Leaf Smut': 'Apply systemic fungicides. Remove infected plant parts and destroy them.',
    'Powdery Mildew': 'Use sulfur-based fungicides or neem oil. Improve air circulation around plants.',
    'Rust': 'Apply fungicides containing propiconazole. Remove infected leaves immediately.',
    'Septoria Leaf Spot': 'Use copper-based fungicides. Practice crop rotation and avoid overhead irrigation.',
    'Yellow Leaf Curl Virus': 'Control whitefly vectors. Remove infected plants. Use virus-resistant varieties.'
}

def disease_home(request):
    return render(request, 'disease/disease.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_file = request.FILES['image']
            
            # Save the uploaded file
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            
            # Process the image for prediction
            predicted_disease, confidence = predict_disease(file_path)
            
            # Get treatment suggestion
            treatment = TREATMENT_SUGGESTIONS.get(predicted_disease, 'Consult with an agricultural expert for proper treatment.')
            
            # Save to database
            detection = DiseaseDetection.objects.create(
                image=filename,
                predicted_disease=predicted_disease,
                confidence=confidence,
                treatment_suggestion=treatment
            )
            
            return render(request, 'disease/result.html', {
                'detection': detection,
                'image_url': fs.url(filename)
            })
            
        except Exception as e:
            return render(request, 'disease/disease.html', {
                'error': f'Error processing image: {str(e)}'
            })
    
    return redirect('disease:disease_home')

def predict_disease(image_path):
    """
    Predict disease from uploaded image
    This is a placeholder function - replace with actual model prediction
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Placeholder prediction (replace with actual model)
        # if disease_model:
        #     predictions = disease_model.predict(image_array)
        #     predicted_class = np.argmax(predictions[0])
        #     confidence = float(np.max(predictions[0]))
        #     disease_name = DISEASE_CLASSES[predicted_class]
        # else:
        #     # Dummy prediction for demonstration
        disease_name = np.random.choice(DISEASE_CLASSES)
        confidence = np.random.uniform(0.7, 0.95)
        
        return disease_name, confidence
        
    except Exception as e:
        return 'Unknown', 0.0