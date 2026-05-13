from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import DiseaseDetection
from .services import predict_disease, get_disease_display_name
import os
import logging

logger = logging.getLogger(__name__)


def disease_home(request):
    return render(request, 'disease/disease.html')


def upload_image(request):
    """Handle plant disease image upload and real ML prediction."""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_file = request.FILES['image']

            # Validate file type
            allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
            if uploaded_file.content_type not in allowed_types:
                return render(request, 'disease/disease.html', {
                    'error': 'Invalid file type. Please upload JPG, PNG, GIF, or WebP images.'
                })

            # Validate file size (max 5MB)
            if uploaded_file.size > 5 * 1024 * 1024:
                return render(request, 'disease/disease.html', {
                    'error': 'File too large. Maximum size is 5MB.'
                })

            # Save the uploaded file
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)

            # Real ML prediction
            disease_class, confidence, treatment_info = predict_disease(file_path)

            # Format display name
            display_name = get_disease_display_name(disease_class)

            # Build treatment text
            treatment_text = treatment_info.get('treatment', 'Consult an agricultural expert.')
            if treatment_info.get('prevention'):
                treatment_text += f"\n\nPrevention: {treatment_info['prevention']}"

            # Save to database
            detection = DiseaseDetection.objects.create(
                image=filename,
                predicted_disease=display_name,
                confidence=confidence,
                treatment_suggestion=treatment_text
            )

            return render(request, 'disease/result.html', {
                'detection': detection,
                'image_url': fs.url(filename),
                'disease_class': disease_class,
                'treatment_info': treatment_info,
                'is_healthy': 'healthy' in disease_class.lower(),
            })

        except Exception as e:
            logger.error("Disease detection error: %s", e)
            return render(request, 'disease/disease.html', {
                'error': f'Error processing image: {str(e)}'
            })

    return redirect('disease:disease_home')