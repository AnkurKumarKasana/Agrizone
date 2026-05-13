from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from django.conf import settings
import json
import logging

from .services import predict_crop, get_supported_crops

logger = logging.getLogger(__name__)


def home(request):
    return render(request, 'crop_recommendation/index.html')

def about(request):
    return render(request, 'crop_recommendation/about.html')

def contact(request):
    return render(request, 'crop_recommendation/contact.html')

def recommend(request):
    return render(request, 'crop_recommendation/recommend.html')


@csrf_exempt
def predict(request):
    """Handle crop prediction requests using the real ML model."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            N = float(data.get('Nitrogen', 0))
            P = float(data.get('Phosphorus', 0))
            K = float(data.get('Potassium', 0))
            temp = float(data.get('Temperature', 0))
            humidity = float(data.get('Humidity', 0))
            ph = float(data.get('pH', 0))
            rainfall = float(data.get('Rainfall', 0))

            # Validate inputs
            if not all([0 <= N <= 200, 0 <= P <= 200, 0 <= K <= 300,
                        0 <= temp <= 60, 0 <= humidity <= 100,
                        0 <= ph <= 14, 0 <= rainfall <= 500]):
                return JsonResponse({
                    'success': False,
                    'error': 'Input values are out of valid range. Please check your entries.'
                })

            # Get ML prediction (top 3 crops with confidence)
            result = predict_crop(N, P, K, temp, humidity, ph, rainfall)

            if result['success']:
                return JsonResponse({
                    'success': True,
                    'predictions': result['predictions'],
                    'model_info': result['model_info'],
                })
            else:
                return JsonResponse({'success': False, 'error': result['error']})

        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)


@csrf_exempt
def send_contact(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            name = data.get('name', '')
            email = data.get('email', '')
            subject = data.get('subject', '')
            message = data.get('message', '')
            
            # Email content
            email_subject = f"Contact Form: {subject}"
            email_message = f"""
            New contact form submission from Agrizone:
            
            Name: {name}
            Email: {email}
            Subject: {subject}
            
            Message:
            {message}
            
            ---
            This message was sent from the Agrizone contact form.
            """
            
            # Send email
            send_mail(
                email_subject,
                email_message,
                settings.DEFAULT_FROM_EMAIL,
                ['abhishekgujjar2200@gmail.com'],
                fail_silently=False,
            )
            
            return JsonResponse({"success": True, "message": "Email sent successfully"})
            
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)
    
    return JsonResponse({"success": False, "error": "Invalid request method"}, status=405)