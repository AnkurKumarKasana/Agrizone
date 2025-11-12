from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import send_mail
from django.conf import settings
import numpy as np
import pickle
import json
import os

# Load ML models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(BASE_DIR, 'standscaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(BASE_DIR, 'minmaxscaler.pkl'), 'rb'))

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
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            N = int(data.get('Nitrogen', 0))
            P = int(data.get('Phosphorus', 0))
            K = int(data.get('Potassium', 0))
            temp = float(data.get('Temperature', 0))
            humidity = float(data.get('Humidity', 0))
            ph = float(data.get('pH', 0))
            rainfall = float(data.get('Rainfall', 0))

            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)
            prediction = model.predict(final_features)

            crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                crop_image = f"crops/{crop.lower()}.jpg"
                return JsonResponse({"success": True, "crop": crop, "image": crop_image})
            else:
                return JsonResponse({"success": False, "error": "Could not determine the best crop"})
        
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)}, status=500)
    
    return JsonResponse({"success": False, "error": "Invalid request method"}, status=405)

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