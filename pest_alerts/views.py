from django.shortcuts import render
from django.http import JsonResponse
from .models import PestAlert
from .services import predict_pest_risk, get_weather_data, get_supported_crops
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def alerts_home(request):
    supported_crops = get_supported_crops()
    return render(request, 'pest_alerts/alerts.html', {
        'supported_crops': supported_crops,
    })


def get_alerts(request):
    """Handle pest alert requests using real ML prediction."""
    crop = request.GET.get('crop', '').strip()
    location = request.GET.get('location', '').strip()

    if not crop or not location:
        return JsonResponse({'success': False, 'error': 'Please enter both crop and location'})

    # Get real weather data
    api_key = os.getenv('OPENWEATHER_API_KEY', '')
    weather_data = get_weather_data(location, api_key)

    # Fallback weather data if API fails
    if not weather_data:
        weather_data = {
            'temp': 28.0,
            'humidity': 75.0,
            'description': 'typical conditions (weather API unavailable)',
            'wind_speed': 10.0,
            'rainfall': 50.0,
        }

    # ML-based pest prediction
    result = predict_pest_risk(crop, weather_data)

    if result['success']:
        return JsonResponse({
            'success': True,
            'alerts': result['alerts'],
            'weather': result['weather'],
            'risk_summary': result['risk_summary'],
        })
    else:
        # Fallback to database alerts if ML fails
        db_alerts = _get_db_alerts(crop, weather_data)
        if db_alerts:
            return JsonResponse({
                'success': True,
                'alerts': db_alerts,
                'weather': weather_data,
                'risk_summary': {
                    'overall_risk': 'MODERATE',
                    'max_risk_score': 50,
                    'threats_detected': len(db_alerts),
                },
            })
        return JsonResponse({
            'success': True,
            'alerts': [],
            'weather': weather_data,
            'risk_summary': {
                'overall_risk': 'MINIMAL',
                'max_risk_score': 0,
                'threats_detected': 0,
            },
            'note': result.get('error', ''),
        })


def _get_db_alerts(crop, weather):
    """Fallback: get alerts from database."""
    temp = weather.get('temp', 25)
    humidity = weather.get('humidity', 70)

    alerts = PestAlert.objects.filter(crop__iexact=crop)
    matching = []

    for alert in alerts:
        match = True
        if alert.min_temp and temp < alert.min_temp:
            match = False
        if alert.max_temp and temp > alert.max_temp:
            match = False
        if alert.min_humidity and humidity < alert.min_humidity:
            match = False
        if alert.max_humidity and humidity > alert.max_humidity:
            match = False

        if match:
            matching.append({
                'pest_name': alert.pest_name or alert.disease_name,
                'probability': 65.0,
                'severity': alert.severity,
                'risk_score': 50.0,
                'symptoms': alert.symptoms,
                'prevention': alert.prevention,
                'treatment': alert.treatment,
            })

    return matching
