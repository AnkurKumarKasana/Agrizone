from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.http import HttpResponse

def test_view(request):
    return HttpResponse('Pest Alerts URL is working! Go to /alerts/ for the full page.')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('test-alerts/', test_view),
    path('', include('crop_recommendation.urls')),
    path('weather/', include('weather.urls')),
    path('disease/', include('disease.urls')),
    path('chatbot/', include('chatbot.urls')),
    path('alerts/', include('pest_alerts.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])