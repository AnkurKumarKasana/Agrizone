from django.urls import path
from . import views

app_name = 'pest_alerts'

urlpatterns = [
    path('', views.alerts_home, name='home'),
    path('get-alerts/', views.get_alerts, name='get_alerts'),
]
