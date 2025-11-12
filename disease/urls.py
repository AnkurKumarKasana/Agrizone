from django.urls import path
from . import views

app_name = 'disease'

urlpatterns = [
    path('', views.disease_home, name='disease_home'),
    path('upload/', views.upload_image, name='upload_image'),
]