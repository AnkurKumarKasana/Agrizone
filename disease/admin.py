from django.contrib import admin
from .models import DiseaseDetection

@admin.register(DiseaseDetection)
class DiseaseDetectionAdmin(admin.ModelAdmin):
    list_display = ['predicted_disease', 'confidence', 'created_at']
    list_filter = ['predicted_disease', 'created_at']
    readonly_fields = ['created_at']