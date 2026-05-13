from django.db import models

class DiseaseDetection(models.Model):
    image = models.ImageField(upload_to='disease_images/')
    predicted_disease = models.CharField(max_length=200, blank=True)
    confidence = models.FloatField(default=0.0)
    treatment_suggestion = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Disease Detection - {self.predicted_disease}"