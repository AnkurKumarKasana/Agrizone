from django.db import models

class PestAlert(models.Model):
    crop = models.CharField(max_length=100)
    pest_name = models.CharField(max_length=200)
    disease_name = models.CharField(max_length=200, blank=True)
    min_temp = models.FloatField(null=True, blank=True)
    max_temp = models.FloatField(null=True, blank=True)
    min_humidity = models.FloatField(null=True, blank=True)
    max_humidity = models.FloatField(null=True, blank=True)
    rainfall_condition = models.CharField(max_length=50, blank=True)
    symptoms = models.TextField()
    prevention = models.TextField()
    treatment = models.TextField()
    severity = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical')
    ])
    
    def __str__(self):
        return f"{self.crop} - {self.pest_name or self.disease_name}"
