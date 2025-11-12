from django.contrib import admin
from .models import PestAlert

@admin.register(PestAlert)
class PestAlertAdmin(admin.ModelAdmin):
    list_display = ['crop', 'pest_name', 'disease_name', 'severity']
    list_filter = ['crop', 'severity']
    search_fields = ['crop', 'pest_name', 'disease_name']
