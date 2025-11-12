from django.contrib import admin
from .models import ChatMessage

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['user_message', 'created_at']
    list_filter = ['created_at']
    readonly_fields = ['created_at']
    search_fields = ['user_message', 'bot_response']