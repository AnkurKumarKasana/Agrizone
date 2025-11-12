from django.core.management.base import BaseCommand
from chatbot.views import get_ai_response

class Command(BaseCommand):
    help = 'Test the AI assistant functionality'

    def handle(self, *args, **options):
        test_questions = [
            "What's the best fertilizer for tomatoes?",
            "How do I control pests naturally?",
            "When should I plant rice?",
            "How to improve soil health?",
        ]
        
        self.stdout.write(self.style.SUCCESS('🌱 Testing AgriGPT AI Assistant'))
        self.stdout.write('=' * 50)
        
        for i, question in enumerate(test_questions, 1):
            self.stdout.write(f'\n{i}. Question: {question}')
            self.stdout.write('-' * 30)
            
            try:
                response = get_ai_response(question)
                self.stdout.write(f'Response: {response}')
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error: {e}'))
        
        self.stdout.write('\n' + '=' * 50)
        self.stdout.write(self.style.SUCCESS('✅ Testing completed!'))