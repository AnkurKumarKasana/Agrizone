# Agrizone Django - Smart Farming Platform

A comprehensive Django-based agricultural platform with AI-powered features including crop recommendation, weather forecasting, disease detection, and an intelligent chatbot assistant.

## Features

1. **Crop Recommendation System** - AI-powered crop suggestions based on soil and environmental parameters
2. **Weather Forecast Integration** - Real-time weather data using OpenWeatherMap API
3. **Plant Disease Detection** - Upload plant images for AI-based disease diagnosis
4. **AI Chatbot Assistant** - Intelligent farming advice using OpenAI GPT

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root and add your API keys:
```
SECRET_KEY=your-django-secret-key-here
DEBUG=True
OPENWEATHER_API_KEY=your-openweather-api-key
OPENAI_API_KEY=your-openai-api-key
```

**Get API Keys:**
- OpenWeatherMap API: https://openweathermap.org/api
- OpenAI API: https://platform.openai.com/api-keys

### 3. Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### 4. Run the Application
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to access the application.

## Project Structure

```
agrizone_django/
├── agrizone/                 # Main Django project
├── crop_recommendation/      # Crop recommendation app
├── weather/                  # Weather forecast app
├── disease/                  # Disease detection app
├── chatbot/                  # AI chatbot app
├── templates/               # HTML templates
├── static/                  # Static files
├── media/                   # Uploaded files
├── model.pkl               # ML model for crop recommendation
├── standscaler.pkl         # Standard scaler
├── minmaxscaler.pkl        # MinMax scaler
└── requirements.txt        # Python dependencies
```

## Usage

### Crop Recommendation
1. Navigate to `/recommend/`
2. Enter soil parameters (N, P, K, pH, etc.)
3. Get AI-powered crop suggestions

### Weather Forecast
1. Go to `/weather/`
2. Enter city name
3. View real-time weather data

### Disease Detection
1. Visit `/disease/`
2. Upload plant leaf image
3. Get disease diagnosis and treatment advice

### AI Assistant
1. Access `/chatbot/`
2. Ask farming-related questions
3. Get expert AI responses

## API Keys Setup

### OpenWeatherMap API
1. Sign up at https://openweathermap.org/
2. Get your free API key
3. Add to `.env` file

### OpenAI API
1. Create account at https://platform.openai.com/
2. Generate API key
3. Add to `.env` file

## Deployment Notes

- Set `DEBUG=False` in production
- Configure proper database (PostgreSQL recommended)
- Set up static file serving
- Configure ALLOWED_HOSTS
- Use environment variables for sensitive data

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please create an issue in the repository or contact the development team.