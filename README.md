# Agrizone - Smart Farming Platform

A comprehensive Django-based agricultural platform with AI-powered features including crop recommendation, weather forecasting, disease detection, pest alerts, and an intelligent chatbot assistant.

## Features

1. **Crop Recommendation System** - ML-powered crop suggestions based on soil parameters (N, P, K, pH, rainfall, temperature, humidity)
2. **Weather Forecast** - Real-time weather data with agricultural insights using OpenWeatherMap API
3. **Plant Disease Detection** - Upload plant images for AI-based disease diagnosis and treatment recommendations
4. **AI Chatbot Assistant** - Intelligent farming advice using Google Gemini AI with comprehensive agricultural knowledge
5. **Pest Alerts System** - Weather-based pest and disease alerts for specific crops and locations

## Tech Stack

- **Backend**: Django 4.2+, Django REST Framework
- **ML/AI**: scikit-learn, Google Generative AI (Gemini), NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
- **APIs**: OpenWeatherMap, Google Gemini
- **Database**: SQLite (development), PostgreSQL (production)
- **Deployment**: Gunicorn, WhiteNoise

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### 1. Clone or Download Project
```bash
git clone <repository-url>
cd Agrizone
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
SECRET_KEY=your-django-secret-key-here
DEBUG=True
OPENWEATHER_API_KEY=your-openweather-api-key
GEMINI_API_KEY=your-gemini-api-key
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-email-app-password
```

**Get API Keys:**
- **OpenWeatherMap**: https://openweathermap.org/api (Free tier available)
- **Google Gemini**: https://makersuite.google.com/app/apikey (Free tier available)

### 4. Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### 5. Load Initial Data (Optional)
```bash
python manage.py loaddata pest_alerts/fixtures/initial_pests.json
```

### 6. Run the Application
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to access the application.

## Project Structure

```
Agrizone/
├── agrizone/                    # Main Django project settings
│   ├── settings.py             # Configuration
│   ├── urls.py                 # URL routing
│   └── wsgi.py                 # WSGI config
├── crop_recommendation/         # Crop recommendation app
│   ├── views.py                # ML model integration
│   └── urls.py                 # Routes
├── weather/                     # Weather forecast app
│   ├── views.py                # OpenWeatherMap API integration
│   └── urls.py                 # Routes
├── disease/                     # Disease detection app
│   ├── views.py                # Image processing & AI diagnosis
│   ├── models.py               # Disease records
│   └── urls.py                 # Routes
├── chatbot/                     # AI chatbot app
│   ├── views.py                # Gemini AI integration
│   ├── models.py               # Chat history
│   └── urls.py                 # Routes
├── pest_alerts/                 # Pest alert system
│   ├── views.py                # Weather-based pest detection
│   ├── models.py               # Pest database
│   └── urls.py                 # Routes
├── templates/                   # HTML templates
│   ├── base.html               # Base template
│   ├── crop_recommendation/    # Crop pages
│   ├── weather/                # Weather pages
│   ├── disease/                # Disease pages
│   ├── chatbot/                # Chat interface
│   └── pest_alerts/            # Alert pages
├── static/                      # Static files (CSS, JS, images)
│   └── crops/                  # Crop images
├── media/                       # User uploaded files
├── model.pkl                    # Trained crop recommendation model
├── standscaler.pkl              # Feature scaler
├── minmaxscaler.pkl             # MinMax scaler
├── requirements.txt             # Python dependencies
├── manage.py                    # Django management script
└── README.md                    # This file
```

## Application Features

### 1. Crop Recommendation (`/` or `/recommend/`)
- Input soil parameters: Nitrogen (N), Phosphorus (P), Potassium (K), pH
- Environmental data: Temperature, Humidity, Rainfall
- ML model predicts best crop with confidence score
- Displays crop image and cultivation tips

### 2. Weather Forecast (`/weather/`)
- Enter any city worldwide
- Real-time weather data (temperature, humidity, wind, pressure)
- Agricultural insights (field work suitability, irrigation needs, spray conditions)
- Farming recommendations based on current weather

### 3. Disease Detection (`/disease/`)
- Upload plant leaf images
- AI-powered disease identification
- Treatment recommendations
- Prevention tips
- Image history stored in database

### 4. AI Chatbot (`/chatbot/`)
- Powered by Google Gemini AI
- Comprehensive agricultural knowledge base
- Specific advice on:
  - Crop cultivation practices
  - Fertilizer schedules with exact NPK ratios
  - Pest & disease management with pesticide names and dosages
  - Seasonal farming calendar
  - Soil health management
  - Irrigation strategies
- Fallback intelligent response system when API unavailable
- Chat history saved in database

### 5. Pest Alerts (`/alerts/`)
- Enter crop name and location
- Weather-based pest risk assessment
- Displays matching pests/diseases for current conditions
- Symptoms, prevention, and treatment information
- Severity levels (Low, Medium, High, Critical)

## API Configuration

### OpenWeatherMap API
1. Sign up at https://openweathermap.org/
2. Navigate to API Keys section
3. Copy your API key
4. Add to `.env` file: `OPENWEATHER_API_KEY=your_key_here`
5. Free tier: 60 calls/minute, 1,000,000 calls/month

### Google Gemini API
1. Visit https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Create new API key
4. Add to `.env` file: `GEMINI_API_KEY=your_key_here`
5. Free tier: 60 requests/minute

## Machine Learning Models

### Crop Recommendation Model
- **Algorithm**: Random Forest Classifier
- **Features**: N, P, K, temperature, humidity, pH, rainfall
- **Output**: Crop name with confidence score
- **Accuracy**: ~95% on test data
- **Crops Supported**: 22 crops (rice, wheat, maize, cotton, sugarcane, etc.)

### Pre-trained Models
- `model.pkl` - Trained Random Forest model
- `standscaler.pkl` - StandardScaler for feature normalization
- `minmaxscaler.pkl` - MinMaxScaler for feature scaling

## Testing

Test individual components:

```bash
# Test weather API
python test_weather.py

# Test Gemini AI
python test_gemini_simple.py

# Test OpenAI (if configured)
python test_openai.py
```

## Deployment

### Production Settings
1. Set `DEBUG=False` in `.env`
2. Configure `ALLOWED_HOSTS` in `settings.py`
3. Use PostgreSQL database
4. Set up static file serving with WhiteNoise
5. Use Gunicorn as WSGI server

### Deploy to Railway/Render
```bash
# Files included for deployment
- Procfile (Gunicorn configuration)
- railway.json (Railway configuration)
- render.yaml (Render configuration)
- build.sh (Build script)
```

### Environment Variables (Production)
```
SECRET_KEY=<strong-secret-key>
DEBUG=False
OPENWEATHER_API_KEY=<your-key>
GEMINI_API_KEY=<your-key>
DATABASE_URL=<postgres-url>
```

## Troubleshooting

### Weather API shows wrong temperature
- Ensure you're using OpenWeatherMap API key (not WeatherAPI)
- Check API key is valid and active
- Verify `.env` file is in project root
- Restart Django server after updating `.env`

### Chatbot not responding
- Check Gemini API key is valid
- Verify internet connection
- Check API quota limits
- Fallback responses will work without API

### Crop recommendation errors
- Ensure all `.pkl` files are in project root
- Check input values are within valid ranges
- Verify scikit-learn version matches training version

### Database errors
- Run migrations: `python manage.py migrate`
- Delete `db.sqlite3` and recreate if corrupted
- Check file permissions

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the repository
- Email: abhishekgujjar2200@gmail.com, nakulbhar7308@gmail.com

## Acknowledgments

- OpenWeatherMap for weather data API
- Google for Gemini AI API
- scikit-learn for ML framework
- Django community for excellent documentation