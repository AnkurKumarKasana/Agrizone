# Agrizone - System Architecture

## Overview
Agrizone is a comprehensive precision agriculture platform built on a modular, scalable architecture integrating machine learning, artificial intelligence, and real-time data processing to provide intelligent farming solutions.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│              (Web Browser - HTML5, CSS3, JavaScript)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│                  Django Templates + Bootstrap                   │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐     │
│  │  Crop    │ Weather  │ Disease  │ Chatbot  │  Pest    │     │
│  │  Pages   │  Pages   │  Pages   │  Pages   │  Pages   │     │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
│                   Django 4.2+ Framework                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              URL Routing (urls.py)                       │  │
│  │  /recommend/  /weather/  /disease/  /chatbot/  /alerts/ │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────┴────────────────────────────┐    │
│  │                  5 Django Apps                         │    │
│  │  ┌────────────┬──────────┬──────────┬──────────┬─────┐│    │
│  │  │   Crop     │ Weather  │ Disease  │ Chatbot  │Pest ││    │
│  │  │Recommend   │ Forecast │Detection │Assistant │Alert││    │
│  │  └────────────┴──────────┴──────────┴──────────┴─────┘│    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                         │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │   ML Engine  │  API Gateway │Image Processor│Rule Engine  │ │
│  │              │              │               │             │ │
│  │ Random Forest│ REST Clients │  PIL/NumPy   │Pest Matching│ │
│  │  Classifier  │              │               │  Algorithm  │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                            │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │  Django ORM  │  Pickle Load │File Storage  │API Requests │ │
│  │              │              │              │             │ │
│  │   Models     │  .pkl files  │FileSystemMgr │  requests   │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA STORAGE LAYER                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │   Database   │  ML Models   │ Media Files  │External APIs│ │
│  │              │              │              │             │ │
│  │SQLite/Postgres│model.pkl    │  /media/    │OpenWeather  │ │
│  │              │standscaler   │  /static/   │Gemini AI    │ │
│  │              │minmaxscaler  │             │             │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module-Wise Architecture

### **Module 1: Crop Recommendation System**

```
┌─────────────────────────────────────────────────────────────┐
│                  CROP RECOMMENDATION FLOW                   │
└─────────────────────────────────────────────────────────────┘

User Input (Web Form)
    │
    ├─ N (Nitrogen)
    ├─ P (Phosphorus)
    ├─ K (Potassium)
    ├─ pH
    ├─ Temperature
    ├─ Humidity
    └─ Rainfall
    │
    ▼
┌─────────────────────────┐
│  Django View (views.py) │
│  - Validate input       │
│  - Parse JSON data      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Preprocessing Pipeline │
│  1. Create NumPy array  │
│  2. MinMaxScaler        │
│     (0-1 normalization) │
│  3. StandardScaler      │
│     (z-score norm)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  ML Prediction Engine   │
│  - Load model.pkl       │
│  - Random Forest (100   │
│    decision trees)      │
│  - Predict crop class   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Post-Processing        │
│  - Map class to crop    │
│  - Get crop image       │
│  - Calculate confidence │
└───────────┬─────────────┘
            │
            ▼
JSON Response → Frontend Display
{
  "crop": "Rice",
  "image": "crops/rice.jpg",
  "confidence": 0.92
}
```

**Technologies:**
- scikit-learn (RandomForestClassifier, MinMaxScaler, StandardScaler)
- NumPy (array operations)
- Pickle (model serialization)
- Django (web framework)

---

### **Module 2: Weather Intelligence System**

```
┌─────────────────────────────────────────────────────────────┐
│                    WEATHER FORECAST FLOW                    │
└─────────────────────────────────────────────────────────────┘

User Input
    │
    └─ City Name (e.g., "Phagwara")
    │
    ▼
┌─────────────────────────┐
│  Django View            │
│  - Receive city name    │
│  - Validate input       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  API Gateway            │
│  - Build API URL        │
│  - Add API key          │
│  - Set timeout (10s)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  OpenWeatherMap API                 │
│  GET /data/2.5/weather              │
│  ?q={city}&appid={key}&units=metric │
└───────────┬─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│  Response Processing    │
│  - Parse JSON           │
│  - Extract:             │
│    • Temperature        │
│    • Humidity           │
│    • Wind speed         │
│    • Pressure           │
│    • Cloud cover        │
│    • Precipitation      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Agricultural Analysis  │
│  - Field work check     │
│    (15-30°C optimal)    │
│  - Irrigation needs     │
│    (40-70% humidity)    │
│  - Spray conditions     │
│    (≤5 m/s wind)        │
└───────────┬─────────────┘
            │
            ▼
JSON Response → Dashboard Display
{
  "temp": 12.5,
  "humidity": 75,
  "insights": {
    "field_work": "Fair",
    "irrigation": "Optimal",
    "spray": "Good"
  }
}
```

**Technologies:**
- OpenWeatherMap API (REST)
- requests library (HTTP client)
- python-dotenv (environment variables)
- Django (web framework)

---

### **Module 3: Disease Detection System**

```
┌─────────────────────────────────────────────────────────────┐
│                  DISEASE DETECTION FLOW                     │
└─────────────────────────────────────────────────────────────┘

User Upload
    │
    └─ Plant Leaf Image (JPEG/PNG)
    │
    ▼
┌─────────────────────────┐
│  Django View            │
│  - Validate file type   │
│  - Check file size      │
│  - Save to media/       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Image Preprocessing    │
│  1. Load with PIL       │
│  2. Resize to 128×128   │
│  3. Convert to array    │
│  4. Normalize (÷255)    │
│  5. Expand dimensions   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  AI Classification      │
│  [Currently Placeholder]│
│  - CNN Model (future)   │
│  - 8 disease classes    │
│  - Confidence score     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Treatment Lookup       │
│  - Map disease to       │
│    treatment protocol   │
│  - Get fungicide info   │
│  - Prevention tips      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Database Storage       │
│  DiseaseDetection Model │
│  - Image path           │
│  - Disease name         │
│  - Confidence           │
│  - Treatment            │
│  - Timestamp            │
└───────────┬─────────────┘
            │
            ▼
Result Page Display
```

**Technologies:**
- PIL/Pillow (image processing)
- NumPy (array operations)
- Django FileSystemStorage (file handling)
- Django ORM (database)
- TensorFlow/Keras (future CNN integration)

---

### **Module 4: AI Chatbot Assistant**

```
┌─────────────────────────────────────────────────────────────┐
│                    CHATBOT FLOW                             │
└─────────────────────────────────────────────────────────────┘

User Question
    │
    └─ "What fertilizer for tomatoes?"
    │
    ▼
┌─────────────────────────┐
│  Django View            │
│  - Receive message      │
│  - Validate input       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Primary AI: Gemini     │
│  - Configure API        │
│  - Set temperature=0.7  │
│  - Max tokens=500       │
│  - Domain prompt        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  Google Gemini 1.5 Flash API        │
│  POST /v1/models/gemini-1.5-flash   │
│  - Agricultural expert prompt       │
│  - User question                    │
└───────────┬─────────────────────────┘
            │
            ├─ Success → Gemini Response
            │
            └─ Fail ──────────┐
                              │
                              ▼
            ┌─────────────────────────┐
            │  Fallback System        │
            │  - Keyword matching     │
            │  - 50+ scenarios        │
            │  - Crop-specific DB     │
            │  - Seasonal calendar    │
            └───────────┬─────────────┘
                        │
                        ▼
┌─────────────────────────┐
│  Response Processing    │
│  - Format text          │
│  - Add timestamp        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Database Storage       │
│  ChatMessage Model      │
│  - User message         │
│  - Bot response         │
│  - Timestamp            │
└───────────┬─────────────┘
            │
            ▼
JSON Response → Chat Interface
{
  "response": "For tomatoes: NPK 19:19:19...",
  "timestamp": "14:30"
}
```

**Technologies:**
- Google Generative AI SDK (google-generativeai)
- Gemini 1.5 Flash LLM
- Django ORM (chat history)
- Python keyword matching (fallback)

---

### **Module 5: Pest Alert System**

```
┌─────────────────────────────────────────────────────────────┐
│                    PEST ALERT FLOW                          │
└─────────────────────────────────────────────────────────────┘

User Input
    │
    ├─ Crop Name (e.g., "Rice")
    └─ Location (e.g., "Phagwara")
    │
    ▼
┌─────────────────────────┐
│  Django View            │
│  - Validate inputs      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Weather Data Fetch     │
│  - Call OpenWeather API │
│  - Get temp & humidity  │
│  - Fallback if fails    │
└───────────┬─────────────┘
            │
            ▼
Current Weather
  temp = 28°C
  humidity = 75%
    │
    ▼
┌─────────────────────────┐
│  Database Query         │
│  PestAlert.objects      │
│  .filter(crop='rice')   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  Matching Algorithm                 │
│  FOR each pest in database:         │
│    IF temp >= min_temp AND          │
│       temp <= max_temp AND          │
│       humidity >= min_humidity AND  │
│       humidity <= max_humidity      │
│    THEN add to alerts               │
└───────────┬─────────────────────────┘
            │
            ▼
Matching Pests
  ├─ Brown Planthopper (High severity)
  └─ Stem Borer (Medium severity)
    │
    ▼
┌─────────────────────────┐
│  Response Builder       │
│  - Pest name            │
│  - Symptoms             │
│  - Prevention           │
│  - Treatment            │
│  - Severity level       │
└───────────┬─────────────┘
            │
            ▼
JSON Response → Alert Display
{
  "alerts": [
    {
      "pest": "Brown Planthopper",
      "symptoms": "Yellowing, hopper burn",
      "treatment": "Imidacloprid 0.4ml/L",
      "severity": "high"
    }
  ]
}
```

**Technologies:**
- Django ORM (pest database queries)
- OpenWeatherMap API (weather data)
- Python conditional logic (matching algorithm)
- SQLite/PostgreSQL (pest database)

---

## 3. Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA FLOW DIAGRAM                         │
└──────────────────────────────────────────────────────────────┘

[User Browser]
      │
      │ HTTP Request
      ▼
[Django URL Router]
      │
      ├──→ /recommend/    → [Crop Module]    → [ML Model]
      │                                            │
      ├──→ /weather/      → [Weather Module] → [OpenWeather API]
      │                                            │
      ├──→ /disease/      → [Disease Module] → [Image Processing]
      │                                            │
      ├──→ /chatbot/      → [Chatbot Module] → [Gemini API]
      │                                            │
      └──→ /alerts/       → [Pest Module]    → [Weather API + DB]
                                                   │
                                                   ▼
                                            [Database Layer]
                                                   │
                                                   ├─ DiseaseDetection
                                                   ├─ ChatMessage
                                                   └─ PestAlert
                                                   │
                                                   ▼
                                            [Response JSON]
                                                   │
                                                   ▼
                                            [Frontend Display]
```

---

## 4. Database Schema

```
┌─────────────────────────────────────────────────────────────┐
│                    DATABASE STRUCTURE                       │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────┐
│  DiseaseDetection        │
├──────────────────────────┤
│ id (PK)                  │
│ image (ImageField)       │
│ predicted_disease (Char) │
│ confidence (Float)       │
│ treatment_suggestion     │
│ created_at (DateTime)    │
└──────────────────────────┘

┌──────────────────────────┐
│  ChatMessage             │
├──────────────────────────┤
│ id (PK)                  │
│ user_message (Text)      │
│ bot_response (Text)      │
│ created_at (DateTime)    │
└──────────────────────────┘

┌──────────────────────────┐
│  PestAlert               │
├──────────────────────────┤
│ id (PK)                  │
│ crop (Char)              │
│ pest_name (Char)         │
│ disease_name (Char)      │
│ min_temp (Float)         │
│ max_temp (Float)         │
│ min_humidity (Float)     │
│ max_humidity (Float)     │
│ rainfall_condition       │
│ symptoms (Text)          │
│ prevention (Text)        │
│ treatment (Text)         │
│ severity (Char)          │
└──────────────────────────┘
```

---

## 5. Technology Stack Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY LAYERS                        │
└─────────────────────────────────────────────────────────────┘

Frontend Layer:
  ├─ HTML5, CSS3, JavaScript
  ├─ Bootstrap 5
  └─ AJAX (async requests)

Application Layer:
  ├─ Django 4.2+ (Python web framework)
  ├─ Django REST Framework
  └─ Django ORM

Machine Learning:
  ├─ scikit-learn (Random Forest, Scalers)
  ├─ NumPy (numerical operations)
  ├─ Pandas (data manipulation)
  └─ Pickle (model serialization)

AI/NLP:
  ├─ Google Generative AI (Gemini 1.5 Flash)
  └─ google-generativeai SDK

Image Processing:
  ├─ PIL/Pillow
  ├─ NumPy
  └─ TensorFlow/Keras (future)

External APIs:
  ├─ OpenWeatherMap API (weather data)
  └─ Google Gemini API (chatbot)

Database:
  ├─ SQLite (development)
  └─ PostgreSQL (production)

Deployment:
  ├─ Gunicorn (WSGI server)
  ├─ WhiteNoise (static files)
  └─ Railway/Render (cloud platforms)

Security:
  ├─ python-decouple (environment variables)
  ├─ Django CSRF protection
  └─ Input validation
```

---

## 6. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION DEPLOYMENT                      │
└─────────────────────────────────────────────────────────────┘

[Internet]
    │
    ▼
[Load Balancer / CDN]
    │
    ▼
[Gunicorn WSGI Server]
    │
    ├─ Worker 1
    ├─ Worker 2
    ├─ Worker 3
    └─ Worker 4
    │
    ▼
[Django Application]
    │
    ├──→ [Static Files] → WhiteNoise → CDN
    │
    ├──→ [Media Files] → File Storage
    │
    ├──→ [Database] → PostgreSQL
    │
    └──→ [External APIs]
         ├─ OpenWeatherMap
         └─ Google Gemini
```

---

## 7. Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                          │
└─────────────────────────────────────────────────────────────┘

Application Security:
  ├─ CSRF Token Protection
  ├─ XSS Prevention (Django templates)
  ├─ SQL Injection Protection (ORM)
  └─ Input Validation & Sanitization

Data Security:
  ├─ Environment Variables (.env)
  ├─ API Key Encryption
  ├─ Password Hashing (Django auth)
  └─ HTTPS (production)

File Security:
  ├─ File Type Validation
  ├─ File Size Limits
  └─ Secure File Storage

API Security:
  ├─ Rate Limiting
  ├─ Timeout Management
  └─ Error Handling
```

---

## 8. Scalability Considerations

**Horizontal Scaling:**
- Multiple Gunicorn workers
- Load balancing across servers
- Database replication

**Vertical Scaling:**
- Increase server resources
- Optimize database queries
- Cache frequently accessed data

**Future Enhancements:**
- Redis caching layer
- Celery for async tasks
- Microservices architecture
- Container orchestration (Docker/Kubernetes)

---

## 9. System Requirements

**Development:**
- Python 3.8+
- 2GB RAM
- 1GB disk space
- Internet connection

**Production:**
- Python 3.8+
- 4GB+ RAM
- 10GB+ disk space
- PostgreSQL database
- Stable internet connection

---

This architecture ensures modularity, scalability, maintainability, and security while providing a robust foundation for future enhancements and integrations.
