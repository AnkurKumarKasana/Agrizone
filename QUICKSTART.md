# 🚀 Quick Start Guide

## Option 1: Automated Setup (Recommended)

### Windows Users:
1. Double-click `run_server.bat`
2. The script will automatically install dependencies and start the server

### All Users:
1. Run the setup script:
   ```bash
   python setup.py
   ```
2. Update API keys in `.env` file
3. Create admin user:
   ```bash
   python manage.py createsuperuser
   ```
4. Start server:
   ```bash
   python manage.py runserver
   ```

## Option 2: Manual Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - OpenWeatherMap: https://openweathermap.org/api
     - OpenAI: https://platform.openai.com/api-keys

3. **Setup Database:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   python manage.py createsuperuser
   ```

4. **Run Server:**
   ```bash
   python manage.py runserver
   ```

## 🌐 Access the Application

- **Main Site:** http://127.0.0.1:8000
- **Admin Panel:** http://127.0.0.1:8000/admin
- **Crop Recommendation:** http://127.0.0.1:8000/recommend/
- **Weather Forecast:** http://127.0.0.1:8000/weather/
- **Disease Detection:** http://127.0.0.1:8000/disease/
- **AI Chatbot:** http://127.0.0.1:8000/chatbot/

## 🔑 Required API Keys

### OpenWeatherMap (Free)
1. Sign up at https://openweathermap.org/
2. Go to API keys section
3. Copy your API key to `.env` file

### OpenAI (Paid)
1. Create account at https://platform.openai.com/
2. Add payment method
3. Generate API key
4. Copy to `.env` file

**Note:** The chatbot will work with fallback responses if OpenAI API key is not provided.

## 🎯 Features Overview

- ✅ **Crop Recommendation** - Works immediately with included ML models
- ✅ **Weather Forecast** - Requires OpenWeatherMap API key
- ✅ **Disease Detection** - Works with placeholder model (add your own model.h5)
- ✅ **AI Chatbot** - Works with fallback responses (OpenAI key optional)

## 🔧 Troubleshooting

**Port already in use:**
```bash
python manage.py runserver 8001
```

**Missing dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

**Database issues:**
```bash
python manage.py migrate --run-syncdb
```

## 📱 Mobile Responsive

The application is fully responsive and works on all devices including smartphones and tablets.

## 🎨 Dark/Light Theme

Toggle between dark and light themes using the button in the navigation bar.