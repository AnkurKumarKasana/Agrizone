@echo off
echo Starting Agrizone Django Server...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if dependencies are installed
python -c "import django" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run migrations if needed
python manage.py migrate --run-syncdb

REM Start the server
echo.
echo Starting Django development server...
echo Visit: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.
python manage.py runserver

pause