#!/usr/bin/env python
"""
Setup script for Agrizone Django project
Run this script to set up the project quickly
"""

import os
import sys
import subprocess
import secrets

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def generate_secret_key():
    """Generate a Django secret key"""
    return secrets.token_urlsafe(50)

def setup_env_file():
    """Create .env file if it doesn't exist"""
    env_file = '.env'
    if not os.path.exists(env_file):
        secret_key = generate_secret_key()
        env_content = f"""SECRET_KEY={secret_key}
DEBUG=True
OPENWEATHER_API_KEY=your-openweather-api-key
OPENAI_API_KEY=your-openai-api-key
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✓ Created .env file with default settings")
        print("⚠️  Please update API keys in .env file")
    else:
        print("✓ .env file already exists")

def main():
    print("🌱 Agrizone Django Setup Script")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Setup environment file
    setup_env_file()
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run migrations
    if not run_command("python manage.py makemigrations", "Creating migrations"):
        print("❌ Failed to create migrations")
        sys.exit(1)
    
    if not run_command("python manage.py migrate", "Applying migrations"):
        print("❌ Failed to apply migrations")
        sys.exit(1)
    
    # Collect static files
    run_command("python manage.py collectstatic --noinput", "Collecting static files")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update API keys in .env file")
    print("2. Create superuser: python manage.py createsuperuser")
    print("3. Run server: python manage.py runserver")
    print("4. Visit: http://127.0.0.1:8000")

if __name__ == "__main__":
    main()