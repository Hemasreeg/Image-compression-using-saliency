"""Simple wrapper to run the web application"""
import os
import sys

# Suppress all debug output
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Run the app
from app import app, init_db

if __name__ == '__main__':
    print("=" * 60)
    print("  Starting AI Portrait Mode Web Application")
    print("=" * 60)
    init_db()
    print("\n✓ Database initialized")
    print("✓ Starting Flask server...")
    
    # Get port from environment variable (for Render deployment)
    # Render sets the PORT environment variable
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"\n🌐 Access the application at: http://0.0.0.0:{port}")
    print("=" * 60 + "\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
