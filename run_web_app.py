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
    print("\n🌐 Access the application at: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
