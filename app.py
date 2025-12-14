#!/usr/bin/env python3
"""
Main entry point for the Flask application.
This ensures Docker can find the app using the standard name.
"""

from secure_app import app

if __name__ == '__main__':
    # This will invoke the __main__ block in secure_app.py
    import secure_app
