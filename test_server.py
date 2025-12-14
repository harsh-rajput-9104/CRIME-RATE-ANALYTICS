#!/usr/bin/env python3
"""
Minimal test sif __name__ == '__main__':
    print("🧪 Starting minimal test server...")
    print("🌐 Access at: http://localhost:5000")
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        import traceback
        traceback.print_exc()debug the login issue
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app, origins=["*"], allow_headers=["Content-Type", "Authorization"])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

@app.route('/auth/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        print(f"Login attempt: {username}")
        
        # Simple test authentication
        if username == 'admin' and password == 'admin123':
            return jsonify({
                'status': 'success',
                'access_token': 'test_token_12345',
                'user': {'username': username, 'role': 'admin'}
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🧪 Starting minimal test server...")
    print("🌐 Access at: http://localhost:5000")
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        import traceback
        traceback.print_exc()
