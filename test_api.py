#!/usr/bin/env python3
"""
API Test Script
This script tests various endpoints of the Crime Analytics API to ensure they're working properly.
"""

import requests
import json
import sys

def test_api(base_url="http://localhost:5000"):
    """Test various endpoints of the API"""
    
    print(f"🧪 Testing API at {base_url}")
    
    # Test health endpoint
    print("\n🔍 Testing /health endpoint...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: {json.dumps(resp.json(), indent=2)}")
            print("✅ Health endpoint working!")
        else:
            print(f"❌ Health endpoint failed: {resp.text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test login endpoint
    print("\n🔍 Testing /auth/login endpoint...")
    try:
        data = {"username": "admin", "password": "admin123"}
        resp = requests.post(
            f"{base_url}/auth/login", 
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            token = resp.json().get('access_token', '')
            print(f"Got access token: {token[:10]}...")
            print("✅ Login endpoint working!")
            return token
        else:
            print(f"❌ Login endpoint failed: {resp.text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    return None

def test_auth_endpoints(base_url="http://localhost:5000", token=None):
    """Test authenticated endpoints"""
    
    if not token:
        print("❌ No token available, skipping authenticated endpoint tests")
        return
    
    # Test user profile endpoint
    print("\n🔍 Testing /auth/me endpoint...")
    try:
        resp = requests.get(
            f"{base_url}/auth/me", 
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: {json.dumps(resp.json(), indent=2)}")
            print("✅ User profile endpoint working!")
        else:
            print(f"❌ User profile endpoint failed: {resp.text}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Main function"""
    base_url = "http://localhost:5000"  # Changed back to port 5000 for main server
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print("🚀 Starting API Tests...")
    token = test_api(base_url)
    
    if token:
        test_auth_endpoints(base_url, token)
    
    print("\n✨ API Tests completed!")

if __name__ == "__main__":
    main()
