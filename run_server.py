#!/usr/bin/env python3
"""
Server runner with better error handling
"""
import time
import traceback

def run_server():
    try:
        print("🔄 Starting server...")
        import secure_app
        print("✅ Server module imported successfully")
        
        # Keep the server running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    run_server()
