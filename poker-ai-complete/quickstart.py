#!/usr/bin/env python3
"""
Quick Start Script - Train model and test integration
"""

import sys
import subprocess


def print_header(text):
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")


def main():
    print_header("🚀 POKER AI QUICK START")
    
    print("This script will:")
    print("1. Test the integration (without server)")
    print("2. Show you how to connect to the game server")
    print()
    
    # Run integration tests
    print_header("Step 1: Running Integration Tests")
    try:
        result = subprocess.run([sys.executable, "integration_test.py"], 
                              capture_output=False)
        if result.returncode != 0:
            print("\n⚠️  Some tests failed. Review the output above.")
            print("You can still try connecting to the server, but there may be issues.")
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return
    
    # Instructions
    print_header("Step 2: Connect to Game Server")
    
    print("Your model is ready to play! Here's how to connect:\n")
    
    print("📋 CHECKLIST:")
    print("   ☐ Go server is running (go run cmd/main.go)")
    print("   ☐ Server is on localhost:8080")
    print("   ☐ API_KEY matches (default: 'dev')")
    print("   ☐ TABLE_ID exists (default: 'table-1')")
    print()
    
    print("🎮 TO PLAY:")
    print("   python poker_client.py")
    print()
    
    print("🤖 TO RUN MULTIPLE BOTS:")
    print("   # Terminal 1")
    print("   python poker_client.py")
    print()
    print("   # Terminal 2 (edit PLAYER_ID to 'bot2' first)")
    print("   python poker_client.py")
    print()
    
    print("📝 CONFIGURATION:")
    print("   Edit poker_client.py to change:")
    print("   - API_KEY (must match server)")
    print("   - TABLE_ID (must match registered table)")
    print("   - PLAYER_ID (unique per bot)")
    print("   - SERVER_HOST/PORT")
    print("   - MODEL_PATH (to load saved weights)")
    print()
    
    print("📚 MORE INFO:")
    print("   See INTEGRATION_GUIDE.md for detailed documentation")
    print()
    
    print_header("✅ Setup Complete!")


if __name__ == "__main__":
    main()
