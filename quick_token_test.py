#!/usr/bin/env python3
"""
Quick HuggingFace token test
"""
import os

def main():
    print("🔍 Checking HuggingFace Token...")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found!")
        print("💡 Set it with: export HF_TOKEN='your_token_here'")
        print("🌐 Get token from: https://huggingface.co/settings/tokens")
        return
    
    print(f"✅ HF_TOKEN found: {hf_token[:10]}...")
    
    # Test basic API access
    try:
        from huggingface_hub import whoami
        user_info = whoami(token=hf_token)
        print(f"✅ Token valid for user: {user_info['name']}")
    except Exception as e:
        print(f"❌ Token validation failed: {e}")
        print("💡 Check if your token has proper permissions")

if __name__ == "__main__":
    main()
