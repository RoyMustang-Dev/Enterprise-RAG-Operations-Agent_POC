import requests
import os
import sys

def check_ollama():
    urls = [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://172.30.170.89:11434" # WSL IP from user logs
    ]
    
    print(f"Diagnostics: Checking Ollama Connection...")
    print(f"Environment HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
    print(f"Environment HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")
    
    success = False
    
    for url in urls:
        print(f"\n--- Testing {url} ---")
        try:
            response = requests.get(url, timeout=5)
            print(f"Status Code: {response.status_code}")
            print(f"Content: {response.text}")
            if response.status_code == 200:
                print("✅ CONNECTION SUCCESSFUL")
                success = True
        except Exception as e:
            print(f"❌ CONNECTION FAILED: {e}")
            
    if success:
        print("\nSUMMARY: At least one endpoint is working.")
        
        # Check available models
        try:
            print("\nChecking available models...")
            tags_url = "http://localhost:11434/api/tags"
            response = requests.get(tags_url)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"Found {len(models)} models:")
                for model in models:
                    print(f" - {model.get('name')}")
            else:
                print(f"Failed to list models. Status: {response.status_code}")
        except Exception as e:
            print(f"Error listing models: {e}")

        # Try a generation
        try:
            print("\nAttempting basic generation...")
            gen_url = "http://localhost:11434/api/generate"
            # If localhost failed but others worked, swap here.
            # But let's verify connectivity first.
        except:
            pass
    else:
        print("\nSUMMARY: All connection attempts failed.")

if __name__ == "__main__":
    check_ollama()
