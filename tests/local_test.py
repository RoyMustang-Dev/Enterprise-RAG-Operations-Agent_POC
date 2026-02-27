import subprocess
import time
import requests

proc = subprocess.Popen(["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(8)
try:
    url = 'http://localhost:8001/api/v1/chat'
    headers = {'accept': 'application/json', 'x-tenant-id': 'enterprise_rag', 'Content-Type': 'application/json'}
    payload = {
        'query': 'the support agent.docx is the document provided by my client in which he has mentioned the complete requirement with the proper flow. can you verify by doing deep research whether its possible for Aditya to build them or not?',
        'chat_history': []
    }
    print("Sending Request to 8001...")
    res = requests.post(url, headers=headers, json=payload, timeout=60)
    print("API RESPONSE:", res.status_code)
except Exception as e:
    print("Request Failed:", e)
finally:
    proc.terminate()
    out, err = proc.communicate(timeout=5)
    print("\n--- STDOUT ---")
    print(out.decode('utf-8', errors='ignore'))
    print("\n--- STDERR ---")
    print(err.decode('utf-8', errors='ignore'))
