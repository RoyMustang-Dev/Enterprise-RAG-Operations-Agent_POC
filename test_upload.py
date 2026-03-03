import subprocess
import time
import requests

with open('server_upload.log', 'w') as f:
    proc = subprocess.Popen(['python', '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8005'], stdout=f, stderr=subprocess.STDOUT)
    
time.sleep(35) # wait for GPU load
url = 'http://127.0.0.1:8005/api/v1/ingest/files'
headers = {'x-tenant-id': 'aditya-ds'}
files = {'files': ('dummy.txt', 'Multi-tenant debugging is important. RAG system works perfectly with Qdrant.', 'text/plain')}

try:
    resp = requests.post(url, headers=headers, files=files)
    print('UPLOAD STATUS', resp.status_code)
except Exception as e:
    pass

url2 = 'http://127.0.0.1:8005/api/v1/chat'
headers2 = {'Content-Type': 'application/json', 'x-tenant-id': 'aditya-ds'}
data2 = {'query': 'what do we have in the sources', 'session_id': 'test-124'}

try:
    resp2 = requests.post(url2, headers=headers2, json=data2)
    print('CHAT STATUS', resp2.status_code)
    print('CHAT RESP', resp2.text)
except Exception as e:
    pass

time.sleep(5)
proc.kill()
