import subprocess
import time
import requests

with open('server_crash.log', 'w') as f:
    proc = subprocess.Popen(['python', '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8005'], stdout=f, stderr=subprocess.STDOUT)
    
time.sleep(35) # wait for GPU load
url = 'http://127.0.0.1:8005/api/v1/chat'
headers = {'Content-Type': 'application/json', 'x-tenant-id': 'aditya-ds'}
data = {'query': 'what do we have in the sources', 'session_id': 'test-124'}

try:
    resp = requests.post(url, headers=headers, json=data)
    print('STATUS', resp.status_code)
except Exception as e:
    pass

time.sleep(5)
proc.kill()
