import requests

url = 'http://localhost:8000/api/v1/chat'
headers = {'Content-Type': 'application/json', 'x-tenant-id': 'aditya-ds'}
data = {'query': 'what do we have in the sources', 'session_id': 'test-123'}

try:
    resp = requests.post(url, headers=headers, json=data)
    print('STATUS', resp.status_code)
    print('RESPONSE', resp.text)
except Exception as e:
    print('ERR', e)
