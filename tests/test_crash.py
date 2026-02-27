import httpx
import asyncio

async def trigger_crash():
    url = 'http://localhost:8000/api/v1/chat'
    headers = {'accept': 'application/json', 'x-tenant-id': 'enterprise_rag', 'Content-Type': 'application/json'}
    payload = {
        'query': 'the support agent.docx is the document provided by my client in which he has mentioned the complete requirement with the proper flow. can you verify by doing deep research whether its possible for Aditya to build them or not?',
        'chat_history': []
    }
    
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(url, headers=headers, json=payload, timeout=60)
            print(res.status_code)
            print(res.text)
    except Exception as e:
        print('Exception:', e)

asyncio.run(trigger_crash())
