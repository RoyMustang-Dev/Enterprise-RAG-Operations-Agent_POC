from app.api.routes import _get_orchestrator
import asyncio
import traceback

async def test():
    try:
        orch = _get_orchestrator()
        query = 'what do we have in the sources'
        tenant_id = 'aditya-ds'
        
        # Proper invocation matching api/routes.py
        res = await orch.invoke(query=query, tenant_id=tenant_id, session_id='test-123')
        print("SUCCESS:", res['answer'])
    except Exception as e:
        traceback.print_exc()

asyncio.run(test())
