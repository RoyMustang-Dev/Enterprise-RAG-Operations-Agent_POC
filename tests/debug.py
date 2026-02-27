import asyncio
from app.supervisor.router import ExecutionGraph

async def main():
    try:
        query = 'the support agent.docx is the document provided by my client in which he has mentioned the complete requirement with the proper flow. can you verify by doing deep research whether its possible for Aditya to build them or not?'
        orc = ExecutionGraph()
        result = await orc.invoke(
            query=query,
            chat_history=[],
            session_id='debug_session',
            tenant_id='enterprise_rag',
            model_provider='groq'
        )
        print('SUCCESS:', result)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(main())
