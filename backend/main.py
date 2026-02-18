from fastapi import FastAPI

app = FastAPI(title="Enterprise RAG Operations Agent")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Enterprise RAG Operations Agent API is running"}
