# Enterprise RAG Operations Agent

## End-to-End Implementation Plan (POC → Production-Ready Architecture)

> Version: 1.0
> Author: Aditya Mishra

Purpose: Build a complete Retrieval-Augmented Generation (RAG) system with ML ingestion, vector retrieval, LLM orchestration, agent workflows, governance, and deployment.

## 0. Project Objective

Build a production-grade GenAI system that allows users to:

- Upload enterprise documents (PDF, DOCX, TXT)
- Crawl URLs
- Generate embeddings
- Store vectors
- Perform semantic retrieval
- Query documents via LLM
- Execute tool actions
- Observe system behavior
- Enforce governance

`This is not a chatbot.`
`This is an Enterprise Knowledge Operations Agent.`

## 1. Core Principles

- Python-first
- Deterministic pipelines
- Explicit orchestration
- Vendor-agnostic LLM routing
- Stateless APIs
- Vector-first retrieval
- Human-in-the-loop
- Observability-first
- Secure by default

`No magic.`
`No implicit chains.`
`All logic explicit.`

## 2. High-Level Architecture
```
[ Frontend (Streamlit) ]
           |
           v
[ FastAPI Gateway ]
           |
           v
[ Input Normalization ]
           |
           v
[ Ingestion Pipeline ]
           |
           v
[ Embedding Engine ]
           |
           v
[ Vector Store (FAISS) ]
           |
           v
[ Retriever ]
           |
           v
[ RAG Orchestrator ]
           |
           v
[ Ollama's Llama 3.2 ]
           |
           v
[ Answer Synthesizer ]
           |
           v
[ Response + Sources ]

Parallel:
PostgreSQL → Metadata
Redis → Sessions
Prometheus → Metrics
```
## 3. Technology Stack

> Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

> Frontend

- Streamlit

> ML

- sentence-transformers (all-MiniLM-L6-v2)
- HuggingFace Transformers

> Vector DB

- FAISS (local)

> LLM

- Ollama's Llama 3.2

> Storage

- PostgreSQL (metadata)
- Redis (session state)

> DevOps

- Docker
- Docker Compose

## 4. Repository Layout

```
enterprise-rag-agent/
│
├── backend/
│   ├── main.py
│   ├── ingestion/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retriever/
│   ├── rag/
│   ├── llm/
│   ├── tools/
│   ├── observability/
│
├── frontend/
│   └── app.py
│
├── data/
│
├── docker/
│
├── requirements.txt
└── README.md
```

## 5. Phase 1 — Input + Ingestion

> Supported Inputs

- PDF
- DOCX
- TXT
- URL

> Steps

- Upload file
- Extract text
- Normalize whitespace
- Chunk text (512 tokens, 20% overlap)
- Attach metadata:
  - filename
  - timestamp
  - source

> Libraries:

- PyMuPDF
- python-docx
- BeautifulSoup
- nltk

> Output:

```
{
  chunk_id,
  text,
  metadata
}
```

## 6. Phase 2 — Embeddings

> Use:

- sentence-transformers/all-MiniLM-L6-v2

> Process:

- Convert chunks → embeddings
- Persist vectors into FAISS
- Persist metadata into PostgreSQL

> Each chunk:

- vector_id
- embedding
- text
- source

## 7. Phase 3 — Vector Store

> FAISS Index:

- IndexFlatL2
- Persistent save/load
- Namespace by workspace.

## 8. Phase 4 — Retrieval

> Given user query:

- Embed query
- Search FAISS top-k (k=5)
- Return chunk texts + metadata

## 9. Phase 5 — RAG Orchestrator

> Build prompt:

```
System:

You are an enterprise knowledge agent.
Answer ONLY from provided context.
If answer not found, say "Not found in knowledge base."


User:

Context:
{{chunks}}

Question:
{{user_query}}
```

- Send to Ollama's Llama 3.2.

## 10. Phase 6 — Ollama's Llama 3.2 Integration

- Use Ollama. 
- No LangChain abstractions.
- Direct API calls only.

## 11. Phase 7 — Answer Synthesis

> Return:

```
{
 answer,
 sources,
 confidence_score
}
```

- Confidence = cosine similarity mean.

## 12. Phase 8 — Frontend (Streamlit)

> Features:

- Upload docs
- Ask questions
- Show retrieved sources
- Display answers
- Session history

## 13. Phase 9 — Observability

> Log:

- queries
- retrieved chunks
- latency
- tokens
- Store in PostgreSQL.
- Expose Prometheus endpoint.

## 14. Phase 10 — Governance

> POC-level:

- API key auth
- Rate limits
- Max tokens per request

## 15. Phase 11 — Tool Actions (Optional Extension)

> Add mock tools:

- send_email()
- create_ticket()

Agent chooses tools based on intent.

## 16. Phase 12 — Dockerization

> Docker services:

- backend
- frontend
- postgres
- redis

## 17. Phase 13 — Evaluation

> Metrics:

- Retrieval Recall
- Answer Faithfulness
- Latency
- Manual test cases.

## 18. Milestones

> Week 1

- Ingestion
- Embeddings
- FAISS

> Week 2

- RAG + Ollama's Llama 3.2
- Streamlit UI

> Week 3

- Observability
- Docker
- Demo

## 19. Final Deliverables

> GitHub repo
> Running Docker system
> Demo video
> README

## Final Statement

> This system must:

- Perform deterministic RAG
- Provide explainable sources
- Be deployable locally
- Be vendor-agnostic
- Support future agent expansion

`No shortcuts.`
`No black boxes.`
`All logic explicit.`