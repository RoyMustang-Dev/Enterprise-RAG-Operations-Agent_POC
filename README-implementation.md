# AKO AI Support Agent: Master Implementation Guide

This document (`README-implementation.md`) serves as the definitive Single Source of Truth for the architecture, routing logic, model selection, and execution roadmap for transforming the Enterprise RAG POC into a production-grade, API-first Multi-Tenant SaaS platform.

It aggregates all design decisions, zero-cost strategies, and feature pipelines discussed and implemented across all project phases.

---

## 1. Executive Summary: The SaaS Multimodal Shift

The objective is to deliver a fully headless, white-label AI SaaS platform that allows external clients to create their own custom support bots. 

**Key Transformation Highlights:**
*   **UI Decoupling:** The monolithic Streamlit application is sequestered. The core system exclusively exposes robust **FastAPI endpoints (REST/OpenAPI)**.
*   **Traceability & Audit Logging:** Every agent action, confidence score, and routing path is explicitly traced, surfaced to the frontend UI, and persistently logged to `data/audit/audit_logs.jsonl` for enterprise accountability.
*   **Multi-Tenancy:** Routes are designed to differentiate tenant data, allowing various clients to customize their Agent Name, Prompt instructions, and branding.
*   **Multimodal Capabilities:** The Chat APIs natively accept `multipart/form-data`, meaning the AI can process standard Text, Voice Notes (.wav/.mp3), Images (.jpg, .png), and temporary Document URLs.
*   **Action-Oriented Agents:** The core `SupervisorAgent` doesn't just answer questions; it routes logic to specialized sub-agents capable of executing Webhooks (e.g., generating Tickets, processing Refunds, engaging Human Handoffs).

---

## 2. API Architecture & Endpoint Design

The backend is structured into modular FastAPI routers (`backend/routers/`). The auto-generated Swagger documentation sits at `/docs`.

### A. Bot Setup & Configuration (`/api/v1/settings`)
*   **`POST /configure`:** Accepts the initial client dashboard payload (Bot Name, Company Logo, Brand Details, System Prompt).

### B. Knowledge Base Ingestion (`/api/v1/ingest`)
*   **`POST /files`:** Handles manual uploads (PDF/DOCX) and converts them into searchable vectors. Supports `append` or `start_fresh` modes.
*   **`POST /crawler`:** Triggers an advanced headless crawler against a provided URL to scrape a website and auto-generate the Knowledge Base (incorporating the `Aeko-test-crawl` repo logic).

### C. Multimodal Chat Execution (`/api/v1/chat`)
*   **`POST /`:** The primary conversation interface. Accepts `message` alongside optional `audio_file`, `image_file`, or `document_file`. Safely returns `<think>` blocks, confidence scores, and hallucination flags.

---

## 3. Explicit LLM Routing & Multi-Provider Strategy

To ensure enterprise stability, prevent vendor lock-in, and optimize costs, the backend implements a deterministic `LLMRouter` (`backend/generation/llm_router.py`).

We explicitly assign *which* model performs *which* task using Environmental Variables (`.env`), separating the "Core Brain" from "Situational Tools."

### How Routing Works:
1.  **Core Reasoning Engine (`CORE_LLM_PROVIDER`):** 
    *   This handles the vast majority of tasks: general chat logic, classifying RAG intents, invoking tool webhooks, and structuring the final response.
    *   *Default Setting:* `sarvam` (The user's preferred enterprise orchestrator).
    *   *Fallback Sequence:* If Sarvam is unavailable, it gracefully degrades: `Anthropic` -> `OpenAI` -> `Groq` -> `Local Ollama`.
2.  **Situational Multimodal Models:**
    *   If a client uploads an image, the `CORE` model might lack vision capabilities.
    *   The `/chat` endpoint explicitly overrides the core logic and requests a target provider: `target_provider="gemini"`.
    *   *The Result:* The `LLMRouter` triggers Gemini *only* for the image extraction sub-task, grabbing the text description, and handing it back to Sarvam to compile the final answer.

---

## 4. The Zero-Cost Model Execution Strategy

We are implementing a matrix of Free-Tier APIs and lightweight Local Models to deliver premium SaaS capabilities without paying per-token API overheads.

### The Categorical Deployment Matrix:

| Capability | Assigned Model | Deployment Type | Cost Tier | Purpose in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Embeddings** | `bge-large-en-v1.5` | 🏠 Local (Python) | 🟢 100% Free | Converts documents & text to vectors. Highly accurate (Must be L2 Normalized). |
| **Vector Index** | `Qdrant` / `Pinecone` | ☁️ Cloud DB | 🟢 Free Tier | Serverless Vector DB (Replaces FAISS). Pinecone = managed, Qdrant = managed + OSS. |
| **Core Brain** | `Sarvam API` | ☁️ Cloud API | 🟢 Free/Client Key | Main logic, agent routing, intent classification. |
| **Failsafe Brain**| `Groq` (Llama 3) | ☁️ Cloud API | 🟢 Free Tier | Lightning-fast text generator (800 tps). Used if main API fails. |
| **Vision (Image)** | `Gemini 1.5 Flash` | ☁️ Cloud API | 🟢 Free Tier | Situational LLM explicitly invoked to describe user-uploaded images/errors. |
| **Audio (STT)** | `faster-whisper` | 🏠 Local (Python) | 🟢 100% Free | Instant offline transcription of Voice Notes (`.wav` -> text). |

---

## 5. Enterprise Data Retrieval Upgrade (Vector DB & Chunking)

The original prototype relied on a flat, local FAISS index (`faiss_store.py`) and basic word-splitting logic. To achieve true Enterprise status, the architecture is being upgraded:

### A. Qdrant vs. Pinecone Database Migration
The local FAISS file is a bottleneck and prevents elastic horizontal scaling. We are transitioning to an Enterprise-grade Cloud Vector DB. 

**Is Pinecone a Downgrade?**
**Absolutely not.** Pinecone is a $1.5 Billion+ enterprise industry standard used by Notion and Zapier. Moving from FAISS to Pinecone or Qdrant is a massive *upgrade*.

**Comparison for a Completely Free Solution:**
*   **Qdrant:** Offers a "1GB Free Forever" Cloud Cluster, which is generous enough to store ~1 million 768-dimensional vectors. It is completely open-source (Rust), meaning if the Cloud tier limits you, you can self-host the exact same DB via Docker for absolutely zero software cost.
*   **Pinecone:** Offers a fully managed "Serverless Starter" free tier featuring 2GB of index storage (~300k - 1 Million vectors) and 2 Million writes/month. It is closed-source and fully managed.
*   **Decision:** We will support **Qdrant** as the primary `VectorStore` adapter since it provides maximum flexibility (Cloud + Local Docker hosting) and fits the "completely free" SaaS scaling ethos beautifully, while keeping Pinecone as an interchangeable backend module.

### B. Intelligent Chunking & Embedding Normalization
*   **Token-Aware Recursive Splitter:** We are replacing the blind "512 word limit" semantic chunker (which unknowingly chops off data because 512 words exceeds 512 tokens). We will use LangChain's `RecursiveCharacterTextSplitter` paired with the BGE HuggingFace tokenizer to chunk perfectly by paragraphs/sentences.
*   **Cosine Normalization:** The `bge-large-en-v1.5` embeddings explicitly require `normalize_embeddings=True` to execute Cosine Similarity correctly. We will enforce this in `embedding_model.py` to prevent retrieval skew.

### C. Advanced Retrieval (Phase 2 RAG)
*   Implementation of **Cross-Encoder Re-ranking** (e.g., retrieving 30 chunks and having an AI reranker strictly score the top 5).
*   Dynamic LLM-based Metadata Extraction (replacing hardcoded filename filters with dynamic `$eq` JSON payload queries).

---

## 6. Implementation Roadmap (Status Tracker)

*   **Phase 6: Exposing API Interfaces ✅ (COMPLETED)**
    *   FastAPI modular routing (`/chat`, `/ingest`, `/settings`).
    *   Deterministic `LLMRouter` multi-provider logic implemented.
*   **Phase 7: Full Traceability & Audit Logging ✅ (COMPLETED)**
    *   Integrated confidence scores, hallucination flags, and true routing optimizations into the Streamlit UI without truncation.
    *   Successfully persisted complex JSONL logging natively in `backend/generation/rag_service.py`.
*   **Phase 8: Enterprise Data Ingestion Upgrade ⏳ (IN PROGRESS)**
    *   Migrate from FAISS to Qdrant/Pinecone `VectorStore`.
    *   Implement Token-Aware Recursive text splitting.
    *   Enforce L2 Normalization on the BGE Embedding model.
*   **Phase 9: External Production Crawler Integration**
    *   Audit and replace legacy scraping scripts with the robust `Aeko-test-crawl` repository.
*   **Phase 10: Multimodal Processing Pipelines**
    *   Implement `faster-whisper` transcription for Audio inputs.
    *   Implement target Gemini Vision processing for Image inputs.
*   **Phase 11: Action-Oriented CRM Agents**
    *   Implement `TicketingAgent`, `RefundAgent`, and `HumanEscalationAgent` Webhooks.
*   **Phase 12: Hardware Optimization & API Fallbacks**
    *   Integrate Cloud Model API routing to bypass local CPU starvation bottlenecks.

---

## 7. Cloud API Fallback Stack (Phase 12)

If a client deploys this Enterprise RAG system on a weak virtual server (e.g., 2 CPU cores, 4GB RAM), downloading the 1.5GB `BAAI` vector model or running a local 8B LLM will permanently crash their system with Out-Of-Memory (OOM) errors. 

To prevent this, the architecture natively intercepts `.env` API keys and physically bypasses local PyTorch execution, transforming the deployment into a highly scalable, serverless Cloud Router.

### 1. Vectorization: Embedding APIs
Cloud embedding APIs offer lightning-fast vector array calculations mapped to identical dimensional bounds (1024) without requiring local GPUs.
| Provider | Model | Cost (per 1M tokens) | Highlights |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `text-embedding-3-small` | **$0.02** | The industry standard for cost-to-performance. Maps a 10,000-page wiki for less than $0.50 in under a minute. |
| **Cohere** | `embed-english-v3.0` | $0.10 | Excellent multilingual support and native Multimodal (Text + Image) embeddings. |
| **Voyage AI** | `voyage-3` | $0.06 | Exceptional at specialized domains (finance, law, code). Massive 200M free token initial tier. |

*Architectural Default:* OpenAI `text-embedding-3-small` (if `OPENAI_API_KEY` is present).

### 2. Core Brain LLM (Reasoning & Orchestration)
The "Core Brain" requires high intelligence, massive context windows, and reliable structured JSON output for strict LangGraph array routings.
| Provider | Model | Cost (per 1M In/Out) | Context Window | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Google** | `gemini-1.5-pro` | $1.25 / $5.00 | **1M - 2M tokens** | **Best Overall Value.** Perfect for stuffing massive chunks of documentation into a single system prompt cheaply. |
| **OpenAI** | `gpt-4o` | $2.50 / $10.00 | 128K tokens | **Fastest & Most Reliable Supervisor.** Unbeatable at strictly following nested tool-calling schemas. |
| **Anthropic** | `claude-3-5-sonnet` | $3.00 / $15.00 | 200K tokens | **Highest Raw Intelligence.** Dominates in dynamic coding synthesis and hallucination-free reasoning. |

*Architectural Default:* `gemini-1.5-pro` offers the ultimate price-to-performance ratio for massive RAG text ingestion.

### 3. Native Vision APIs (Image Parsing)
If the web crawler encounters complex structural diagrams or UI screenshots, simple OCR fails natively. These models "read" arrays purely visually.
| Provider | Model | Cost | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Google** | `gemini-1.5-flash` | $0.075 (per 1M) | **Mass Scaling.** Feed it hundreds of page screenshots and extract Markdown tables instantly for pennies. |
| **Anthropic** | `claude-3-5-sonnet` | ~$0.003 - $0.01 per Image | **Precision Data.** Flawless at transcribing highly convoluted enterprise chart data into strict JSON format structures. |

---

# LLM Provider Comparison Matrix: Sarvam vs Groq

Based on the capabilities extracted from the Groq API and our existing implementation of `Sarvam AI`, here is a comprehensive analysis of how they stack up for this Enterprise RAG system.

### Current Baseline: Sarvam AI (`sarvam-m`)
- **Capabilities:** Core reasoning, highly optimized for Indian enterprise context (indic languages). 
- **Infrastructure:** Cloud API.
- **Cost:** Free Tier / Pay-as-you-go based on client key.
- **Latency:** Moderate (standard HTTP round trips).
- **Known Issues:** The `/chat/completions` exact structure requires custom workarounds (e.g., streaming breaks if `reasoning_effort` is sent).

### The New Contender: Groq
Groq utilizes LPU (Language Processing Unit) architecture, dramatically reducing Time-To-First-Token (TTFT) and achieving massive generation speeds (often 800+ tokens per second). 

Based on the `GET /models` payload you provided, here are the standout Enterprise-grade models available to us for free on Groq:

#### 1. The Core Reasoning Brain: `llama-3.3-70b-versatile`
- **Context Window:** 131,072 tokens.
- **Max Output:** 32,768 tokens.
- **Why it matters:** Llama 3.3 70B is an absolute powerhouse. It rivals GPT-4o in reasoning capabilities. Running on Groq, it will generate complex, highly intelligent answers natively at speeds Sarvam cannot match physically. It perfectly drops in as our `CORE_LLM_PROVIDER`.

#### 2. The Speedy Failsafe / Smalltalk Bypass: `llama-3.1-8b-instant`
- **Context Window:** 131,072 tokens.
- **Why it matters:** This model is so fast on Groq that it feels instantaneous. We can use this to power the `SmalltalkAgent` and handle rapid intent classification (the `SupervisorAgent`).

#### 3. Voice Transcription (STT): `whisper-large-v3-turbo` / `whisper-large-v3`
- **Why it matters:** We currently planned to use `faster-whisper` locally for Phase 10 Multimodal Voice Notes. Groq offers OpenAI's Whisper model via API instantly. We can offload STT processing to Groq to save local CPU overhead without paying OpenAI API costs.

#### 4. The Analytics/Code Agent: `qwen2.5-coder-32b` (implied by `qwen3-32b` in the list)
- **Why it matters:** Qwen models are exceptional at structured data extraction (JSON parsing from documents).

---

### Groq vs Sarvam Verdict

| Feature | Sarvam AI | Groq (Llama 3.3 70B) |
| :--- | :--- | :--- |
| **Speed/Latency** | Moderate (Standard) | **Extreme (800+ tokens/sec)** |
| **Reasoning Quality**| Good | **State-of-the-Art (Rivals GPT-4o)** |
| **Multimodal STT** | None | **Built-in (Whisper Large V3)** |
| **Context Window** | Standard API limits | **Massive (128K context for giant RAG chunks)**|
| **Streaming UI Feel**| Good | **Incredible (Feels like real-time reading)** |

### Recommendation
**We should absolutely migrate the `LLMRouter` to prioritize Groq.** The Llama 3.3 70B model running on LPU architecture will aggressively solve the latency we typically experience with document reasoning. 

Furthermore, because it uses standard OpenAI client libraries (`openai.AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key="...")`), integrating it will instantly fix the weird prompt formatting bugs we had to code around for Sarvam.

---

### Phase 9: Pinokio Assessment
You asked me to check out `pinokio.co/docs/#/`. 
Pinokio is effectively an "AI Browser/OS". It packages complex AI applications (like Stable Diffusion, local LLMs, and ComfyUI) into one-click installable widgets locally. 

**How we can harness Pinokio:**
Rather than forcing enterprise clients to clone this repo, install Python `venv`, and run Uvicorn/Streamlit manually in two terminals... we can write a `pinokio.json` setup script. 
This means a client could install our "Enterprise RAG Agent" completely isolated on their machine with **a single click** using Pinokio's framework. It's the ultimate no-code deployment strategy for this codebase.

---

## Comprehensive Deep Dive: Groq API Model Capabilities

The provided Groq API key unlocks access to a series of highly requested, cutting-edge, and even preview-tier models. Below is a thorough evaluation of every model listed in the payload and how they can be leveraged in our Enterprise SaaS architecture:

### 1. The Core Reasoning Engines (General Purpose)
*   **`llama-3.3-70b-versatile` (Meta):** A 70 billion parameter model functioning as the current state-of-the-art open-source logic engine. It features a 131K context window. **Use Case:** Our primary `CORE_LLM_PROVIDER` for complex RAG synthesis and routing decisions. It rivals GPT-4o but runs at 800+ TPS natively on Groq.
*   **`openai/gpt-oss-120b` & `openai/gpt-oss-20b` (OpenAI):** Highly efficient open-weight Mixture-of-Experts (MoE) models from OpenAI, optimized for function calling and logic. The 120B version is a heavy-duty reasoning engine, while the 20B is a low-latency variant. **Use Case:** Excellent alternatives to Llama 3.3 for rigorous agentic workflows or structured tool executions.
*   **`moonshotai/kimi-k2-instruct-0905` & `kimi-k2-instruct` (Moonshot AI):** Renowned for industry-leading context windows. The `0905` version supports an astronomical **262,144 tokens**. **Use Case:** Perfect for "Needle In A Haystack" RAG queries where we need to feed entire books or hundreds of documents in a single prompt without chunking.

### 2. The Speedy Assistants (Low Latency / High Throughput)
*   **`llama-3.1-8b-instant` (Meta):** An ultra-fast, lightweight 8B model. **Use Case:** Pefectly powers our `SmalltalkAgent` and acts as the instantaneous intent classifier for the `SupervisorAgent`.
*   **`qwen/qwen3-32b` (Alibaba Cloud):** The latest Qwen architecture uniquely excels at coding and structured JSON generation. **Use Case:** Deployed as an "Extraction Agent" to parse unstructured data into pure JSON formats for CRM integrations.

### 3. Agentic & Tool-Use Specialists (Compound Systems)
*   **`groq/compound` & `groq/compound-mini` (Groq):** Advanced composite AI systems currently in Beta, specifically designed to autonomously solve problems using external tools (like web search or code execution interpreters). They string together multiple Llama and GPT-OSS models under the hood. **Use Case:** Future-proofing for Phase 11, where our agents need to autonomously browse the live web or execute Python scripts to answer a user's query.

### 4. The Preview Models: Meta's Next Generation (Llama 4 Series)
It appears Groq has early access to preview builds of Meta's unreleased upcoming **Llama 4** architecture:
*   **`meta-llama/llama-4-scout-17b-16e-instruct`:** A 109B parameter MoE model with 17B active parameters, reportedly featuring a mind-blowing **10 million token context window** and native multimodal image support. 
*   **`meta-llama/llama-4-maverick-17b-128e-instruct`:** The largest Llama 4 variant (400B total parameters), designed as the pinnacle of coding, reasoning, and creative generation.
*   **Use Case:** Highly experimental, but gives our platform immediate access to the absolute bleeding-edge of AI reasoning once stabilized.

### 5. Safety & Content Moderation (Trust & Safety Middleware)
*   **`openai/gpt-oss-safeguard-20b`:** A reasoning-based safety classifier that interprets developer policies to block malicious content transparently.
*   **`meta-llama/llama-prompt-guard-2-86m` & `22m`:** Lightning-fast micro-models specifically built to detect prompt-injection attacks and jailbreaks natively.
*   **`meta-llama/llama-guard-4-12b`:** Llama 4's native multimodal safety classifier.
*   **Use Case:** Critical for Enterprise SaaS. We can interject these models as "Middleware" in the API to automatically block users from attempting to jailbreak the bot or generate inappropriate content, saving our core LLM from executing malicious instructions.

### 6. Regional & Auditory Specialists
*   **`allam-2-7b` (SDAIA):** A specialized bilingual Arabic-English model trained by Saudi Arabia's AI authority.
*   **`canopylabs/orpheus-arabic-saudi` & `orpheus-v1-english`:** Specialized Text-To-Speech models prioritizing explicit vocal directions and regional dialects natively hosted on GroqCloud.
*   **Use Case:** Opens up the Middle Eastern Enterprise market by allowing our RAG pipeline to generate native Saudi Arabic responses and audio natively.

### 7. Multimodal Audio Extraction (Speech-To-Text)
*   **`whisper-large-v3` & `whisper-large-v3-turbo` (OpenAI):** The absolute gold standard for Speech-To-Text processing. 
*   **Use Case:** Phase 10 Multimodal Chat. Users can send WhatsApp voice notes to the API. We pass the audio to Whisper on Groq's API, it instantly transcribes it at zero cost to our local CPU, and the text is fed into the RAG pipeline.

---

## Phase 12: Architecture Design - The "Mixture of Experts" Multi-Agent OS

As discussed in your audio note, relying on a single monolith agent to handle all queries limits the system's potential. We can leverage Groq's instantaneous speed to construct a true **Multi-Agent System (MAS)** mimicking a "Mixture of Experts" architecture using **LangGraph**.

Instead of one prompt trying to do everything, we deploy a network of tailored micro-agents, each armed with specific tools and system prompts. 

### The Orchestrator: Autonomous Supervisor Agent
The central nervous system of the architecture is the **Supervisor Agent**. Powered by the lightning-fast `llama-3.1-8b-instant`, it intercepts every user query. 
*   **Function:** It is strictly a semantic router. It does not answer questions. It analyzes the intent and autonomously forwards the state to the correct Expert.
*   **Autonomy:** It prevents hallucinations. If a user asks to book a flight, it routes to Booking. If the user asks about an enterprise PDF, it routes to RAG.

### The Specialist Network (The "Experts")

1.  **The Enterprise RAG Agent**
    *   **Driven by:** `llama-3.3-70b-versatile`
    *   **Tools:** `QdrantRetriever`, `DocumentMetadataFilter`
    *   **Purpose:** Strict, grounded fact-answering based ONLY on uploaded vector knowledge. If it doesn't know, it says it doesn't know.

2.  **The Support & Troubleshooting Agent**
    *   **Driven by:** `qwen-32b` (Excellent at structured steps)
    *   **Tools:** `KnowledgeBaseSearch`, `SubmitSupportTicketAPI`
    *   **Purpose:** Walks users through technical issues using a decision tree. If unresolved, it uses a tool to autonomously push a ticket payload (JSON) into Jira/Zendesk.

3.  **The Lead Generation / Capture Agent**
    *   **Driven by:** `llama-3.3-70b-versatile`
    *   **Tools:** `CRM_API_Push` (Salesforce/HubSpot)
    *   **Purpose:** Detects buying intent. Seamlessly transitions the conversation into collecting the user's Name, Email, and Company Size, then autonomously fires a payload into the CRM.

4.  **The Booking & Scheduling Agent**
    *   **Driven by:** `groq/compound` (For multi-tool calendar execution)
    *   **Tools:** `CheckGoogleCalendarBusy`, `CreateCalendarInvite`
    *   **Purpose:** Understands natural date language ("Next Tuesday afternoon"), queries the local calendar for free slots, proposes times, and creates the Google Meet invite.

5.  **The E2E Conversational / Smalltalk Agent**
    *   **Driven by:** `llama-3.1-8b-instant`
    *   **Purpose:** Handles greetings, pleasantries, and out-of-bounds questions playfully without burning expensive API tokens or hitting vector databases.

### How It Connects Autonomously
Because we are utilizing **LangGraph**, the `AgentState` Dictionary is passed continuously between nodes. 
A user might start by asking for support (Routed -> Support Agent). The Support Agent solves the issue. The user then says, *"Thanks! I'd love to schedule a demo of the premium tier."* 
The graph detects this shift in intent. The Supervisor immediately routes the trailing state down to the **Booking Agent**, which takes over the conversation context seamlessly. 

This architecture allows us to plug-and-play infinite tools and CRMs without ever bloating the main engine.
