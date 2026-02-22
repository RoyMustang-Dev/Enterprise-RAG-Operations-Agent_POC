import os
import sys
import re
import asyncio
import nest_asyncio
import requests
import streamlit as st

nest_asyncio.apply()

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.crawler import crawl_url
from backend.generation.rag_service import RAGService
from backend.vectorstore.faiss_store import FAISSStore

# Initialize Resources (cached)
@st.cache_resource
def get_pipeline():
    return IngestionPipeline()

@st.cache_resource
def get_rag_service():
    # Force cache reload to pick up FAISS constructor fixes
    return RAGService()

pipeline = get_pipeline()
rag_service = get_rag_service()

st.set_page_config(page_title="Enterprise RAG Agent", layout="wide")

st.title("Enterprise RAG Operations Agent")

st.sidebar.header("Operations")
option = st.sidebar.selectbox("Choose Action", ["Chat", "Ingest Documents", "System Health"])

# Global LLM Settings
st.sidebar.markdown("---")
st.sidebar.header("LLM Settings")

llm_choice = st.sidebar.radio(
    "Select Model Provider:",
    ("Local (Ollama llama3.2)", "Sarvam (sarvam-m)")
)

sarvam_api_key = ""
if "Sarvam" in llm_choice:
    sarvam_api_key = st.sidebar.text_input("Sarvam API Key", type="password")
    if not sarvam_api_key:
        st.sidebar.warning("API Key required for Sarvam models.")


if option == "System Health":
    st.subheader("System Status")
    try:
        res = requests.get("http://127.0.0.1:8000/health")
        if res.status_code == 200:
            st.success(f"Backend is online: {res.json()}")
        else:
            st.error("Backend returned an error")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")


elif option == "Ingest Documents":
    st.subheader("Ingest Documents")
    
    # Display Current Knowledge Base (Moved to Top)
    with st.expander("📂 Current Knowledge Base (Click to View)", expanded=False):
        try:
            # We need to instantiate FAISSStore safely
            # Note: This might be slow if the index is huge, but for POC it's fine.
            # Using specific import to avoid circular dep if needed, but it's already imported.
            store = FAISSStore() 
            docs = store.get_all_documents()
            if docs:
                st.write(f"**Total Documents:** {len(docs)}")
                for doc in docs:
                    st.text(f"• {doc}")
            else:
                st.info("Knowledge Base is empty.")
        except Exception as e:
            st.error(f"Could not load Knowledge Base details: {e}")
            
    st.divider()
    
    st.markdown("Upload files, enter a URL, or do both. The system will process all inputs into the knowledge base.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Upload Files")
        uploaded_files = st.file_uploader("References (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    with col2:
        st.markdown("### 2. Crawl URL")
        url_input = st.text_input("Target URL (e.g., https://learnnect.com)")

    # Ingestion Mode Selection
    st.subheader("Ingestion Settings")
    ingestion_mode = st.radio(
        "Ingestion Mode:",
        ("Start Fresh (Clear Logic)", "Append to Knowledge Base"),
        help="Start Fresh will wipe the existing database. Append will add new documents to it."
    )
    
    if ingestion_mode == "Append to Knowledge Base":
        st.warning("⚠️ Warning: When appending, ensure you are not uploading duplicate documents. Logic to detect duplicates is currently experimental.")

    if st.button("Start Ingestion"):
        if not uploaded_files and not url_input:
            st.error("Please provide at least one source (File or URL).")
        else:
            with st.spinner("Processing... This may take a while."):
                
                paths_to_process = []
                metadatas_to_process = []
                
                # Process Files
                if uploaded_files:
                    st.write(f"📂 Saving {len(uploaded_files)} file(s)...")
                    os.makedirs("data/uploaded_docs", exist_ok=True)
                    
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join("data/uploaded_docs", uploaded_file.name)
                        try:
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            paths_to_process.append(temp_path)
                            metadatas_to_process.append({"type": "file", "original_name": uploaded_file.name})
                        except Exception as e:
                            st.error(f"❌ Error saving {uploaded_file.name}: {e}")

                # Process URL
                if url_input:
                    st.write(f"🌐 Crawling {url_input}...")
                    
                    # Ensure Proactor loop on Windows for Playwright
                    if sys.platform == "win32":
                        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        text = loop.run_until_complete(crawl_url(url_input))
                        loop.close()
                        
                        if text:
                            # Re-parsing URL to guess path
                            from urllib.parse import urlparse
                            parsed_url = urlparse(url_input)
                            domain = parsed_url.netloc
                            path = parsed_url.path.strip("/")
                            folder_name = re.sub(r'[<>:"/\\|?*]', '_', f"{domain}_{path}" if path else domain)
                            content_path = os.path.join("data", "crawled_docs", folder_name, "content.txt")
                            
                            if os.path.exists(content_path):
                                paths_to_process.append(content_path)
                                metadatas_to_process.append({"type": "url", "source_url": url_input})
                                st.write(f"✅ Crawled and indexed content from {url_input}")
                            else:
                                st.warning(f"Crawled but content file not found at {content_path}")
                        else:
                            st.error(f"❌ Failed to extract content from {url_input}")
                    except Exception as e:
                        st.error(f"❌ Error crawling URL: {e}")
                        st.exception(e)

                # Run Pipeline
                if paths_to_process:
                    st.write("🧠 Generating Embeddings & Updating Vector Store...")
                    try:
                        pipeline = get_pipeline()
                        reset_flag = True if ingestion_mode == "Start Fresh (Clear Logic)" else False
                        
                        num_chunks = pipeline.run_ingestion(paths_to_process, metadatas=metadatas_to_process, reset_db=reset_flag)
                        if num_chunks:
                            st.success(f"✅ Successfully ingested {num_chunks} chunks into the Knowledge Base!")
                            # Reload the RAG Service's vector store to pick up changes
                            rag_service.vector_store.load()
                            st.rerun()
                        else:
                            st.warning("Pipeline ran but no chunks were added.")
                    except Exception as e: # Catch all
                        st.error(f"Pipeline Error: {e}")

elif option == "Chat":
    st.subheader("Chat with Knowledge Base")
    
    # Check if Knowledge Base exists
    if not os.path.exists("data/vectorstore.faiss"):
        st.warning("⚠️ Knowledge Base not found!")
        st.info("Please select **'Ingest Documents'** from the sidebar to populate the Knowledge Base.")
    else:
        st.success("✅ Knowledge Base is Ready!")

        # Initialize chat history
        history_path = os.path.join("data", "chat_history.json")
        if "messages" not in st.session_state:
            st.session_state.messages = []
            if os.path.exists(history_path):
                try:
                    import json
                    with open(history_path, "r") as f:
                        st.session_state.messages = json.load(f)
                except Exception:
                    pass

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}:** {source.get('source', 'Unknown')}")

        # Accept user input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                import re
                import difflib
                
                # Check for greeting to set initial UI status accurately
                is_greeting = False
                normalized = re.sub(r'[^a-zA-Z0-9\s]', '', prompt).strip().lower()
                words = normalized.split()
                g_words = ["hi", "hello", "hey", "greetings", "sup", "howdy", "morning", "afternoon", "evening"]
                if normalized in g_words:
                    is_greeting = True
                elif len(words) <= 5:
                    for w in words:
                        if difflib.get_close_matches(w, g_words, n=1, cutoff=0.7):
                            is_greeting = True
                            break
                            
                initial_status = "Agent: Smalltalk" if is_greeting else "Analyzing documents..."
                
                with st.status(initial_status, expanded=True) as status:
                    try:
                        # Determine model parameters
                        if "Ollama" in llm_choice:
                            provider = "ollama"
                            model_name = "llama3.2:1b"
                        else:
                            provider = "sarvam"
                            model_name = "sarvam-m"
                            
                        # Re-initialize RAGService with selected model
                        # (Normally we'd implement a set_model method, but re-init is fine for testing)
                        current_rag = RAGService(model_provider=provider, model_name=model_name, api_key=sarvam_api_key)
                        
                        # Define status update callback for UI
                        def update_ui_status(msg):
                            status.update(label=msg)
                            st.write(f"⚙️ {msg}")

                        stream_state = {"text": ""}
                        def stream_to_ui(chunk: str):
                            stream_state["text"] += chunk
                            message_placeholder.markdown(stream_state["text"] + "▌")

                        response_data = current_rag.answer_query(prompt, status_callback=update_ui_status, streaming_callback=stream_to_ui)
                        
                        status.update(label="Complete!", state="complete", expanded=False)
                        
                        answer = response_data["answer"]
                        sources = response_data["sources"]
                        
                        message_placeholder.markdown(answer)
                        
                        # Display Trust & Traceability Metrics
                        confidence = response_data.get("confidence", 0.0)
                        verdict = response_data.get("verifier_verdict", "UNKNOWN")
                        is_hallucinated = response_data.get("is_hallucinated", False)
                        
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric(label="Retrieval Confidence", value=f"{confidence * 100:.0f}%")
                        with cols[1]:
                            st.metric(label="Verifier Verdict", value=verdict)
                        with cols[2]:
                            st.metric(label="Hallucinated?", value="⚠️ YES" if is_hallucinated else "✅ NO")
                            
                        if is_hallucinated:
                            st.warning("Warning: The system flagged this answer as potentially drifting from the retrieved context. Please verify.")
                        
                        attributions = response_data.get("attributions", [])
                        if attributions:
                            with st.expander("Sentence-Level Source Mapping"):
                                for attr in attributions:
                                    st.markdown(f"**\"{attr['sentence']}\"**")
                                    st.caption(f"↳ Source: `{attr['source']}` (Confidence: {attr['similarity']*100:.0f}%)")
                                    st.divider()

                        if sources or response_data.get("search_query"):
                            with st.expander("View RAG Traceability (Sources & Optimizations)"):
                                # Issue 6 & 7: Show Query Rewriting and Optimizations
                                opt = response_data.get("optimizations", {})
                                if response_data.get("search_query") and response_data.get("search_query") != prompt:
                                     st.markdown(f"**Original Query:** `{prompt}`")
                                     st.markdown(f"**Optimized Search Query:** `{response_data['search_query']}` *(Autonomous Rewriting Applied)*")
                                     st.divider()
                                
                                st.markdown("**Latency Optimizations Applied:**")
                                st.caption(f"- **Short-Circuited:** `{'Yes' if opt.get('short_circuited') else 'No'}` *(Bypassed Heavy Retrieval)*\n- **Vector Caching:** `Active` *(LRU Cache on SentenceTransformer)*\n- **Dynamic Temp:** `{opt.get('temperature')}` | **Compute Effort:** `{opt.get('reasoning_effort')}`")
                                st.divider()
                                
                                if sources:
                                    for i, source in enumerate(sources):
                                        st.markdown(f"**Source {i+1}:** {source.get('source', 'Unknown')}")
                                        st.caption(f"Snippet: {source.get('text', '')[:200]}...")
                        
                        # User Feedback Loop
                        st.write("Was this helpful?")
                        msg_idx = len(st.session_state.messages)
                        feedback_key = f"feedback_{msg_idx}"
                        
                        if feedback_key not in st.session_state:
                             st.session_state[feedback_key] = None
                             
                        if st.session_state[feedback_key] is None:
                            f_cols = st.columns([1, 1, 10])
                            with f_cols[0]:
                                 if st.button("👍", key=f"up_{msg_idx}"):
                                      st.session_state[feedback_key] = "positive"
                                      # Here we would normally connect to an API endpoint or DB to save feedback
                                      import json, datetime, os
                                      feedback_dir = os.path.join("data", "audit")
                                      os.makedirs(feedback_dir, exist_ok=True)
                                      with open(os.path.join(feedback_dir, "feedback.jsonl"), "a") as f:
                                            f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(), "query": prompt, "rating": "positive"}) + "\n")
                                      st.rerun()
                            with f_cols[1]:
                                 if st.button("👎", key=f"dn_{msg_idx}"):
                                      st.session_state[feedback_key] = "negative"
                                      import json, datetime, os
                                      feedback_dir = os.path.join("data", "audit")
                                      os.makedirs(feedback_dir, exist_ok=True)
                                      with open(os.path.join(feedback_dir, "feedback.jsonl"), "a") as f:
                                            f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(), "query": prompt, "rating": "negative"}) + "\n")
                                      st.rerun()
                        else:
                             st.success(f"Feedback recorded: {st.session_state[feedback_key]}. Thank you!")
                                    
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "search_query": response_data.get("search_query"),
                            "optimizations": response_data.get("optimizations"),
                            "confidence": response_data.get("confidence", 0.0),
                            "verifier_verdict": response_data.get("verifier_verdict", "UNKNOWN"),
                            "is_hallucinated": response_data.get("is_hallucinated", False),
                            "attributions": response_data.get("attributions", [])
                        })
                        import json
                        with open(os.path.join("data", "chat_history.json"), "w") as f:
                            json.dump(st.session_state.messages, f)
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Is Ollama running?"})
