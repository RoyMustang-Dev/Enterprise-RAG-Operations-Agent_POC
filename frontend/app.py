import os
import sys
import re
import asyncio
import nest_asyncio
import streamlit as st

nest_asyncio.apply()

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.crawler import crawl_url

# Initialize pipeline (outside the main interaction loop to avoid reload overhead if possible, 
# but Streamlit re-runs scripts top-to-bottom. Ideally cache this).
@st.cache_resource
def get_pipeline():
    return IngestionPipeline()

pipeline = get_pipeline()

st.set_page_config(page_title="Enterprise RAG Agent", layout="wide")

st.title("Enterprise RAG Operations Agent")

st.sidebar.header("Operations")
option = st.sidebar.selectbox("Choose Action", ["Chat", "Ingest Documents", "System Health"])

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
    
    st.markdown("Upload files, enter a URL, or do both. The system will process all inputs into the knowledge base.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Upload Files")
        uploaded_files = st.file_uploader("References (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    with col2:
        st.markdown("### 2. Crawl URL")
        url_input = st.text_input("Target URL (e.g., https://learnnect.com)")

    if st.button("Start Ingestion", type="primary"):
        if not uploaded_files and not url_input:
            st.warning("Please provide at least one file or a URL.")
        else:
            with st.status("Processing Ingestion...", expanded=True) as status:
                
                paths_to_process = []
                metadatas_to_process = []
                
                # Process Files
                if uploaded_files:
                    st.write(f"📂 Saving {len(uploaded_files)} file(s)...")
                    os.makedirs("data", exist_ok=True)
                    
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join("data", uploaded_file.name)
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
                            # Crawler saves to data/crawled_docs/<domain>/content.txt
                            # We need to find the latest folder or specific folder
                            # For simplicity, we assume the crawler behavior and manually construct path 
                            # OR better: have crawler return the path.
                            # Re-parsing URL to guess path (Fragile but works for now)
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
                        num_chunks = pipeline.run_ingestion(paths_to_process, metadatas_to_process)
                        if num_chunks:
                            st.success(f"✅ Successfully ingested {num_chunks} chunks into the Knowledge Base!")
                        else:
                            st.warning("Pipeline ran but no chunks were added.")
                    except EventHandler as e: # Catch all
                        st.error(f"Pipeline Error: {e}")
                    
                status.update(label="Ingestion Complete!", state="complete", expanded=False)

elif option == "Chat":
    st.subheader("Chat with Knowledge Base")
    user_input = st.text_input("Ask a question:")
    if user_input:
        st.write("You asked:", user_input)
        # TODO: Implement RAG logic
