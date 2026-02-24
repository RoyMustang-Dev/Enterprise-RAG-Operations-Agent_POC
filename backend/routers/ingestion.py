"""
Knowledge Ingestion API Router

Provides modular FastAPI endpoints for ingesting raw data into the Vector Database.
Supports multiple ingestion modalities including direct document uploads (PDF, DOCX) 
and asynchronous external web crawling.
"""
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import logging
import asyncio
from backend.ingestion.crawler_service import CrawlerService
from backend.ingestion.pipeline import IngestionPipeline
from urllib.parse import urlparse
from enum import Enum
import uuid
from fastapi import BackgroundTasks

# Memory dictionary to track background job statuses globally
ingestion_jobs = {}

class IngestionMode(str, Enum):
    append = "append"
    start_fresh = "start_fresh"


# Initialize the router to namespace ingestion-specific REST operations
router = APIRouter(
    prefix="/api/v1/ingest",
    tags=["Knowledge Ingestion"],
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dependency Management
# -----------------------------------------------------------------------------
def get_pipeline() -> IngestionPipeline:
    """
    Factory function to instantiate the IngestionPipeline.
    Separated to support future dependency injection or mocking during tests.
    
    Returns:
        IngestionPipeline: Initialized instance handling chunking and vector processing.
    """
    return IngestionPipeline()

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
def _run_files_background(paths_to_process, metadatas_to_process, reset_db, job_id):
    """Executes the file embedding pipeline fully detached from the main HTTP thread."""
    try:
        pipeline = get_pipeline()
        pipeline.run_ingestion(
            file_paths=paths_to_process, 
            metadatas=metadatas_to_process, 
            reset_db=reset_db, 
            job_tracker=ingestion_jobs[job_id]
        )
    except Exception as e:
        logger.error(f"Background Ingestion Error: {str(e)}")
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"] = str(e)


@router.post("/files")
async def ingest_document(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    mode: IngestionMode = Form(IngestionMode.append, description="Options: 'append' (keep existing data) or 'start_fresh' (wipe DB)")
):
    """
    **Manual File Upload Ingestion**
    
    Accepts an array of multipart document files (e.g., .pdf, .docx).
    Saves them to temporary static storage, and executes the vector extraction pipeline.
    """
    try:
        filenames = [f.filename for f in files]
        logger.info(f"API Ingesting documents: {filenames} (Mode: {mode})")
        
        # Determine if the vector database should clear its existing contents before writing
        reset_db = True if mode == "start_fresh" else False
        pipeline = get_pipeline()
        
        # Track local filepaths and generated metadata for the vector chunker
        paths_to_process = []
        metadatas_to_process = []
        
        # Ensure the safe temporary upload directory exists
        os.makedirs("data/uploaded_docs", exist_ok=True)
        
        # Process multi-part uploads sequentially
        for uploaded_file in files:
            temp_path = os.path.join("data/uploaded_docs", uploaded_file.filename)
            content = await uploaded_file.read()
            
            # Persist byte chunks safely to disk
            with open(temp_path, "wb") as f:
                f.write(content)
                
            paths_to_process.append(temp_path)
            # Attach structural metadata dict determining document source types
            metadatas_to_process.append({"type": "file", "original_name": uploaded_file.filename})
            
        # Detach execution so frontend is not blocked
        job_id = str(uuid.uuid4())
        ingestion_jobs[job_id] = {"status": "processing", "chunks_added": 0, "total_chunks": 0, "type": "files_upload"}
        background_tasks.add_task(_run_files_background, paths_to_process, metadatas_to_process, reset_db, job_id)
        
        return {
            "status": "accepted", 
            "message": f"Successfully queued {len(filenames)} files for background extraction.",
            "job_id": job_id,
            "mode_applied": mode
        }
    except Exception as e:
        logger.error(f"Ingestion Queue Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize document upload queue.")

@router.get("/progress/{job_id}")
async def check_progress(job_id: str):
    """
    **Streaming Context Poller**
    
    Provides highly granular, real-time analytics to the UI determining the percentage 
    of chunks converted to mathematical vectors so far.
    """
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Ingestion Job ID not found.")
        
    job = ingestion_jobs[job_id]
    if job["status"] == "failed":
         raise HTTPException(status_code=500, detail=f"Job failed: {job.get('error', 'Unknown Error')}")
         
    return job

@router.get("/status")
async def get_ingestion_status():
    """
    **Knowledge Base Status Check**
    
    Returns the distinct documents inside the DB and the number of total active vectors.
    """
    try:
        pipeline = get_pipeline()
        docs = pipeline.vector_store.get_all_documents()
        total_vectors = pipeline.vector_store.ntotal
        return {
            "status": "success",
            "total_documents": len(docs),
            "documents": docs,
            "total_vectors": total_vectors
        }
    except Exception as e:
        logger.error(f"Status Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve database status.")

def _run_crawler_background(url, max_depth, save_folder, mode, job_id):
    """Executes the Playwright headless web scraping and subsequent vector encoding fully in the background."""
    print(f"\n[BACKGROUND THREAD] _run_crawler_background initiated for job {job_id} / depth {max_depth}")
    try:
        import sys
        import asyncio
        if sys.platform == "win32":
            print("[BACKGROUND THREAD] Applying WindowsProactorEventLoopPolicy...")
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        print("[BACKGROUND THREAD] Creating new isolated asyncio loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        print(f"[BACKGROUND THREAD] Instantiating CrawlerService for {url}...")
        crawler = CrawlerService()
        
        reset_db = True if mode == "start_fresh" else False
        pipeline = get_pipeline()
        is_first_batch = [True]
        
        async def process_live_batch(batch_items):
            print(f"[BACKGROUND THREAD] Streaming {len(batch_items)} pages natively to Vector Engine...")
            ingestion_jobs[job_id]["status"] = "crawling_and_extracting"
            
            paths = []
            metas = []
            import os
            import hashlib
            
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
                
            for item in batch_items:
                # item: (session_id, current_url, title, clean_content, depth, status)
                current_url = item[1]
                title = item[2]
                content = item[3]
                
                safe_name = hashlib.md5(current_url.encode()).hexdigest() + ".txt"
                path = os.path.join(save_folder, safe_name)
                    
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"== {title} ==\n{content}")
                    
                paths.append(path)
                metas.append({"type": "url", "source_url": current_url})
                
            reset = reset_db if is_first_batch[0] else False
            is_first_batch[0] = False
            
            # Sub-thread the synchronous heavy embedding mathematics safely away from Playwright's core event loops
            def ingest_sync():
                pipeline.run_ingestion(paths, metadatas=metas, reset_db=reset, job_tracker=ingestion_jobs[job_id], mark_completed=False)
                
            await asyncio.to_thread(ingest_sync)
            
        print("[BACKGROUND THREAD] Blocking until Playwright crawler resolves completion...")
        result = loop.run_until_complete(
            crawler.crawl_url(
                url=url, 
                save_folder=save_folder,
                simulate=False,
                recursive=(max_depth > 1),
                max_depth=max_depth,
                on_batch_extracted=process_live_batch
            )
        )
        print(f"[BACKGROUND THREAD] Playwright crawler successfully resolved: {result.get('status')}")
        loop.close()
        
        if result.get("saved_files") or not is_first_batch[0]:
            print("[BACKGROUND THREAD] Vector Engine streaming execution formally complete!")
            ingestion_jobs[job_id]["status"] = "completed"
        else:
             print("[BACKGROUND THREAD] No unstructured text output physically extracted.")
             ingestion_jobs[job_id]["status"] = "failed"
             ingestion_jobs[job_id]["error"] = "No unstructured text output generated by Playwright crawler."

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        logger.error(f"Crawler Background Error:\n{trace}")
        print(f"[BACKGROUND THREAD - FATAL] Target background engine fully crashed natively: {e}")
        print(trace)
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"] = str(e)


@router.post("/crawler")
async def trigger_crawler(
    background_tasks: BackgroundTasks,
    url: str = Form(..., description="The target URL to crawl (e.g., https://example.com)"),
    max_depth: int = Form(1, description="Depth of recursion (1 = single parent page, 2 = parent + direct child links). It can go up to 4."),
    mode: IngestionMode = Form(IngestionMode.append, description="Ingestion mode: 'append' adds to KB, 'start_fresh' clears the database first.")
):
    """
    **External Website Crawler Trigger**
    
    Spawns an advanced headless Playwright instance against the provided URL.
    Saves extracted structured text into the static 'crawled_docs' directory, 
    and then automatically pipes them into the semantic ingestion pipeline.
    """
    try:
        logger.info(f"API Triggering background crawler mapping on {url} (Depth: {max_depth})")
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path.strip("/")
        folder_name = f"{domain}_{path}".replace("/", "_") if path else domain
        save_folder = os.path.join("data", "crawled_docs", folder_name)
        
        # Instantiate background job
        job_id = str(uuid.uuid4())
        ingestion_jobs[job_id] = {"status": "initializing_crawler", "chunks_added": 0, "total_chunks": 0, "type": "web_crawl"}
        
        # Dispatch to queue loop
        background_tasks.add_task(_run_crawler_background, url, max_depth, save_folder, mode, job_id)

        # Drop the HTTP connection so Streamlit receives a response inside 0.1ms
        return {
            "status": "accepted", 
            "message": f"Successfully queued crawler extraction for {url}.",
            "job_id": job_id,
            "mode_applied": mode
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Crawler Queue Error:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanly dispatch crawler task. Reason: {str(e)}")
