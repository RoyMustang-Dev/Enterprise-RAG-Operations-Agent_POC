import os
import time
from typing import List, Dict

from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_text
from backend.embeddings.embedding_model import EmbeddingModel
from backend.vectorstore.faiss_store import FAISSStore

class IngestionPipeline:
    """
    Orchestrates the entire document ingestion workflow.
    
    Flow:
    1. Loader: Reads file from disk.
    2. Chunker: Splits text into manageable segments.
    3. Embedder: Converts text segments into vector representations.
    4. Vector Store: Saves vectors and metadata for retrieval.
    """
    def __init__(self):
        """Initializes the pipeline with EmbeddingModel and FAISSStore."""
        self.embedder = EmbeddingModel()
        self.vector_store = FAISSStore()

    def run_ingestion(self, file_paths: List[str], metadatas: List[Dict] = None, reset_db: bool = False) -> int:
        """
        Runs the full ingestion pipeline for a list of files or crawled content paths.
        
        Args:
            file_paths (List[str]): List of absolute paths to the files to ingest.
            metadatas (List[Dict], optional): List of metadata dictionaries corresponding to each file.
                                              Defaults to None.
                                              
        Returns:
            int: The total number of text chunks successfully ingested and stored.
        """
        if not file_paths:
            return 0

        total_chunks = 0
        all_chunks = []
        all_metadatas = []
        
        # 0. Reset Knowledge Base if requested
        if reset_db:
            print("Resetting Knowledge Base...")
            self.vector_store.clear()

        print(f"Starting ingestion for {len(file_paths)} files...")
        
        for i, file_path in enumerate(file_paths):
            try:
                print(f"Processing: {file_path}")
                # 1. Load Text
                text = load_document(file_path)
                if not text:
                    print(f"Warning: No text extracted from {file_path}")
                    continue
                    
                # 2. Chunk Text
                chunks = chunk_text(text)
                if not chunks:
                    continue
                
                # Prepare metadata for each chunk
                base_meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                base_meta.update({
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "ingested_at": time.time()
                })
                
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadatas.append(base_meta.copy())
                    
                total_chunks += len(chunks)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        if not all_chunks:
            print("No chunks to ingest.")
            return 0

        # 3. Generate Embeddings (Batch)
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = []
        try:
            embeddings = self.embedder.generate_embeddings(all_chunks)
            if not embeddings:
                # If we have chunks but no embeddings, it's a failure
                if len(all_chunks) > 0:
                     raise ValueError("Embedding model returned no embeddings for chunks.")
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            raise e
        
        # 4. Store in Vector DB
        print("Adding to vector store...")
        try:
            self.vector_store.add_documents(all_chunks, embeddings, all_metadatas)
        except Exception as e:
            print(f"Vector store addition failed: {e}")
            raise e
        
        print(f"Ingestion complete. Added {len(all_chunks)} chunks.")
        return len(all_chunks)

if __name__ == "__main__":
    # Test run
    pipeline = IngestionPipeline()
    # Mock usage: pipeline.run_ingestion(["tests/data/sample.txt"])
