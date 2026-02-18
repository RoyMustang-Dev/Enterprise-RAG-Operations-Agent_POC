import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Optional

class FAISSStore:
    """
    Wrapper around FAISS index for vector storage and retrieval.
    Includes metadata storage since FAISS only stores vectors.
    """
    def __init__(self, dimension: int = 384, index_path: str = "data/vectorstore.faiss", metadata_path: str = "data/metadata.pkl"):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: List[Dict] = []
        
        # Load existing index if available
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load()

    def add_documents(self, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """
        Adds documents and their embeddings to the store.
        """
        if not chunks or not embeddings:
            return

        # Convert to numpy array
        vectors = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(vectors)
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            meta = metadatas[i] if i < len(metadatas) else {}
            meta['text'] = chunk  # Store text content in metadata for retrieval
            self.metadata.append(meta)
            
        print(f"Added {len(chunks)} documents to vector store. Total: {self.index.ntotal}")
        self.save()

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Searches the index for the top-k most similar vectors.
        """
        if self.index.ntotal == 0:
            return []

        # Convert query to numpy
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(distances[0][i])
                results.append(result)
                
        return results

    def save(self):
        """Persists the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Vector store saved to {self.index_path}")

    def load(self):
        """Loads the FAISS index and metadata from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"Loaded vector store with {self.index.ntotal} documents")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            # Reset if load fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
