import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ingestion.loader import load_document
from backend.ingestion.crawler import crawl_url
from backend.ingestion.chunker import chunk_text

def test_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    print(f"\nProcessing File: {file_path}")
    try:
        text = load_document(file_path)
        print(f"Extracted {len(text)} characters.")
        if text:
            print("-" * 50)
            print(text[:500] + "..." if len(text) > 500 else text)
            print("-" * 50)
            
            chunks = chunk_text(text)
            print(f"Generated {len(chunks)} chunks.")
            if chunks:
                print(f"First chunk preview: {chunks[0][:100]}...")
        else:
            print("Warning: No text extracted.")
            
    except Exception as e:
        print(f"Error processing file: {e}")

def test_url(url):
    print(f"\nCrawling URL: {url}")
    try:
        text = crawl_url(url)
        print(f"Extracted {len(text)} characters.")
        if text:
            print("-" * 50)
            print("Content saved to 'data/crawled_docs/...'")
            print(text[:500] + "..." if len(text) > 500 else text)
            print("-" * 50)
        else:
            print("Failed to extract text.")
            
    except Exception as e:
        print(f"Error crawling URL: {e}")

def main():
    while True:
        print("\n--- Manual Test Menu ---")
        print("1. Test Document File (PDF, DOCX, TXT)")
        print("2. Test URL Crawler")
        print("3. Test End-to-End Ingestion (Embeddings & Storage)")
        print("q. Exit")
        
        choice = input("Enter choice: ").strip()
        
        if choice == "1":
            path = input("Enter absolute file path: ").strip().strip('"')
            test_file(path)
        elif choice == "2":
            url = input("Enter URL (e.g., https://example.com): ").strip()
            test_url(url)
        elif choice == "3":
             print("\n[INFO] Testing End-to-End Ingestion...")
             from backend.ingestion.pipeline import IngestionPipeline
             
             # Create dummy test file
             test_file = "tests/manual_test_doc.txt"
             with open(test_file, "w") as f:
                 f.write("This is a test document for the RAG system. It should be chunked and embedded.")
                 
             pipeline = IngestionPipeline()
             try:
                 count = pipeline.run_ingestion([test_file], [{"source": "manual_test"}])
                 print(f"\n[SUCCESS] Ingested {count} chunks.")
                 
                 # Verify storage
                 print("[INFO] Verifying Vector Store...")
                 query_vec = pipeline.embedder.generate_embedding("test document")
                 results = pipeline.vector_store.search(query_vec, k=1)
                 if results:
                     print(f"[SUCCESS] Retrieved: {results[0]['text']}")
                     print(f"[SUCCESS] Score: {results[0]['score']}")
                 else:
                     print("[ERROR] Retrieval failed.")
             except Exception as e:
                 print(f"[ERROR] Ingestion failed: {e}")
             finally:
                 if os.path.exists(test_file):
                     os.remove(test_file)

        elif choice.lower() == "q":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
