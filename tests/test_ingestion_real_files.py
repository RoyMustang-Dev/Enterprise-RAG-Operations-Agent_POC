import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_text

TEST_DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../test-docs'))

def test_ingestion():
    print(f"Scanning {TEST_DOCS_DIR}...")
    
    if not os.path.exists(TEST_DOCS_DIR):
        print(f"Directory {TEST_DOCS_DIR} not found!")
        return

    files = [f for f in os.listdir(TEST_DOCS_DIR) if os.path.isfile(os.path.join(TEST_DOCS_DIR, f))]
    
    pdf_count = 0
    docx_count = 0
    txt_count = 0
    
    for f in files:
        file_path = os.path.join(TEST_DOCS_DIR, f)
        ext = os.path.splitext(f)[1].lower()
        
        print(f"\nProcessing {f}...")
        
        try:
            text = load_document(file_path)
            print(f"  - Extracted {len(text)} characters.")
            
            if not text:
                print("  - WARNING: No text extracted.")
                continue
                
            chunks = chunk_text(text)
            print(f"  - Generated {len(chunks)} chunks.")
            
            if chunks:
                print(f"  - First chunk preview: {chunks[0][:100]}...")
            
            if ext == ".pdf":
                pdf_count += 1
            elif ext == ".docx":
                docx_count += 1
            elif ext == ".txt":
                txt_count += 1
                
        except Exception as e:
            print(f"  - FAILED: {e}")

    print("\nSummary:")
    print(f"PDFs: {pdf_count}/3")
    print(f"DOCXs: {docx_count}/3")
    print(f"TXTs: {txt_count}/2")
    
    if pdf_count >= 3 and docx_count >= 3 and txt_count >= 2:
        print("\nSUCCESS: All required file types tested.")
    else:
        print("\nWARNING: Not all required file counts met (3 PDF, 3 DOCX, 2 TXT).")

if __name__ == "__main__":
    test_ingestion()
