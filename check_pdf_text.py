from backend.ingestion.loader import load_document
import os

pdf_path = os.path.join("data", "uploaded_docs", "Updated_Resume_DS.pdf")
try:
    text = load_document(pdf_path)
    print(f"--- START TEXT ({len(text)} chars) ---")
    print(text)
    print("--- END TEXT ---")
    
    if "Akobot" in text or "AKOBOT" in text:
        print("\nSUCCESS: 'Akobot' found in text.")
    else:
        print("\nFAILURE: 'Akobot' NOT found in text.")
except Exception as e:
    print(f"Error: {e}")
