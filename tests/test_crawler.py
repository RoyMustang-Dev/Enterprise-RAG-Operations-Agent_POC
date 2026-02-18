import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ingestion.crawler import crawl_url

def test_crawler():
    test_url = "https://learnnect.com"
    print(f"Crawling {test_url}...")
    
    text = crawl_url(test_url)
    
    if text:
        print("\nSUCCESS: Extracted text:")
        print("-" * 50)
        print(text[:500])  # Print first 500 chars
        print("-" * 50)
    else:
        print("\nFAILED: No text extracted.")

if __name__ == "__main__":
    test_crawler()
