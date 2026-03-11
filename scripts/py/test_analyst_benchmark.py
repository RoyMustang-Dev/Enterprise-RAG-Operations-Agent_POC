import httpx
import json
import uuid

API_URL = "http://localhost:8000/api/v1/business_analyst/chat"
SESSION_ID = str(uuid.uuid4())

def run_benchmark():
    print(f"Executing End-to-End Analytics Benchmark on Session: {SESSION_ID}")
    
    file_path = "data/uploads/marketing_ecommerce_benchmark.csv"
    
    # We ask a highly complex question that mandates Python execution over the 8000 rows.
    query = "Calculate our total global Revenue and total Marketing Spend. Then, tell me if our Return on Ad Spend (ROAS) is positive."
    
    # Send the massive CSV via Multipart Form
    with open(file_path, "rb") as f:
        files = {"files": ("marketing_ecommerce_benchmark.csv", f, "text/csv")}
        data = {
            "query": query,
            "session_id": SESSION_ID
        }
        
        headers = {"x-tenant-id": "default"}
        
        # We allow a much longer timeout because of the dataset density and multi-turn ReAct loop
        with httpx.Client(timeout=180.0) as client:
            response = client.post(API_URL, data=data, files=files, headers=headers)
            
            if response.status_code == 200:
                print("\n--- ANALYTICS DASHBOARD RESPONSE (200 OK) ---")
                print(json.dumps(response.json(), indent=2))
            else:
                print("\n--- API FAILURE ---")
                print(f"Error: {response.status_code} {response.reason_phrase} for url: {response.url}")
                print(f"Response Body: {response.text}")

if __name__ == "__main__":
    run_benchmark()
