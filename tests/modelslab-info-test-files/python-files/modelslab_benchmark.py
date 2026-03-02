import requests
import time
import json
import csv
import statistics
from typing import List, Dict

API_KEY = "wQrSd1q6vvHXovWpiLALJlrZFJvcJzN4iUVex283wYgiOcGHh0CB6DurTwq6"
BASE_URL = "https://modelslab.com/api/v7/llm/chat/completions"


# ===============================
# SAMPLE TRANSCRIPT (MeeTify test)
# ===============================

SAMPLE_TRANSCRIPT = """
17:00 Aditya: Welcome everyone.
17:01 Tabish: Thanks for joining.
17:03 Aditya: Today we discuss Q1 revenue.
17:05 Abdus: We saw 15% growth.
17:07 Abhisjek: Marketing improved lead generation.
17:10 Aditya: Action item - increase ad spend.
"""


# ===============================
# MODEL TEST FUNCTION
# ===============================

def test_model(model_id: str) -> Dict:

    payload = {
        "key": API_KEY,
        "model_id": model_id,
        "messages": [
            {"role": "system", "content": "You are a meeting summarization AI."},
            {"role": "user", "content": f"Convert this into structured minute-to-minute meeting notes:\n{SAMPLE_TRANSCRIPT}"}
        ],
        "max_tokens": 800,
        "temperature": 0.3
    }

    start = time.time()
    response = requests.post(BASE_URL, json=payload)
    end = time.time()

    latency = end - start

    data = response.json()

    if response.status_code != 200 or data.get("status") == "error" or "choices" not in data:
        return {
            "model": model_id,
            "error": data.get("message", response.text)
        }

    output_text = data["choices"][0]["message"]["content"]

    token_count = len(output_text.split())
    throughput = token_count / latency if latency > 0 else 0

    structured_score = 1 if "17:00" in output_text else 0
    speaker_score = 1 if "Aditya" in output_text else 0

    return {
        "model": model_id,
        "latency_sec": round(latency, 3),
        "tokens_generated": token_count,
        "tokens_per_sec": round(throughput, 2),
        "structured_score": structured_score,
        "speaker_score": speaker_score,
        "output_preview": output_text[:300]
    }


# ===============================
# MAX TOKEN DETECTION
# ===============================

def detect_max_tokens(model_id: str):

    test_tokens = 512

    while test_tokens <= 8192:

        payload = {
            "key": API_KEY,
            "model_id": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": test_tokens
        }

        response = requests.post(BASE_URL, json=payload)
        
        try:
            data = response.json()
            if response.status_code != 200 or data.get("status") == "error":
                return test_tokens // 2
        except Exception:
            if response.status_code != 200:
                return test_tokens // 2

        test_tokens *= 2

    return test_tokens


# ===============================
# EXPORT CSV
# ===============================

def export_csv(results: List[Dict]):

    keys = results[0].keys()

    with open("model_benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# ===============================
# MARKDOWN TABLE GENERATOR
# ===============================

def generate_markdown_table(results: List[Dict]):

    headers = results[0].keys()

    table = "| " + " | ".join(headers) + " |\n"
    table += "|---" * len(headers) + "|\n"

    for row in results:
        table += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"

    with open("model_comparison.md", "w", encoding="utf-8") as f:
        f.write(table)


# ===============================
# MAIN RUNNER
# ===============================

def main():

    # Replace this with your actual discovered models list
    models_to_test = [
        "meta-llama-3-8B-instruct",
        "mistral-7b-instruct"
    ]

    results = []

    for model in models_to_test:
        print(f"Testing {model}...")
        result = test_model(model)

        if "error" not in result:
            result["max_token_estimate"] = detect_max_tokens(model)

        results.append(result)

    export_csv(results)
    generate_markdown_table(results)

    print("\nBenchmark Complete. Files generated:")
    print("- model_benchmark_results.csv")
    print("- model_comparison.md")


if __name__ == "__main__":
    main()