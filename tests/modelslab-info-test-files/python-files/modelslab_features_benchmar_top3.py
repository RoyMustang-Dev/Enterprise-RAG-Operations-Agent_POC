import requests
import time
import json
import csv
from typing import List, Dict

# ==========================
# CONFIG
# ==========================

API_KEY = "wQrSd1q6vvHXovWpiLALJlrZFJvcJzN4iUVex283wYgiOcGHh0CB6DurTwq6"
BASE_AGENT_URL = "https://modelslab.com/api/agents/v1/models"
LLM_INFERENCE_URL = "https://modelslab.com/api/v7/llm/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

FEATURE_CATEGORIES = {
    "embedding": "Embedding Models",
    "coder": "Coder Models",
    "chat_models": "LLM Chat Models",
    "orchestrator": "Agent Orchestrators",
    "stt": "STT",
    "tts": "TTS",
    "live_convo": "Live Conversation Models",
    "tool_calling": "Tool Calling Models",
    "agentic": "Models with Autonomous Agentic Flow capabilities",
    "prompt_guard": "Prompt Guard Models",
    "prompt_gen": "Prompt Generation Models",
    "deep_reasoning": "Deep Reasoning + Thinking Models",
    "vision": "Vision Models (OCR + True Vision)",
    "summarizer": "Summarizer Models"
}

SAFE_TEST_PROMPT = "Summarize: The meeting discussed revenue growth and action items."


# ==========================
# MODEL FETCHER
# ==========================

def fetch_models_by_feature(feature: str) -> List[Dict]:
    try:
        with open("filtered_llm_models2.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(feature, [])
    except Exception as e:
        print(f"Failed to fetch local model map: {e}")
        return []


# ==========================
# METADATA EXTRACTION
# ==========================

def extract_model_metadata(model: Dict) -> Dict:
    return {
        "model_id": model.get("model_id"),
        "provider": model.get("provider", "Unknown"),
        "context_window": model.get("context_window", 0),
        "supports_function_calling": model.get("supports_function_calling", False),
        "api_calls": int(model.get("api_calls", 0)),
        "created_at": model.get("created_at", "Unknown")
    }


# ==========================
# SELECT TOP 3
# ==========================

def select_top_3(models: List[Dict]) -> List[Dict]:

    sorted_models = sorted(
        models,
        key=lambda x: (x["api_calls"], x["context_window"] or 0),
        reverse=True
    )

    return sorted_models[:3]


# ==========================
# BENCHMARK FUNCTION
# ==========================

def benchmark_model(model_id: str, category: str) -> Dict:
    
    # Only standard LLMs should be subjected to the Text Completion test prompt
    if category not in ["LLM Chat Models", "Coder Models", "Agent Orchestrators", "Tool Calling Models", "Deep Reasoning + Thinking Models", "Summarizer Models"]:
        return {
            "model_id": model_id,
            "status": "skipped (multimodal model requires diverse testing env)",
            "latency": 0.0,
            "tokens_generated": 0,
            "tokens_per_sec": 0
        }

    payload = {
        "key": API_KEY,
        "model_id": model_id,
        "messages": [
            {"role": "system", "content": "You are a summarizer."},
            {"role": "user", "content": SAFE_TEST_PROMPT}
        ],
        "max_tokens": 200,
        "temperature": 0.2
    }

    start = time.time()
    response = requests.post(LLM_INFERENCE_URL, json=payload)
    end = time.time()

    latency = round(end - start, 3)

    if response.status_code != 200:
        return {
            "model_id": model_id,
            "status": f"failed: {response.status_code}",
            "latency": latency,
            "tokens_generated": 0,
            "tokens_per_sec": 0
        }

    try:
        output_data = response.json()
        if "choices" not in output_data or output_data.get("status") == "error":
            return {
                "model_id": model_id,
                "status": f"failed: {output_data.get('message', 'No valid choices array')}",
                "latency": latency,
                "tokens_generated": 0,
                "tokens_per_sec": 0
            }
        output = output_data["choices"][0]["message"]["content"]
    except Exception as e:
        return {
            "model_id": model_id,
            "status": f"failed: parse error {str(e)}",
            "latency": latency,
            "tokens_generated": 0,
            "tokens_per_sec": 0
        }

    token_count = len(output.split())

    return {
        "model_id": model_id,
        "status": "success",
        "latency": latency,
        "tokens_generated": token_count,
        "tokens_per_sec": round(token_count / latency, 2) if latency > 0 else 0
    }


# ==========================
# CSV EXPORT
# ==========================

def export_csv(results: List[Dict], filename: str):
    if not results:
        print("No successful benchmarks extracted to export.")
        return
        
    keys = results[0].keys()

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# ==========================
# MAIN PIPELINE
# ==========================

def main():

    final_results = []

    for category, feature in FEATURE_CATEGORIES.items():

        print(f"\nFetching models for: {category}")

        raw_models = fetch_models_by_feature(feature)

        extracted = [extract_model_metadata(m) for m in raw_models]

        top_models = select_top_3(extracted)

        print(f"Top 3 selected for {category}:")
        for m in top_models:
            print("-", m["model_id"])

        for model in top_models:
            print(f"Benchmarking {model['model_id']}...")
            benchmark_result = benchmark_model(model["model_id"], feature)
            benchmark_result["category"] = feature
            benchmark_result["context_window"] = model["context_window"]
            benchmark_result["supports_function_calling"] = model["supports_function_calling"]
            final_results.append(benchmark_result)

    export_csv(final_results, "modelslab_top3_benchmark.csv")

    print("\nBenchmark complete. Results saved to modelslab_top3_benchmark.csv")


if __name__ == "__main__":
    main()