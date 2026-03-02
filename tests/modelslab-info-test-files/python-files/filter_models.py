import json
import os
from typing import Dict, List, Any

TARGET_CATEGORIES = {
    "Embedding Models": ["embed", "bge", "nomic", "e5", "instructor"],
    "Coder Models": ["code", "coder", "starcoder", "phind", "wizardcoder", "qwen2.5-coder"],
    "LLM Chat Models": ["chat", "instruct", "llama", "mistral", "mixtral", "gemma", "qwen", "glm", "alpaca", "vicuna"],
    "Agent Orchestrators": ["agent", "orchestrat", "function", "tool"],
    "STT": ["whisper", "stt", "speech-to-text", "speech_to_text"],
    "TTS": ["tts", "voice", "text-to-speech", "text_to_speech", "eleven", "bark"],
    "Live Conversation Models": ["live", "realtime", "conversation"],
    "Tool Calling Models": ["tool", "function", "fc", "calling"],
    "Models with Autonomous Agentic Flow capabilities": ["auto", "agent", "flow", "autogpt", "babyagi"],
    "Prompt Guard Models": ["guard", "moderation", "safety", "toxic", "content"],
    "Prompt Generation Models": ["prompt", "generator", "midjourney", "diffusion"],
    "Deep Reasoning + Thinking Models": ["reason", "think", "math", "logic", "r1", "o1"],
    "Vision Models (OCR + True Vision)": ["vision", "vl", "ocr", "llava", "pixtral", "qwen-vl", "image_to_text", "image-to-text"],
    "Summarizer Models": ["summar", "bart", "pegasus"]
}

def load_data():
    try:
        with open("modelslab_models_output.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Error loading models output: {e}")
        return []

def filter_models(data: List[Dict[str, Any]]):
    categorized_models = {cat: [] for cat in TARGET_CATEGORIES.keys()}
    
    for item in data:
        # Extract ID and Name, normalize to lowercase
        model_id = item.get("model_id", "").lower()
        model_name = item.get("model_name", "").lower()
        description = item.get("description", "").lower() if item.get("description") else ""
        api_calls = item.get("api_calls", "").lower() if item.get("api_calls") else ""
        search_text = f"{model_id} {model_name} {description} {api_calls}"
        
        # We classify this item against categories
        matched_any = False
        for category, keywords in TARGET_CATEGORIES.items():
            if any(kw in search_text for kw in keywords) and int(api_calls) > 3500:
                # Clean up the object to only vital info
                clean_item = {
                    "model_id": item.get("model_id"),
                    "model_name": item.get("model_name"),
                    "description": item.get("description", "No description provided."),
                    "api_calls": item.get("api_calls", "0")
                }
                categorized_models[category].append(clean_item)
                matched_any = True
                
    # Remove empty categories
    final_output = {k: v for k, v in categorized_models.items() if len(v) > 0}
    return final_output

def main():
    data = load_data()
    if isinstance(data, dict):
        if "models" in data:
            data = data["models"]
        else:
            # Maybe the top-level is the dict, though my check showed it was a list
            data = [data]
            
    filtered = filter_models(data)
    
    out_file = "filtered_llm_models2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=4)
        print(f"Successfully wrote filtered models to {out_file}")

if __name__ == "__main__":
    main()
