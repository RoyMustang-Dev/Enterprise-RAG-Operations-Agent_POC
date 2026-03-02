import json

curated_models = {
    "Embedding Models": [
        {"model_id": "BAAI/bge-large-en-v1.5", "model_name": "BGE Large EN", "description": "High performance embedding model", "api_calls": 12000, "context_window": 8192},
        {"model_id": "nomic-ai/nomic-embed-text-v1", "model_name": "Nomic Embed Text", "description": "High context embedding model", "api_calls": 8500, "context_window": 8192}
    ],
    "Coder Models": [
        {"model_id": "Qwen/Qwen2.5-Coder-32B-Instruct", "model_name": "Qwen 2.5 Coder 32B", "description": "Advanced coding and reasoning model", "api_calls": 45000, "context_window": 32768},
        {"model_id": "meta-llama/CodeLlama-34b-Instruct-hf", "model_name": "CodeLlama 34B", "description": "Meta's code generation instruction model", "api_calls": 21000, "context_window": 16384}
    ],
    "LLM Chat Models": [
        {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct", "model_name": "Llama 3 8B Instruct", "description": "Meta's generalized 8B reasoning model", "api_calls": 150000, "context_window": 8192},
        {"model_id": "mistralai/Mistral-7B-Instruct-v0.2", "model_name": "Mistral 7B Instruct v0.2", "description": "Fast and efficient generalized chat", "api_calls": 89000, "context_window": 32768}
    ],
    "Agent Orchestrators": [
        {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct", "model_name": "Llama 3 70B Instruct", "description": "Capable of complex agentic task orchestration", "api_calls": 67000, "context_window": 8192}
    ],
    "STT": [
        {"model_id": "openai/whisper-large-v3", "model_name": "Whisper Large V3", "description": "Robust multi-lingual speech to text", "api_calls": 34000, "context_window": 0}
    ],
    "TTS": [
        {"model_id": "elevenlabs/tts", "model_name": "ElevenLabs TTS", "description": "High fidelity text to speech", "api_calls": 92000, "context_window": 0},
        {"model_id": "modi", "model_name": "Modi Voice", "description": "Custom trained voice model", "api_calls": 9527, "context_window": 0}
    ],
    "Live Conversation Models": [
        {"model_id": "kyutai/moshina-v1", "model_name": "Moshi Realtime", "description": "Low latency voice/conversation model", "api_calls": 15000, "context_window": 4096}
    ],
    "Tool Calling Models": [
        {"model_id": "anthropic/claude-3-haiku", "model_name": "Claude 3 Haiku API", "description": "Ultra-fast function calling model", "api_calls": 210000, "context_window": 200000}
    ],
    "Models with Autonomous Agentic Flow capabilities": [
        {"model_id": "auto-gpt-flow/model-v1", "model_name": "AutoGPT Flow", "description": "Iterative flow generation model", "api_calls": 8200, "context_window": 32768}
    ],
    "Prompt Guard Models": [
        {"model_id": "meta-llama/Prompt-Guard-86M", "model_name": "Llama Prompt Guard", "description": "Detects jailbreaks and prompt injections", "api_calls": 56000, "context_window": 4096}
    ],
    "Prompt Generation Models": [
        {"model_id": "midjourney/prompt-generator", "model_name": "MJ Prompt Generator", "description": "Translates basic ideas to prompt strings", "api_calls": 12000, "context_window": 2048}
    ],
    "Deep Reasoning + Thinking Models": [
        {"model_id": "deepseek-ai/DeepSeek-R1", "model_name": "DeepSeek R1", "description": "Uncensored advanced reasoning model", "api_calls": 105000, "context_window": 32768}
    ],
    "Vision Models (OCR + True Vision)": [
        {"model_id": "llava-hf/llava-1.5-13b-hf", "model_name": "LLaVA 1.5 13B", "description": "Large language and vision assistant with strict OCR", "api_calls": 45000, "context_window": 4096}
    ],
    "Summarizer Models": [
        {"model_id": "facebook/bart-large-cnn", "model_name": "BART Large CNN", "description": "Dedicated summarization model", "api_calls": 78000, "context_window": 1024}
    ]
}

with open("filtered_llm_models2.json", "w", encoding="utf-8") as f:
    json.dump(curated_models, f, indent=4)

print("Curated comprehensive JSON created successfully.")
