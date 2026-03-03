"""
Few-shot generation specs for each stage.
These are prompts used by the generator script to create real-world examples.
"""

FEW_SHOT_SPECS = {
    "intent_classifier": {
        "model": "llama-3.1-8b-instant",
        "examples": 150,
        "prompt": (
            "Generate {n} real-world user inputs for intent classification. "
            "Return JSON array of objects with keys: user, assistant_output. "
            "assistant_output must be strict JSON: {{\"intent\":\"...\",\"confidence\":0.0-1.0}}. "
            "Allowed intents: greeting, smalltalk, out_of_scope, rag_question, code_request, analytics_request, multimodal_audio, other."
        ),
    },
    "metadata_extractor": {
        "model": "llama-3.1-8b-instant",
        "examples": 150,
        "prompt": (
            "Generate {n} real-world user queries that include metadata constraints "
            "(names, dates, departments, doc types). Return JSON array of objects with "
            "user and assistant_output. assistant_output must follow the metadata_extractor schema."
        ),
    },
    "query_rewriter": {
        "model": "openai/gpt-oss-120b",
        "examples": 150,
        "prompt": (
            "Generate {n} real-world user prompts and the ideal JSON output for a query rewriter. "
            "Return JSON array of objects with user and assistant_output."
        ),
    },
    "rag_synthesis": {
        "model": "llama-3.3-70b-versatile",
        "examples": 150,
        "prompt": (
            "Generate {n} real-world RAG questions and ideal assistant JSON outputs "
            "with answer and confidence. Assume answers are grounded."
        ),
    },
    "coder_agent": {
        "model": "qwen/qwen3-32b",
        "generator_model": "llama-3.1-8b-instant",
        "examples": 150,
        "prompt": (
            "Generate {n} real-world code/analytics requests and ideal assistant outputs "
            "that include explanation + code in markdown. Return JSON array."
        ),
    },
    "hallucination_verifier": {
        "model": "sarvam-m",
        "generator_model": "llama-3.1-8b-instant",
        "examples": 150,
        "prompt": (
            "Generate {n} real-world answers with context and expected verifier JSON "
            "({{\"hallucinated\":...,\"unsupported_claims\":[...]}}). Return JSON array."
        ),
    },
    "reward_scorer": {
        "model": "llama-3.1-8b-instant",
        "examples": 150,
        "prompt": (
            "Generate {n} real-world response grading examples with expected JSON "
            "({{\"score\":...,\"rationale\":\"...\"}}). Return JSON array."
        ),
    },
    "security_guard": {
        "model": "llama-prompt-guard-2-86m",
        "generator_model": "llama-3.1-8b-instant",
        "examples": 150,
        "prompt": (
            "Generate {n} real-world prompts including benign and malicious cases. "
            "Return JSON array of objects with user and assistant_output (strict JSON)."
        ),
    },
}
