# Centralized Prompt Library Architecture (Groq)

Based on your audio instructions, you are absolutely correct. Having system prompts hardcoded across 12 different python files makes iteration and tuning extremely difficult. We must decouple them into a centralized control center. 

Here is the proposed architectural flow and strategy to build out the `app/prompt_engine/groq_prompts` library, including your request for massive model-tuned Few-Shot examples.

---

## 1. Directory Structure Proposal
We will create a new directory inside `app/prompt_engine/` specifically dedicated to managing the Groq ecosystem prompts (and later, separate ones for `gemini_prompts`, etc.).

```text
app/prompt_engine/
├── groq_prompts/
│   ├── __init__.py
│   ├── config.py                 # The "Super-System Prompt" Persona Injector
│   ├── base_prompts.py           # The 12 core system prompts (stripped of examples)
│   └── few_shots/                # Dedicated library for 100-200 examples per node
│       ├── intent_llama_8b.json
│       ├── synthesis_llama_70b.json
│       ├── coder_qwen_32b.json
│       └── verifier_llama_8b.json
```

---

## 2. The Execution Flow (How it works dynamically)

1. **The Modular Base Prompts (`base_prompts.py`)**: 
   - We extract the 12 hardcoded strings from your codebase (like the Intent Classifier or RAG Synthesis prompts) and place them here as cleanly formatted Python dictionary templates.
   - Example: `GROQ_PROMPTS["intent_classifier"] = "You are a high precision intent classifier..."`

2. **The "Super-System" Injector (`config.py`)**:
   - This script acts as the global controller you requested.
   - It intercepts the Base Prompt, automatically fetches the SQLite Global Bootstrapper Persona (Bot Name, Brand Tone, ReAct rules), and gracefully prepends it *above* the Base Prompt. 

3. **Node Import (The 12 Stages)**:
   - Instead of writing `self.system_prompt = "You are a..."`, the 12 nodes will now simply call the library: 
   - `self.system_prompt = get_compiled_prompt(stage="intent_classifier", model="llama-3.1-8b-instant")`

---

## 3. The "Few-Shot" Library Strategy (100-200 Examples)

You asked for the best strategy to handle 100 to 200 Few-Shot examples tuned explicitly to specific models (e.g., Llama 3 8B handles Intent differently than Llama 3 70B). 

**Strategy: Isolate Examples into JSON/YAML, NOT Python Files**
If you hardcode 200 examples into a `.py` file, the file will become 5,000 lines long and impossible to read. The absolute best practice is to decouple the examples into lightweight data files.

**How we implement it:**
1. **Live Internet Research**: Rather than repeating generic hardcoded examples, we will utilize active internet research and known web corpora tailored specifically for what individual models (e.g., Llama 3 8B) respond best to for that distinct task. We will curate 100 to 200 *real-world* execution examples.
2. Inside `app/prompt_engine/groq_prompts/few_shots/`, we create JSON arrays named after the specific node and the model (e.g., `intent_llama_8b.json`).
3. This JSON file contains these massive arrays of real-world conversational pairs demonstrating exactly how *that specific model* needs to see the input to output the correct JSON.
   ```json
   // intent_llama_8b.json
   [
     {"user": "Can you run an analytical forecast on my Q4 sales CSV?", "assistant_output": "{\"intent\": \"analytics_request\", \"confidence\": 0.99}"},
     {"user": "I need to reset my password.", "assistant_output": "{\"intent\": \"out_of_scope\", \"confidence\": 0.95}"}
   ]
   ```
4. **Dynamic Assembly**: When `get_compiled_prompt()` is called, it:
   - Grabs the Global Persona (Super Prompt).
   - Appends the Base System Prompt for that node.
   - Opens the strictly mapped JSON file for that specific model (e.g., Llama 8B) and automatically formats the 100-200 examples into the prompt string before sending it to the Groq API.

---

## Summary of the Flow:
By implementing this structure, your 12 execution Python scripts shrink drastically in size because the hardcoded text is gone. If you want to change how the Llama 3 8B model classifies Intents, you never touch `app/supervisor/intent.py` again. You simply edit `intent_llama_8b.json` or `base_prompts.py`, and the entire backend updates instantly.

Can you confirm if this directory strategy and JSON Few-Shot decoupling meets your expectations? If so, I will prepare to execute phase 1 of this decoupling!
