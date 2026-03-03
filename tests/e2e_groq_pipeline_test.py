import asyncio
import json
import os
import sys

# Ensure backend modules can be resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.prompt_engine.bootstrapper import PersonaBootstrapper
from app.prompt_engine.groq_prompts.config import PersonaCacheManager

# RAG Nodes
from app.prompt_engine.guard import PromptInjectionGuard
from app.prompt_engine.rewriter import PromptRewriter
from app.supervisor.intent import IntentClassifier
from app.reasoning.complexity import ComplexityClassifier
from app.supervisor.planner import AdaptivePlanner
from app.agents.coder import CoderAgent
from app.retrieval.metadata_extractor import MetadataExtractor
from app.reasoning.synthesis import SynthesisEngine
from app.reasoning.verifier import HallucinationVerifier
from app.rlhf.reward_model import OnlineRewardModel

def print_layer(layer_index, layer_name, input_data, system_prompt, output_data):
    print(f"\n{'='*80}")
    print(f"[{layer_index}/12] LAYER: {layer_name}")
    print(f"{'='*80}")
    print(f"\n--- SYSTEM PROMPT (Dynamically Injected) ---")
    print(system_prompt)
    print(f"\n--- INPUT ---")
    print(input_data)
    print(f"\n--- OUTPUT ---")
    if isinstance(output_data, dict):
        print(json.dumps(output_data, indent=2))
    else:
        print(output_data)
    print("\n")


async def execute_e2e_rag_workflow():
    print("\n================== ENTERPRISE RAG E2E PIPELINE TRACE ==================\n")
    
    # ---------------------------------------------------------
    # LAYER 0: Person Bootstrapper (Universal API Endpoint Sync)
    # ---------------------------------------------------------
    print(">>> INITIALIZING PERSONA BOOTSTRAPPER <<<")
    bot_name = "EnterpriseSecureBot"
    brand_details = "A highly secure corporate AI handling sensitive financial data."
    raw_instructions = "Be strictly professional, evaluate data accurately, and output markdown blocks."
    
    bootstrapper = PersonaBootstrapper()
    expanded_prompt = bootstrapper.expand_persona(raw_instructions, bot_name, brand_details)
    
    # Persisting simulating the POST /api/v1/agents endpoint behavior
    bootstrapper.persist_agent(
        bot_name=bot_name, 
        logo_path="app/static/logos/test_logo.png", 
        brand_details=brand_details, 
        welcome_message="Initializing Secure Ops.", 
        raw_prompt=raw_instructions, 
        expanded_prompt=expanded_prompt
    )
    
    print("\n--- BOOTSTRAPPED PERSONA (Saved to SQLite) ---")
    print(expanded_prompt)
    
    # Force Cache refresh
    cache = PersonaCacheManager()
    cache.refresh_cache()
    
    # The Query to text the system
    query = "Write a deterministic Python script to analyze our Q3 enterprise sales dataset and check if revenue > 1M."
    print(f"\n>>> USER QUERY INJECTED: '{query}' <<<\n")
    
    # ---------------------------------------------------------
    # LAYER 1: Prompt Injection Guard
    # ---------------------------------------------------------
    guard = PromptInjectionGuard()
    guard_res = guard.evaluate(query)
    print_layer(1, "Prompt Injection Guard (llama-prompt-guard-2-86m)", query, guard.system_prompt, guard_res)
    if guard_res.get("is_malicious"):
        print("MALICIOUS PAYLOAD DETECTED (or API 503). CONTINUING FOR TRACE PURPOSES NO MATTER WHAT.")
        
    # ---------------------------------------------------------
    # LAYER 2: Intent Classifier
    # ---------------------------------------------------------
    classifier = IntentClassifier()
    intent_res = await classifier.classify(query)
    print_layer(2, "Intent Classifier", query, classifier.system_prompt, intent_res)
    
    # ---------------------------------------------------------
    # LAYER 3: Complexity Classifier
    # ---------------------------------------------------------
    complexity = ComplexityClassifier()
    complexity_res = await complexity.score(query)
    print_layer(3, "Complexity Classifier", query, complexity.system_prompt, {"complexity_score": complexity_res})
    
    # ---------------------------------------------------------
    # LAYER 4: Prompt Rewriter (MoE)
    # ---------------------------------------------------------
    rewriter = PromptRewriter()
    rewriter_res = await rewriter.rewrite(query, intent_res.get("intent"))
    print_layer(4, "Prompt Rewriter (MoE)", f"Query: {query}\nIntent: {intent_res.get('intent')}", rewriter.system_prompt, rewriter_res)
    
    # ---------------------------------------------------------
    # LAYER 5: Adaptive Planner
    # ---------------------------------------------------------
    planner = AdaptivePlanner()
    plan_input = {"intent": intent_res.get("intent"), "complexity": complexity_res}
    plan_res = planner.generate_plan({"intent": intent_res.get("intent"), "optimizations": {"complexity_score": complexity_res}})
    print_layer(5, "Adaptive Planner", plan_input, "N/A (Heuristic Rule Engine Layer)", plan_res)
    
    # ---------------------------------------------------------
    # LAYER 6: Metadata Extractor
    # ---------------------------------------------------------
    extractor = MetadataExtractor()
    extractor_res = await extractor.extract_filters(query)
    print_layer(6, "Metadata Extractor", query, extractor.system_prompt, extractor_res)
    
    # Let's mock Context Chunks simulating Qdrant DB Retrieval
    dummy_context = [
        {"page_content": "Q3 2023 Enterprise Sales Dataset: Total Revenue reached 1.2M USD. The compliance rules state that scripts must use pandas."},
        {"page_content": "Financial Analysis policy mandates all scripts check for null values."}
    ]
    
    # ---------------------------------------------------------
    # LAYER 7 & 8: Code Engine vs Synthesis Engine branching
    # ---------------------------------------------------------
    # Since it's a code request, it goes to the Coder. We will trace BOTH to prove prompt injection works in both.
    coder = CoderAgent()
    coder_state = {"query": query, "context_chunks": dummy_context}
    coder_res = await coder.ainvoke(coder_state)
    print_layer(7, "Coder Agent (qwen_qwen3-32b)", coder_state, coder.system_prompt, {"answer": coder_res.get("answer"), "confidence": coder_res.get("confidence")})
    
    synthesis = SynthesisEngine()
    sys_state = {"query": query, "context_chunks": dummy_context, "optimizations": {}}
    sys_res = synthesis.synthesize(query, dummy_context)
    print_layer(8, "Synthesis Agent (Deep Reasoning)", sys_state, synthesis.system_prompt, {"answer": sys_res.get("answer")})
    
    # Pick the Coder answer to move forward since intent = code_request
    final_answer = coder_res.get("answer")
    
    # ---------------------------------------------------------
    # LAYER 9: Hallucination Verifier
    # ---------------------------------------------------------
    verifier = HallucinationVerifier()
    verifier_res = verifier.verify(final_answer, dummy_context)
    print_layer(9, "Hallucination Verifier (sarvam-m)", f"Query: {query}\nAnswer Context: {dummy_context}", verifier.system_prompt, verifier_res)
    
    # ---------------------------------------------------------
    # LAYER 10: Online Reward Model (A/B Test Simulation)
    # ---------------------------------------------------------
    reward = OnlineRewardModel()
    reward_res = await reward.score_response(query, dummy_context, final_answer)
    print_layer(10, "Online Reward Model (RLAIF Scoring)", f"Evaluating Answer against Ground Truth", reward.system_prompt, {"weighted_rlaif_score": reward_res})

    print("\n================== END-TO-END TRACE COMPLETED SUCCESSFULLY ==================\n")

if __name__ == "__main__":
    asyncio.run(execute_e2e_rag_workflow())
