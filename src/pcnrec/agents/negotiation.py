import json
import hashlib
from pcnrec.llm.gemini_client import GeminiClient
from pcnrec.llm.schemas import ProofCertificate, NegotiationRound, ComputedStats
from pcnrec.agents.prompts import SYSTEM_PROMPT_USER_ADVOCATE, SYSTEM_PROMPT_PLATFORM_POLICY, SYSTEM_PROMPT_MEDIATOR
from pcnrec.verify.verifier import verify_certificate
import pandas as pd

def format_candidates(candidates_df):
    """
    Formats candidates for LLM prompt.
    """
    records = candidates_df[['item_idx', 'title', 'genres', 'popularity_bin', 'cand_score']].to_dict('records')
    # Use item_idx as ID
    return json.dumps(records, indent=2)

def run_negotiation(user_id, candidates_df, items_df, config, gemini_client: GeminiClient):
    """
    Runs the PCN negotiation loop.
    """
    top_n = config['pcn']['top_n']
    max_rounds = config['pcn']['max_rounds']
    constraints = config['constraints']
    
    # Window
    window_size = config['pcn']['candidate_window']
    candidates_window = candidates_df.head(window_size)
    candidates_str = format_candidates(candidates_window)
    candidates_ids = set(candidates_window['item_idx'].values)
    
    constraints_str = json.dumps(constraints, indent=2)
    
    # Round 1: Agents
    # User Advocate
    user_prompt = f"Candidate List:\n{candidates_str}\n\nWhat are the best items for this user?"
    user_summary = gemini_client.generate_text(user_prompt, system_instruction=SYSTEM_PROMPT_USER_ADVOCATE)
    
    # Policy Agent
    policy_prompt = f"Candidate List:\n{candidates_str}\n\nEnforce these constraints:\n{constraints_str}"
    policy_summary = gemini_client.generate_text(policy_prompt, system_instruction=SYSTEM_PROMPT_PLATFORM_POLICY.format(constraints=constraints_str))
    
    verifier_feedback = "None"
    
    for round_id in range(1, max_rounds + 1):
        # Mediator
        mediator_prompt = SYSTEM_PROMPT_MEDIATOR.format(
            top_n=top_n,
            constraints=constraints_str,
            candidates=candidates_str,
            user_summary=user_summary,
            policy_summary=policy_summary,
            feedback=verifier_feedback
        )
        
        try:
            certificate = gemini_client.generate_structured(
                mediator_prompt, 
                schema_model=ProofCertificate,
                system_instruction="You are a JSON-speaking Mediator."
            )
            
            # Post-processing: Compute signature
            sig_content = f"{user_id}-{certificate.selected_item_ids}-{constraints}-{config['run']['run_id']}"
            certificate.signature = hashlib.sha256(sig_content.encode()).hexdigest()
            
            # Verify
            verification = verify_certificate(certificate, items_df, candidates_ids)
            
            # Ablation check
            require_pass = config.get('pcn', {}).get('require_verifier_pass', True)
            
            if verification['pass'] or not require_pass:
                 return {
                     "result": "success",
                     "certificate": certificate,
                     "verifier_result": verification,
                     "selected_item_ids": certificate.selected_item_ids
                 }
            else:
                # Feedback loop
                verifier_feedback = f"Verification Failed. Reasons: {verification['reasons']}"
                # If last round, fail
                if round_id == max_rounds:
                     return {
                         "result": "fail_max_rounds",
                         "certificate": certificate,
                         "verifier_result": verification,
                         "selected_item_ids": certificate.selected_item_ids
                     }
                     
        except Exception as e:
            # If LLM fails completely
            return {
                "result": "error",
                "error": str(e)
            }
            
    return {"result": "unknown_failure"}
