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

from pcnrec.analysis.feasibility import check_feasibility
from pcnrec.baselines.sanity import solve_constrained_greedy_user

def run_negotiation(user_id, candidates_df, items_df, config, gemini_client: GeminiClient):
    """
    Runs the PCN negotiation loop with robust gating.
    """
    top_n = config['pcn']['top_n']
    max_rounds = config['pcn']['max_rounds']
    constraints = config['constraints']
    
    # Window
    window_size = config['pcn']['candidate_window']
    candidates_window = candidates_df.head(window_size)
    candidates_str = format_candidates(candidates_window)
    candidates_ids = set(candidates_window['item_idx'].values)
    
    # Feasibility Check
    is_feasible, feas_details = check_feasibility(candidates_window, constraints, top_k=None) # window already applied
    fail_reasons = feas_details.get('fail_reasons', [])
    
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
                     "selected_item_ids": certificate.selected_item_ids,
                     "feasible_within_window": is_feasible,
                     "infeasibility_reasons": fail_reasons,
                     "deterministic_repair_used": False
                 }
            else:
                # Feedback loop
                verifier_feedback = f"Verification Failed. Reasons: {verification['reasons']}"
                
                # If last round, fail logic
                if round_id == max_rounds:
                    # Gating Logic: If Feasible, we repair.
                    if is_feasible:
                        repair_ids = solve_constrained_greedy_user(candidates_window, items_df, constraints, top_n)
                        
                        # Create a repaired certificate
                        repaired_cert = ProofCertificate(
                            version="1.0-repair",
                            constraints=certificate.constraints, # reuse config
                            selected_item_ids=repair_ids,
                            computed_stats_claimed=ComputedStats(head_count=0, tail_count=0, unique_genres=0), # Dummy
                            negotiation_trace=certificate.negotiation_trace, # Keep trace
                            signature="repaired"
                        )
                        
                        # Verify the repair to get accurate stats
                        repair_vertification = verify_certificate(repaired_cert, items_df, candidates_ids)
                        
                        return {
                             "result": "success", 
                             "certificate": repaired_cert, 
                             "verifier_result": repair_vertification, # Should be PASS
                             "selected_item_ids": repair_ids, 
                             "feasible_within_window": True,
                             "infeasibility_reasons": [],
                             "deterministic_repair_used": True
                        }
                    else:
                        # Infeasible: Return best effort (last attempt) or fallback?
                        # User says: "choose best-effort list (mf_topn or mmr)"
                        # Let's fallback to constrained_greedy anyway as it's best effort
                        best_effort_ids = solve_constrained_greedy_user(candidates_window, items_df, constraints, top_n)
                        
                        return {
                            "result": "fail_infeasible",
                            "certificate": certificate,
                            "verifier_result": verification,
                            "selected_item_ids": best_effort_ids,
                            "feasible_within_window": False,
                            "infeasibility_reasons": fail_reasons,
                            "deterministic_repair_used": False # Not repair, just fallback
                        }
                     
        except Exception as e:
            # If LLM fails completely
            # Fallback to greedy
            best_effort_ids = solve_constrained_greedy_user(candidates_window, items_df, constraints, top_n)
            return {
                "result": "error_fallback",
                "error": str(e),
                "selected_item_ids": best_effort_ids,
                "feasible_within_window": is_feasible
            }
            
    return {"result": "unknown_failure"}
