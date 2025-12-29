from pcnrec.llm.gemini_client import GeminiClient
from pcnrec.llm.schemas import RecommendationSelection
from pcnrec.agents.prompts import SYSTEM_PROMPT_SINGLE_LLM
from pcnrec.agents.negotiation import format_candidates

def run_single_llm(user_id, candidates_df, config, gemini_client, user_profile: str = "User profile not available."):
    """
    Runs a single LLM call to select top_n items.
    """
    top_n = config['pcn']['top_n']
    window_size = config['pcn']['candidate_window']
    candidates_window = candidates_df.head(window_size)
    candidates_str = format_candidates(candidates_window)
    candidates_ids = set(candidates_window['item_idx'].values)
    
    # Fallback default
    fallback_items = candidates_window.head(top_n)['item_idx'].tolist()
    
    prompt = SYSTEM_PROMPT_SINGLE_LLM.format(
        top_n=top_n,
        candidates=candidates_str,
        user_profile=user_profile
    )
    
    try:
        selection = gemini_client.generate_structured(
            prompt,
            schema_model=RecommendationSelection
        )
        
        # Check subset property locally
        selected = selection.selected_item_ids
        
        # Validation: All selected must be in candidate window
        invalid_ids = [s for s in selected if s not in candidates_ids]
        
        if invalid_ids:
            # Fallback
            return {
                "result": "success",
                "selection": selection, # Keep original object for reference? Or null?
                "selected_item_ids": fallback_items, # FALLBACK
                "fallback_used": True,
                "invalid_reason": f"Hallucinated items: {invalid_ids}"
            }
            
        return {
            "result": "success",
            "selection": selection,
            "selected_item_ids": selected,
            "fallback_used": False
        }
        
    except Exception as e:
        return {
            "result": "error",
            "error": str(e),
            "selected_item_ids": fallback_items, # FALLBACK on error
            "fallback_used": True
        }
