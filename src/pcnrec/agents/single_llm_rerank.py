from pcnrec.llm.gemini_client import GeminiClient
from pcnrec.llm.schemas import RecommendationSelection
from pcnrec.agents.prompts import SYSTEM_PROMPT_SINGLE_LLM
from pcnrec.agents.negotiation import format_candidates

def run_single_llm(user_id, candidates_df, config, gemini_client: GeminiClient):
    """
    Runs a single LLM call to select top_n items.
    """
    top_n = config['pcn']['top_n']
    window_size = config['pcn']['candidate_window']
    candidates_window = candidates_df.head(window_size)
    candidates_str = format_candidates(candidates_window)
    candidates_ids = set(candidates_window['item_idx'].values)
    
    prompt = SYSTEM_PROMPT_SINGLE_LLM.format(
        top_n=top_n,
        candidates=candidates_str
    )
    
    try:
        selection = gemini_client.generate_structured(
            prompt,
            schema_model=RecommendationSelection
        )
        
        # Check subset property locally just in case
        selected = selection.selected_item_ids
        if not set(selected).issubset(candidates_ids):
            # Fallback or truncate?
            # For baseline, we report what it output, but maybe warn.
            # Or enforce it by filtering.
            valid = [s for s in selected if s in candidates_ids]
            # Fill if needed? 
            # Let's return as is, let eval handle metrics or partials.
            pass
            
        return {
            "result": "success",
            "selection": selection,
            "selected_item_ids": selection.selected_item_ids
        }
        
    except Exception as e:
        return {
            "result": "error",
            "error": str(e)
        }
