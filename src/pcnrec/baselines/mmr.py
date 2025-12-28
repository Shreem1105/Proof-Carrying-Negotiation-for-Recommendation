import numpy as np
import pandas as pd
from pcnrec.utils.logging import setup_logger

logger = setup_logger(__name__)

def jaccard_similarity(genres1, genres2):
    """
    Computes Jaccard similarity between two sets of genres.
    genres can be string (pipe separated) or set/list.
    """
    if isinstance(genres1, str):
        set1 = set(genres1.split('|'))
    else:
        set1 = set(genres1)
        
    if isinstance(genres2, str):
        set2 = set(genres2.split('|'))
    else:
        set2 = set(genres2)
        
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def popularity_similarity(pop_bin1, pop_bin2):
    """
    Fallback similarity if no genres: 1 if same bin, else 0.
    """
    return 1.0 if pop_bin1 == pop_bin2 else 0.0

def mmr_rerank(user_candidates, items_df, lambda_param, top_n):
    """
    Reranks candidates using MMR.
    user_candidates: DataFrame with ['item_idx', 'cand_score']
    items_df: DataFrame index by 'item_idx' (internal_id) with ['genres', 'popularity_bin']
    """
    # Sorted by relevance initially
    candidates = user_candidates.sort_values('cand_score', ascending=False).to_dict('records')
    
    selected_items = []
    
    # Pre-fetch attributes for speed
    # We want a dict: item_idx -> attributes
    # We only care about candidates
    cand_ids = [c['item_idx'] for c in candidates]
    cand_items_info = items_df.loc[cand_ids]
    
    while len(selected_items) < top_n and len(candidates) > 0:
        best_score = -np.inf
        best_item_idx = -1
        best_item_entry = None
        
        # In first iteration, pick max relevance (since similarity penalty logic handles empty S)
        # Standard MMR: argmax_{i in R} [ lambda * sim(u, i) - (1-lambda) * max_{j in S} sim(i, j) ]
        # Here relevance is sim(u, i) -> cand_score
        
        for item in candidates:
            relevance = item['cand_score']
            item_idx = item['item_idx']
            
            # calculate max similarity to selected
            max_sim = 0.0
            for selected in selected_items:
                sel_idx = selected['item_idx']
                
                # Check if genres exist
                # items_df might have NaN for genres if dataset lacks it
                g1 = cand_items_info.at[item_idx, 'genres']
                g2 = cand_items_info.at[sel_idx, 'genres']
                
                if pd.notna(g1) and pd.notna(g2) and g1 != '' and g2 != '':
                    sim = jaccard_similarity(g1, g2)
                else:
                    # Fallback to popularity bin
                    p1 = cand_items_info.at[item_idx, 'popularity_bin']
                    p2 = cand_items_info.at[sel_idx, 'popularity_bin']
                    sim = popularity_similarity(p1, p2)
                
                if sim > max_sim:
                    max_sim = sim
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_item_idx = item_idx
                best_item_entry = item
                
        # Move best item to selected
        if best_item_entry:
            selected_entry = best_item_entry.copy()
            selected_entry['mmr_score'] = best_score
            selected_entry['base_score'] = best_item_entry['cand_score'] # rename
            # Add metadata
            selected_entry['popularity_bin'] = cand_items_info.at[best_item_idx, 'popularity_bin']
            selected_items.append(selected_entry)
            
            # Remove from candidates list
            candidates = [c for c in candidates if c['item_idx'] != best_item_idx]
        else:
            break
            
    return selected_items

def run_mmr_for_users(candidates_df, items_df, lambda_param, top_n):
    """
    Runs MMR for all users in candidates_df.
    """
    logger.info(f"Running MMR with lambda={lambda_param}, top_n={top_n}")
    
    # Ensure items_df is indexed by internal_id (item_idx)
    if items_df.index.name != 'internal_id' and 'internal_id' in items_df.columns:
        items_df = items_df.set_index('internal_id')
    
    results = []
    
    # Group by user
    for user_idx, group in candidates_df.groupby('user_idx'):
        reranked = mmr_rerank(group, items_df, lambda_param, top_n)
        for rank, item in enumerate(reranked):
            item['rank'] = rank + 1
            results.append(item)
            
    return pd.DataFrame(results)
