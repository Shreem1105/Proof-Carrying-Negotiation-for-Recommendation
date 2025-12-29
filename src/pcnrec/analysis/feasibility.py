import pandas as pd
import numpy as np
from pcnrec.verify.recompute import compute_head_tail_counts, compute_unique_genres
from pcnrec.verify.constraints import check_max_head, check_min_tail, check_min_unique_genres

def check_feasibility(candidates_df: pd.DataFrame, constraints: dict, top_k: int = None):
    """
    Checks if the constraints can be satisfied using ONLY the provided candidates.
    If top_k is specified, only considers the top k candidates by 'cand_score' (assumed sorted or requiring sort).
    
    Returns:
        is_feasible (bool)
        details (dict): {
            'head_count': int,
            'tail_count': int,
            'unique_genres': int,
            'max_possible_tail': int,
            'max_possible_unique_genres': int,
            'fail_reasons': list[str]
        }
    """
    # Use copy to avoid mutating
    df = candidates_df.copy()
    
    # Sort by score if available, though usually candidates are already sorted or we just take the top window
    if 'cand_score' in df.columns:
        df = df.sort_values('cand_score', ascending=False)
        
    if top_k is not None:
        df = df.head(top_k)
        
    # We need to pick exactly 10 items (or whatever top_n is config)
    # But for feasibility "Is it possible to pick 10 items that satisfy?"
    # It's a subset selection problem.
    # However, easy checks:
    # 1. Do we have enough tail items in the window?
    # 2. Do we have enough unique genres in the window?
    
    # Actually, the constraint is on the SELECTED Top-N.
    # So we must verify if there EXISTS a subset of size N (target) within Window (available) that satisfies constraints.
    
    # Constraints usually:
    # - Max Head <= H
    # - Min Tail >= T
    # - Min Unique Genres >= G
    # - No Duplicates (trivial if candidates are unique)
    
    # Assuming target list size is N=10 (standard)
    target_n = 10
    
    pop_config = constraints.get('popularity', {})
    div_config = constraints.get('diversity', {})
    
    max_head = pop_config.get('max_head_in_topn', 10)
    min_tail = pop_config.get('min_tail_in_topn', 0)
    min_genres = div_config.get('min_unique_genres_in_topn', 0)
    
    # Available resources in window
    avail_tail = (df['popularity_bin'] == 'tail').sum()
    avail_head = (df['popularity_bin'] == 'head').sum()
    avail_torso = (df['popularity_bin'] == 'torso').sum() # total - head - tail
    
    # Unique genres in window
    # genres strings like "Action|Comedy"
    all_genres = set()
    for gs in df['genres'].dropna():
        for g in gs.split('|'):
            all_genres.add(g)
    avail_unique_genres = len(all_genres)
    
    fail_reasons = []
    
    # Check 1: Tail Shortage
    # We need to pick min_tail items that are tail.
    if avail_tail < min_tail:
        fail_reasons.append('tail_shortage')
    
    # Check 2: Head Conflict is more subtle. We verify Max Head.
    # If we MUST pick items, and we run out of non-head items, we might be forced to pick head.
    # We need N items total.
    # Non-Head Available = Tail + Torso
    avail_non_head = avail_tail + avail_torso
    # If we pick all available non-head, we still need (N - avail_non_head) items.
    # These MUST come from Head.
    # So min_head_needed = max(0, N - avail_non_head)
    # If min_head_needed > max_head, then we are forced to violate max_head.
    min_head_needed = max(0, target_n - avail_non_head)
    if min_head_needed > max_head:
        fail_reasons.append('head_forced_violation')
        
    # Check 3: Genre Shortage
    # Can we pick N items that cover G genres?
    # This is a set cover problem, hard to solve exactly in general, but usually:
    # If total unique genres in window < min_genres, it's impossible.
    if avail_unique_genres < min_genres:
        fail_reasons.append('genre_shortage_window')
        
    # A stricter greedy check for genres isn't strictly necessary for "feasibility" 
    # unless the window is very small, but simple count is a good lower bound.
    
    is_feasible = len(fail_reasons) == 0
    
    return is_feasible, {
        'avail_tail': int(avail_tail),
        'avail_head': int(avail_head),
        'avail_unique_genres': int(avail_unique_genres),
        'fail_reasons': fail_reasons
    }
