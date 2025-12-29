import pandas as pd
import numpy as np

def run_mf_topn(candidates_df, top_n=10):
    """
    Returns top-N items by cand_score for each user.
    """
    # Assuming df sorted by score desc
    return candidates_df.groupby('user_id').head(top_n).groupby('user_id')['item_idx'].apply(list).to_dict()

def run_constrained_greedy(candidates_df, items_df, constraints, top_n=10, window=100):
    """
    Determinisitically selects items to satisfy constraints.
    Returns dict: user_id -> [list of item_ids]
    """
    
    # Pre-process candidates
    # Ensure sorted
    df = candidates_df.sort_values(['user_id', 'cand_score'], ascending=[True, False])
    
    users = df['user_id'].unique()
    results = {}
    
    pop_config = constraints.get('popularity', {})
    div_config = constraints.get('diversity', {})
    
    max_head = pop_config.get('max_head_in_topn', 10)
    min_tail = pop_config.get('min_tail_in_topn', 0)
    min_genres = div_config.get('min_unique_genres_in_topn', 0)
    
def solve_constrained_greedy_user(user_cands_df, items_df, constraints, top_n=10):
    """
    Solves for a single user. Returns list of item_ids.
    """
    valid_items = []
    for _, row in user_cands_df.iterrows():
        iid = row['item_idx']
        if iid not in items_df.index:
            continue
        idata = items_df.loc[iid]
        
        pbin = idata.get('popularity_bin', 'torso')
        gstr = idata.get('genres', "")
        gset = set(gstr.split('|')) if gstr else set()
        
        valid_items.append({
            'id': iid,
            'bin': pbin,
            'genres': gset,
            'score': row['cand_score']
        })
        
    pop_config = constraints.get('popularity', {})
    div_config = constraints.get('diversity', {})
    
    max_head = pop_config.get('max_head_in_topn', 10)
    min_tail = pop_config.get('min_tail_in_topn', 0)
    
    selected = []
    curr_head = 0
    curr_tail = 0
    
    for item in valid_items:
        if len(selected) >= top_n:
            break
            
        slots_rem = top_n - len(selected)
        tail_needed = max(0, min_tail - curr_tail)
        
        is_tail = item['bin'] == 'tail'
        is_head = item['bin'] == 'head'
        
        # Rule 1: Must we take a tail item?
        if slots_rem <= tail_needed:
            if is_tail:
                selected.append(item)
                curr_tail += 1
            # Skip non-tail if we are forced to take tail
            continue 
        
        # Rule 2: Max Head limit
        if is_head and curr_head >= max_head:
            continue
            
        # Otherwise take it
        selected.append(item)
        if is_tail: curr_tail += 1
        if is_head: curr_head += 1
        
    return [x['id'] for x in selected]

def run_constrained_greedy(candidates_df, items_df, constraints, top_n=10, window=100):
    """
    Determinisitically selects items to satisfy constraints.
    Returns dict: user_id -> [list of item_ids]
    """
    
    # Pre-process candidates
    # Ensure sorted
    df = candidates_df.sort_values(['user_id', 'cand_score'], ascending=[True, False])
    
    users = df['user_id'].unique()
    results = {}
    
    for uid in users:
        u_cands = df[df['user_id'] == uid].head(window)
        results[uid] = solve_constrained_greedy_user(u_cands, items_df, constraints, top_n)
        
    return results
