from typing import List, Dict

def compute_head_tail_counts(selected_ids: List[int], items_df):
    """
    Computes count of head, torso, tail items in selection.
    items_df index should be internal_id.
    """
    # Create valid subset
    subset = items_df.loc[selected_ids]
    
    counts = subset['popularity_bin'].value_counts()
    
    return {
        'head': int(counts.get('head', 0)),
        'torso': int(counts.get('torso', 0)),
        'tail': int(counts.get('tail', 0))
    }

def compute_unique_genres(selected_ids: List[int], items_df):
    """
    Computes number of unique genres in selection.
    """
    subset = items_df.loc[selected_ids]
    
    unique_genres = set()
    for genres_str in subset['genres']:
        if isinstance(genres_str, str):
            for g in genres_str.split('|'):
                if g: unique_genres.add(g)
    
    return len(unique_genres)

def compute_entropy_proxy(selected_ids: List[int], items_df):
    """
    A simple proxy for entropy: number of unique popularities or bins.
    Real entropy might need probability distribution. 
    Let's just return None for now unless we need it for constraints.
    """
    return None
