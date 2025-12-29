from pcnrec.verify.recompute import compute_head_tail_counts, compute_unique_genres
from pcnrec.verify.constraints import check_no_duplicates, check_max_head, check_min_tail, check_min_unique_genres
from pcnrec.llm.schemas import ProofCertificate

def verify_certificate(certificate: ProofCertificate, items_df, candidates_shown_ids: set):
    """
    Verifies the certificate against constraints using trusted items_df.
    Also checks that selected items are a subset of candidates_shown.
    """
    selected_ids = certificate.selected_item_ids
    config_constraints = certificate.constraints.model_dump()
    
    reasons = []
    passed = True
    
    # 0. Subset Check
    if not set(selected_ids).issubset(candidates_shown_ids):
        passed = False
        reasons.append(f"Selection contains items not in candidate window: {set(selected_ids) - candidates_shown_ids}")
        # Fail hard on this? Yes.
        return {
            "pass": False,
            "reasons": reasons,
            "recomputed": {}
        }

    # 1. Recompute Stats
    pop_counts = compute_head_tail_counts(selected_ids, items_df)
    unique_genres = compute_unique_genres(selected_ids, items_df)
    
    recomputed = {
        "head_count": pop_counts['head'],
        "tail_count": pop_counts['tail'],
        "unique_genres": unique_genres
    }

    # 2. Check Constraints
    
    # Safety: No Duplicates
    if config_constraints.get('safety', {}).get('no_duplicates', True):
        if not check_no_duplicates(selected_ids):
            passed = False
            reasons.append("Duplicate items found.")

    # Popularity: Max Head
    max_head = config_constraints.get('popularity', {}).get('max_head_in_topn')
    if max_head is not None:
        if not check_max_head(pop_counts['head'], max_head):
            passed = False
            reasons.append(f"Too many head items: {pop_counts['head']} > {max_head}")

    # Popularity: Min Tail
    min_tail = config_constraints.get('popularity', {}).get('min_tail_in_topn')
    if min_tail is not None:
        if not check_min_tail(pop_counts['tail'], min_tail):
            passed = False
            reasons.append(f"Too few tail items: {pop_counts['tail']} < {min_tail}")

    # Diversity: Min Unique Genres
    min_genres = config_constraints.get('diversity', {}).get('min_unique_genres_in_topn')
    if min_genres is not None:
        if not check_min_unique_genres(unique_genres, min_genres):
            passed = False
            reasons.append(f"Low diversity: {unique_genres} < {min_genres} unique genres")

    return {
        "pass": passed,
        "reasons": reasons,
        "recomputed": recomputed
    }
