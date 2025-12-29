import numpy as np

def dcg_at_k(r, k):
    r = np.asarray(r).astype(float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def compute_metrics_for_user(selected_ids, ground_truth_ids, items_df=None, k=10):
    """
    Computes Recall@K, NDCG@K, Tail%, Entropy.
    selected_ids: list of int
    ground_truth_ids: set of int
    items_df: dataframe indexed by internal_id
    """
    # Relevance list for NDCG
    relevance = [1 if i in ground_truth_ids else 0 for i in selected_ids]
    
    recall = sum(relevance) / len(ground_truth_ids) if ground_truth_ids else 0.0
    ndcg = ndcg_at_k(relevance, k)
    
    metrics = {
        f'recall@{k}': recall,
        f'ndcg@{k}': ndcg
    }
    
    if items_df is not None:
        # Tail Exposure
        subset = items_df.loc[selected_ids]
        tail_count = (subset['popularity_bin'] == 'tail').sum()
        metrics[f'tail_prop@{k}'] = tail_count / len(selected_ids) if selected_ids else 0.0
        
    return metrics
