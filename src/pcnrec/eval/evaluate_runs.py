import pandas as pd
import numpy as np
from pcnrec.eval.metrics import compute_metrics_for_user
from pcnrec.runs.io import read_results, save_summary

def evaluate_run(run_dir, test_df, items_df, k=10, subset_users=None):
    """
    Evaluates a single run directory against test set.
    """
    results = read_results(run_dir)
    if not results:
        return {}, pd.DataFrame()
        
    # Build GT lookup
    gt_lookup = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    metrics_list = []
    
    for row in results:
        uid = row['user_id']
        
        if subset_users is not None and uid not in subset_users:
            continue
            
        selected = row['selected_item_ids']
        if items_df is not None:
             selected = [sid for sid in selected if sid in items_df.index]
        
        gt = gt_lookup.get(uid, set())
        
        m = compute_metrics_for_user(selected, gt, items_df, k=k)
        
        # Extended metrics
        verifier_pass = False
        if 'verifier' in row:
             verifier_pass = row['verifier'].get('pass', False)
        elif 'verifier_result' in row:
             verifier_pass = row['verifier_result'].get('pass', False)
        
        # Check if implicitly passed (e.g. sanity baselines assume pass?)
        # Actually sanity baselines don't have verifier_result. 
        # But for comparison table we want "PassRate".
        # We can re-check verification here if we wanted to be 100% sure for baselines.
        # But let's stick to what's in JSON for now. 
        # For sanity baselines, maybe we should run verifier? 
        # The user didn't explicitly ask to run verifier on baselines, but "PassRate" is a key metric.
        # If baselines don't define it, it's 0 or N/A.
        # However, constrained_greedy SHOULD pass.
        
        m['verifier_pass'] = 1.0 if verifier_pass else 0.0
        m['repair_used'] = 1.0 if row.get('deterministic_repair_used', False) else 0.0
        m['fallback_used'] = 1.0 if row.get('fallback_used', False) else 0.0
        m['user_id'] = uid
        
        metrics_list.append(m)
        
    if not metrics_list:
        return {}, pd.DataFrame()

    df_m = pd.DataFrame(metrics_list)
    summary = df_m.mean().to_dict()
    summary['count'] = len(df_m)
    
    # Save (only if full run?)
    if subset_users is None:
        save_summary(run_dir, summary)
    
    return summary, df_m
