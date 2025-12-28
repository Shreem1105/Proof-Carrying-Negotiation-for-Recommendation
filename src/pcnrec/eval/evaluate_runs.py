import pandas as pd
import numpy as np
from pcnrec.eval.metrics import compute_metrics_for_user
from pcnrec.runs.io import read_results, save_summary

def evaluate_run(run_dir, test_df, items_df, k=10):
    """
    Evaluates a single run directory against test set.
    """
    results = read_results(run_dir)
    if not results:
        return {}
        
    # Build GT lookup
    # test_df: user_idx, item_idx
    gt_lookup = test_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
    
    metrics_list = []
    verifier_passes = 0
    total = 0
    
    for row in results:
        uid = row['user_id']
        selected = row['selected_item_ids']
        
        gt = gt_lookup.get(uid, set())
        
        m = compute_metrics_for_user(selected, gt, items_df, k=k)
        metrics_list.append(m)
        
        # Verifier rate (for PCN)
        if 'verifier' in row:
            if row['verifier'].get('pass', False):
                verifier_passes += 1
        elif 'verifier_result' in row: # handle if named differently
             if row['verifier_result'].get('pass', False):
                verifier_passes += 1
        
        total += 1
        
    df_m = pd.DataFrame(metrics_list)
    summary = df_m.mean().to_dict()
    
    if total > 0:
        summary['verifier_pass_rate'] = verifier_passes / total
    else:
        summary['verifier_pass_rate'] = 0.0
        
    summary['count'] = total
    
    # Save
    save_summary(run_dir, summary)
    
    return summary, df_m
