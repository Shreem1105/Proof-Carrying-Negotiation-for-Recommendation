import pandas as pd
import json
import os
import argparse
import sys
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from pcnrec.utils.io import load_config, load_parquet
from pcnrec.analysis.feasibility import check_feasibility

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = os.path.join(config['dataset']['output_dir'], args.run_id)
    pcn_results_path = os.path.join(output_dir, "runs", "pcnrec", "results.jsonl")
    cand_path = os.path.join(output_dir, "candidates", "candidates_topk.parquet")
    items_path = os.path.join(output_dir, "data", "items.parquet")
    
    if not os.path.exists(pcn_results_path):
        print(f"Results not found: {pcn_results_path}")
        return
        
    print(f"Loading results from {pcn_results_path}...")
    results = []
    with open(pcn_results_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
                
    # Load candidates for ground-truth feasibility check
    print(f"Loading candidates from {cand_path}...")
    cands_df = load_parquet(cand_path)
    # Ensure items metadata for verification re-check if needed
    items_df = load_parquet(items_path)
    if 'item_idx' in items_df.columns:
        items_df = items_df.set_index('item_idx')

    window = config['pcn']['candidate_window']
    constraints = config['constraints']
    
    feasible_but_failed = []
    reasons = []
    
    total_feasible = 0
    total_users = 0
    
    print(f"Analyzing {len(results)} users with W={window}...")
    
    for row in results:
        uid = row['user_id']
        total_users += 1
        
        # 1. Check Feasibility (Ground Truth)
        u_cands = cands_df[cands_df['user_idx'] == uid]
        is_feasible_gt, _ = check_feasibility(u_cands, constraints, top_k=window)
        
        if is_feasible_gt:
            total_feasible += 1
            
            # 2. Check Result Status
            # We trust 'verifier' field if present, or check 'status'
            passed = False
            verifier_res = row.get('verifier', {})
            if verifier_res and verifier_res.get('pass'):
                passed = True
            
            if not passed:
                # This is the target group
                fail_reasons = verifier_res.get('fail_reasons', ["Unknown failure"])
                repair_used = row.get('repair_used', False) # Might not be tracked in jsonl directly, logic check needed
                
                # Check if deterministic repair was attempted?
                # The jsonl doesn't always explicitly save 'deterministic_repair_used' unless we added it.
                # But we can check if 'status' is 'success' or 'fail_max_rounds'
                
                info = {
                    'user_id': uid,
                    'status': row.get('status'),
                    'verifier_reasons': ";".join(fail_reasons),
                    'repair_attempted': 'unknown', # Need to check code or logs
                    'items_selected': len(row.get('selected_item_ids', []))
                }
                feasible_but_failed.append(info)
                reasons.extend(fail_reasons)

    count_failed = len(feasible_but_failed)
    pass_rate = (total_feasible - count_failed) / total_feasible if total_feasible > 0 else 0
    
    print(f"Feasible Users: {total_feasible}/{total_users}")
    print(f"Failed among Feasible: {count_failed}")
    print(f"Pass Rate among Feasible: {pass_rate:.4f}")
    
    if count_failed > 0:
        print("\nTop Failure Reasons:")
        print(pd.Series(reasons).value_counts().head(10))
        
        # Save breakdown
        out_csv = os.path.join(output_dir, "analysis", "pcn_failure_breakdown.csv")
        pd.DataFrame(feasible_but_failed).to_csv(out_csv, index=False)
        print(f"\nSaved breakdown to {out_csv}")
    else:
        print("\nNo failures among feasible users! Win condition met?")

if __name__ == "__main__":
    main()
