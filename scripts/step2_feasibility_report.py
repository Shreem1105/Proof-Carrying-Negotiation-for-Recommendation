import argparse
import os
import sys
import pandas as pd
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config
from pcnrec.analysis.feasibility import check_feasibility
from pcnrec.utils.logging import setup_logger

logger = setup_logger("feasibility_report")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_dir = os.path.join(config['dataset']['output_dir'], args.run_id)
    cand_path = os.path.join(run_dir, "candidates", "candidates_topk.parquet")
    items_path = os.path.join(run_dir, "data", "items.parquet")
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(cand_path):
        logger.error(f"Candidates not found at {cand_path}")
        return

    cands_df = pd.read_parquet(cand_path)
    # cands_df already has 'genres' and 'popularity_bin' from generation step
    joined_df = cands_df
    
    # Rename item_idx to item_id for consistency if needed, but logic uses columns directly
    if 'user_idx' in joined_df.columns:
        joined_df = joined_df.rename(columns={'user_idx': 'user_id'})
    
    windows = [10, 20, 30, 50, 80, 100, 120]
    results = []
    
    constraints = config['constraints']
    logger.info(f"Checking constraints: {constraints}")
    
    unique_users = joined_df['user_id'].unique()
    logger.info(f"Analyzing {len(unique_users)} users over windows {windows}")
    
    for w in windows:
        feasible_count = 0
        reasons_counts = {'tail_shortage': 0, 'head_forced_violation': 0, 'genre_shortage_window': 0}
        
        for uid in unique_users:
            # unique candidates for user
            u_cands = joined_df[joined_df['user_id'] == uid]
            
            # check feasibility at window w
            is_feasible, details = check_feasibility(u_cands, constraints, top_k=w)
            
            if is_feasible:
                feasible_count += 1
            else:
                for r in details['fail_reasons']:
                    reasons_counts[r] += 1
                    
        res = {
            'window': w,
            'feasible_rate': feasible_count / len(unique_users),
            'tail_shortage_rate': reasons_counts['tail_shortage'] / len(unique_users),
            'head_conflict_rate': reasons_counts['head_forced_violation'] / len(unique_users),
            'genre_shortage_rate': reasons_counts['genre_shortage_window'] / len(unique_users)
        }
        results.append(res)
        logger.info(f"W={w}: Feasible={res['feasible_rate']:.2f}, TailShortage={res['tail_shortage_rate']:.2f}")
        
    # Save results
    out_csv = os.path.join(out_dir, "feasibility.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    logger.info(f"Saved {out_csv}")
    
    out_json = os.path.join(out_dir, "feasibility_summary.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
