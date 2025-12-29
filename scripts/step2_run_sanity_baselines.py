import argparse
import os
import sys
import pandas as pd
import json

sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config, ensure_dir
from pcnrec.utils.logging import setup_logger
from pcnrec.baselines.sanity import run_mf_topn, run_constrained_greedy
from pcnrec.baselines.mmr import run_mmr_for_users
from pcnrec.runs.io import append_result_row

logger = setup_logger("step2_sanity_baselines")

def save_baseline_results(output_dir, method_name, results_dict):
    """
    Saves dict {uid: [items]} to jsonl
    """
    method_dir = os.path.join(output_dir, "runs", method_name)
    ensure_dir(method_dir)
    res_path = os.path.join(method_dir, "results.jsonl")
    
    # Clear existing
    if os.path.exists(res_path):
        os.remove(res_path)
        
    count = 0
    with open(res_path, 'w') as f:
        for uid, items in results_dict.items():
            row = {
                "user_id": int(uid),
                "selected_item_ids": [int(i) for i in items],
                "method": method_name,
                "status": "success"
            }
            f.write(json.dumps(row) + "\n")
            count += 1
    logger.info(f"Saved {count} rows to {res_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--max_users", type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_dir = os.path.join(config['dataset']['output_dir'], args.run_id)
    cand_path = os.path.join(run_dir, "candidates", "candidates_topk.parquet")
    items_path = os.path.join(run_dir, "data", "items.parquet")
    
    if not os.path.exists(cand_path):
        logger.error("Candidates not found.")
        return
        
    cands_df = pd.read_parquet(cand_path)
    items_df = pd.read_parquet(items_path)
    
    # Rename user_idx -> user_id
    if 'user_idx' in cands_df.columns:
        cands_df = cands_df.rename(columns={'user_idx': 'user_id'})
        
    if args.max_users:
        users = cands_df['user_id'].unique()[:args.max_users]
        cands_df = cands_df[cands_df['user_id'].isin(users)]
        
    top_n = 10 # Hardcoded or from config?
    # Config has config['mmr']['top_n'], let's use that for all
    if 'mmr' in config and 'top_n' in config['mmr']:
        top_n = config['pcn']['top_n']
    window = config['pcn']['candidate_window']
    
    logger.info(f"Running baselines with top_n={top_n}, constrained_greedy_window={window}")
    
    # 1. mf_topn
    logger.info("Running mf_topn...")
    mf_res = run_mf_topn(cands_df, top_n=top_n)
    save_baseline_results(run_dir, "mf_topn", mf_res)
    
    # 2. constrained_greedy
    logger.info("Running constrained_greedy...")
    # Need constraints
    constraints = config['constraints']
    cg_res = run_constrained_greedy(cands_df, items_df, constraints, top_n=top_n, window=window)
    save_baseline_results(run_dir, "constrained_greedy", cg_res)
    
    # 3. mmr
    logger.info("Running mmr...")
    lambda_param = config.get('mmr', {}).get('lambda', 0.5)
    # run_mmr_for_users returns DF
    # run_mmr_for_users expects user_idx
    mmr_input = cands_df.rename(columns={'user_id': 'user_idx'})
    mmr_df = run_mmr_for_users(mmr_input, items_df, lambda_param, top_n)
    # Convert DF to dict
    mmr_res = mmr_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    save_baseline_results(run_dir, "mmr", mmr_res)

if __name__ == "__main__":
    main()
