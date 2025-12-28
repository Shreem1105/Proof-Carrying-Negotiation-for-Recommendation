import argparse
import sys
import os
from tqdm import tqdm
import time

sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config, load_parquet
from pcnrec.utils.logging import setup_logger
from pcnrec.utils.seed import set_seed
from pcnrec.llm.gemini_client import GeminiClient
from pcnrec.agents.single_llm_rerank import run_single_llm
from pcnrec.runs.io import append_result_row, read_results, save_manifest
from pcnrec.runs.manifest import create_manifest

logger = setup_logger("step2_run_single_llm")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--max_users", type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_id = args.run_id
    config['run']['run_id'] = run_id
    
    # Paths
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    run_output_dir = os.path.join(output_dir, "runs", "single_llm")
    
    cand_path = os.path.join(output_dir, "candidates", "candidates_topk.parquet")
    if not os.path.exists(cand_path):
        logger.error(f"Candidates not found: {cand_path}")
        sys.exit(1)
        
    candidates_df = load_parquet(cand_path)
    
    # Initialize
    set_seed(42)
    gemini = GeminiClient(config)
    
    # Manifest
    if not os.path.exists(run_output_dir):
        os.makedirs(run_output_dir)
        save_manifest(run_output_dir, create_manifest(config))
    
    # Resume check
    done_users = set()
    existing = read_results(run_output_dir)
    for r in existing:
        done_users.add(r['user_id'])
    
    logger.info(f"Already done: {len(done_users)} users.")
    
    # Filter users
    all_users = candidates_df['user_idx'].unique()
    if args.max_users:
        all_users = all_users[:args.max_users]
        
    users_to_process = [u for u in all_users if u not in done_users]
    logger.info(f"Processing {len(users_to_process)} users...")
    
    for uid in tqdm(users_to_process):
        user_cands = candidates_df[candidates_df['user_idx'] == uid].copy()
        
        start_t = time.time()
        result = run_single_llm(uid, user_cands, config, gemini)
        end_t = time.time()
        
        row = {
            "user_id": int(uid),
            "timing_ms": {"total": (end_t - start_t) * 1000},
            "candidates_shown": user_cands[['item_idx', 'title', 'genres', 'popularity_bin', 'cand_score']].to_dict('records')
        }
        
        if result['result'] == 'success':
            row['selected_item_ids'] = result['selected_item_ids']
            row['status'] = 'success'
        else:
            row['status'] = 'error'
            row['error'] = result.get('error')
            row['selected_item_ids'] = [] 
            
        append_result_row(run_output_dir, row)
        
    logger.info(f"Done. Results in {run_output_dir}")

if __name__ == "__main__":
    main()
