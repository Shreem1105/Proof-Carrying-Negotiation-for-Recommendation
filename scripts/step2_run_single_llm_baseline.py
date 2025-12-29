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
    parser.add_argument("--shard", type=str, default=None, help="Format: index/total, e.g., 0/4")
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
    
    # Load items and train for profile
    data_dir = os.path.join(output_dir, "data")
    items_path = os.path.join(data_dir, "items.parquet")
    train_path = os.path.join(data_dir, "interactions_train.parquet")
    
    items_df = load_parquet(items_path)
    train_df = load_parquet(train_path)
    
    # Ensure indices
    if items_df.index.name != 'item_idx' and 'item_idx' not in items_df.columns:
        # Assuming index is item_idx if not column
        pass
    
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
    all_users = sorted(candidates_df['user_idx'].unique())
    if args.max_users:
        all_users = all_users[:args.max_users]
        
    # Sharding
    if args.shard:
        shard_idx, shard_total = map(int, args.shard.split('/'))
        chunk_size = len(all_users) // shard_total + 1
        start_idx = shard_idx * chunk_size
        end_idx = min((shard_idx + 1) * chunk_size, len(all_users))
        all_users = all_users[start_idx:end_idx]
        logger.info(f"Shard {shard_idx+1}/{shard_total}: Processing {len(all_users)} users")
        
    users_to_process = [u for u in all_users if u not in done_users]
    logger.info(f"Processing {len(users_to_process)} users...")
    
    # Import profile util
    from pcnrec.utils.profiling import get_user_profile
    
    for uid in tqdm(users_to_process):
        user_cands = candidates_df[candidates_df['user_idx'] == uid].copy()
        
        # Verify columns exist: title, genres, popularity_bin
        # They should be in candidates_df
        
        # Profile
        profile_str = get_user_profile(uid, train_df, items_df)
        
        start_t = time.time()
        result = run_single_llm(uid, user_cands, config, gemini, user_profile=profile_str)
        end_t = time.time()
        
        row = {
            "user_id": int(uid),
            "timing_ms": {"total": (end_t - start_t) * 1000},
            "candidates_shown": user_cands[['item_idx', 'title', 'genres', 'popularity_bin', 'cand_score']].head(config['pcn']['candidate_window']).to_dict('records')
        }
        
        row['selected_item_ids'] = result.get('selected_item_ids', [])
        row['status'] = result['result']
        row['fallback_used'] = result.get('fallback_used', False)
        if result['result'] == 'error':
            row['error'] = result.get('error')
            
        append_result_row(run_output_dir, row)
        
    logger.info(f"Done. Results in {run_output_dir}")

if __name__ == "__main__":
    main()
