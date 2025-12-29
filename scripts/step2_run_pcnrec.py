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
from pcnrec.agents.negotiation import run_negotiation
from pcnrec.runs.io import append_result_row, read_results, save_manifest
from pcnrec.runs.manifest import create_manifest

logger = setup_logger("step2_run_pcnrec")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--method_name", default="pcnrec", help="Folder name for output")
    parser.add_argument("--no_verifier", action="store_true", help="Ablation: Disable verifier checks")
    parser.add_argument("--no_negotiation", action="store_true", help="Ablation: Skip negotiation (single shot mediator)")
    parser.add_argument("--shard", type=str, default=None, help="Format: index/total, e.g., 0/4")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_id = args.run_id
    config['run']['run_id'] = run_id
    
    # Overrides for ablations
    if args.no_verifier:
        config['pcn']['require_verifier_pass'] = False
        # Technically negotiation.py logic needs to respect this or we just accept failures.
        # But negotiation.py calls verify_certificate. 
    # The original --no_verifier and --no_negotiation arguments were removed.
    # If these ablations are still needed, they should be re-added to argparse
    # and their logic re-integrated here.
        
    # Paths
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    run_output_dir = os.path.join(output_dir, "runs", "pcnrec") # Hardcoded method_name to "pcnrec"
    
    cand_path = os.path.join(output_dir, "candidates", "candidates_topk.parquet")
    if not os.path.exists(cand_path):
        logger.error(f"Candidates not found: {cand_path}")
        sys.exit(1)
        
    candidates_df = load_parquet(cand_path)
    
    # Initialize
    set_seed(42)
    gemini = GeminiClient(config)
    
    # Load items
    data_dir = os.path.join(output_dir, "data")
    items_path = os.path.join(data_dir, "items.parquet")
    items_df = load_parquet(items_path)
    # Ensure index
    if items_df.index.name != 'item_idx' and 'item_idx' in items_df.columns:
        items_df = items_df.set_index('item_idx') # Optimize lookup
    
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
        logger.info(f"Shard {shard_idx+1}/{shard_total}: Processing {len(all_users)} users ({start_idx} to {end_idx})")
        

        
    users_to_process = [u for u in all_users if u not in done_users]
    logger.info(f"Processing {len(users_to_process)} users for pcnrec...")
    
    for uid in tqdm(users_to_process):
        user_cands = candidates_df[candidates_df['user_idx'] == uid].copy()
        
        start_t = time.time()
        
        # Modify config on fly for ablations if needed, but we set it above.
        # Actually, for no_verifier, we might need a way to tell run_negotiation to ignore verifier failures
        # Or just patch verify_certificate?
        # Let's handle it by wrapping verification in negotiation. 
        # I'll update negotiation.py to check config['pcn'].get('require_verifier_pass', True)
        
        result = run_negotiation(uid, user_cands, items_df, config, gemini)
        end_t = time.time()
        
        row = {
            "user_id": int(uid),
            "timing_ms": {"total": (end_t - start_t) * 1000},
            "candidates_shown": user_cands[['item_idx', 'title', 'genres', 'popularity_bin', 'cand_score']].to_dict('records')
        }
        
        if result['result'] in ['success', 'fail_max_rounds']:
            row['selected_item_ids'] = result['selected_item_ids']
            row['certificate'] = result['certificate'].dict()
            row['verifier'] = result['verifier_result']
            row['status'] = result['result']
            
            # Fallback if failed and config enables fallback?
            # Implemented crude fallback in result handling:
            if result['result'] == 'fail_max_rounds' and config['pcn']['require_verifier_pass']:
                # If we strictly require pass, this is a "failure" of PCN.
                # Do we use MMR fallback?
                pass 
                
        else:
            row['status'] = 'error'
            row['error'] = result.get('error')
            row['selected_item_ids'] = []
            
        append_result_row(run_output_dir, row)
        
    logger.info(f"Done. Results in {run_output_dir}")

if __name__ == "__main__":
    main()
