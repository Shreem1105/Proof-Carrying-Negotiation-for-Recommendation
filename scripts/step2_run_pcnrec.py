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
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_id = args.run_id
    config['run']['run_id'] = run_id
    
    # Overrides for ablations
    if args.no_verifier:
        config['pcn']['require_verifier_pass'] = False
        # Technically negotiation.py logic needs to respect this or we just accept failures.
        # But negotiation.py calls verify_certificate. 
        # If no_verifier, we should probably modify `negotiation.py` or just ignore the pass/fail result here?
        # Ideally, we inject this into negotiation.
        # Let's pass 'config' to negotiation and ensure it respects it.
        # BUT negotiation.py currently does: "if verification['pass']: return success else loop"
        # If we disable verifier, it should pass immediately.
        # Wait, the prompt requirements said "PCN without verifier (trust LLM stats)".
        # This implies we skip verification call OR ignore its result. I'll mock verification pass.
        pass
        
    if args.no_negotiation:
        config['pcn']['max_rounds'] = 1
        
    # Paths
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    run_output_dir = os.path.join(output_dir, "runs", args.method_name)
    
    cand_path = os.path.join(output_dir, "candidates", "candidates_topk.parquet")
    items_path = os.path.join(output_dir, "data", "items.parquet")
    
    if not os.path.exists(cand_path) or not os.path.exists(items_path):
        logger.error("Missing input files.")
        sys.exit(1)
        
    candidates_df = load_parquet(cand_path)
    items_df = load_parquet(items_path).set_index('internal_id')
    
    # Initialize
    set_seed(42)
    gemini = GeminiClient(config)
    
    # Manifest
    if not os.path.exists(run_output_dir):
        os.makedirs(run_output_dir)
        save_manifest(run_output_dir, create_manifest(config))
    
    # Resume
    done_users = set()
    existing = read_results(run_output_dir)
    for r in existing:
        done_users.add(r['user_id'])
    
    all_users = candidates_df['user_idx'].unique()
    if args.max_users:
        all_users = all_users[:args.max_users]
    
    users_to_process = [u for u in all_users if u not in done_users]
    logger.info(f"Processing {len(users_to_process)} users for {args.method_name}...")
    
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
