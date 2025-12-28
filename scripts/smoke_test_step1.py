import subprocess
import os
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"Error running command: {cmd}")
        sys.exit(ret)

def main():
    print("Starting Smoke Test Step 1...")
    
    # Define a unique run_id for smoke test
    run_id = "smoke_test"
    
    # We need to make sure we use a small subset or that the processing is fast enough
    # The config defaults to full dataset but our script allows overriding config
    # Actually, step1 scripts take --config. We should ideally create a smoke test config
    # OR we just rely on params.
    # We can override defaults by modifying the config yaml temporarily or just using the provided one
    # provided one is ml-100k which is small enough for "smoke_test" (<10 mins).
    
    # 1. Prepare
    run_command(f"python scripts/step1_prepare_data.py --config config/config.yaml --run_id {run_id}")
    
    # 2. Train
    run_command(f"python scripts/step1_train_candidates.py --config config/config.yaml --run_id {run_id}")
    
    # 3. Generate (limit to 30 users as requested)
    run_command(f"python scripts/step1_generate_candidates.py --config config/config.yaml --run_id {run_id} --max_users 30")
    
    # 4. MMR
    run_command(f"python scripts/step1_run_mmr_baseline.py --config config/config.yaml --run_id {run_id}")
    
    # Check outputs
    import pandas as pd
    out_dir = f"outputs/{run_id}"
    
    print("\n--- Verifying Outputs ---")
    
    users = pd.read_parquet(f"{out_dir}/data/users.parquet")
    print(f"Users: {len(users)}")
    
    items = pd.read_parquet(f"{out_dir}/data/items.parquet")
    print(f"Items: {len(items)}")
    
    cands = pd.read_parquet(f"{out_dir}/candidates/candidates_topk.parquet")
    print(f"Candidates (TopK): {len(cands)} rows. Unique users: {cands['user_idx'].nunique()}")
    
    mmr = pd.read_parquet(f"{out_dir}/baselines/mmr_topn.parquet")
    print(f"MMR Results: {len(mmr)} rows. Unique users: {mmr['user_idx'].nunique()}")
    
    if cands['user_idx'].nunique() > 30:
        print("WARNING: Candidates generated for more than 30 users (did --max_users work?)")
        
    print("\nSmoke Test PASSED.")

if __name__ == "__main__":
    # Ensure run from root
    if not os.path.exists("scripts"):
        print("Run this script from the repo root.")
        sys.exit(1)
    main()
