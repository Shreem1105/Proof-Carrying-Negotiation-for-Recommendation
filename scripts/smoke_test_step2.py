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
    print("Starting Smoke Test Step 2...")
    
    run_id = "smoke2"
    
    # Prerequisite: Check if step 1 outputs exist for smoke2, if not run step 1 prep
    if not os.path.exists(f"outputs/{run_id}/candidates/candidates_topk.parquet"):
        print("Running Step 1 Prep for smoke2...")
        # Reduce size for speed
        run_command(f"python scripts/step1_prepare_data.py --config config/config.yaml --run_id {run_id}")
        run_command(f"python scripts/step1_train_candidates.py --config config/config.yaml --run_id {run_id}")
        run_command(f"python scripts/step1_generate_candidates.py --config config/config.yaml --run_id {run_id} --max_users 30")
    
    # 1. Single LLM Baseline
    run_command(f"python scripts/step2_run_single_llm_baseline.py --config config/config.yaml --run_id {run_id} --max_users 10")
    
    # 2. PCN-Rec
    run_command(f"python scripts/step2_run_pcnrec.py --config config/config.yaml --run_id {run_id} --max_users 10")
    
    # 3. Ablations
    run_command(f"python scripts/step2_run_ablations.py --config config/config.yaml --run_id {run_id} --max_users 10")
    
    # 4. Evaluate
    run_command(f"python scripts/step2_evaluate.py --config config/config.yaml --run_id {run_id}")
    
    print("\n--- Smoke Test Step 2 PASSED ---")
    print("Check outputs/smoke2/evaluation_summary.csv")

if __name__ == "__main__":
    main()
