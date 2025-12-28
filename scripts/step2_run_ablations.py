import subprocess
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--max_users", type=int, default=None)
    args = parser.parse_args()
    
    # Ablation 1: No Verifier
    cmd1 = f"python scripts/step2_run_pcnrec.py --config {args.config} --run_id {args.run_id} --method_name pcnrec_no_verifier --no_verifier"
    if args.max_users:
        cmd1 += f" --max_users {args.max_users}"
        
    print(f"Running Ablation 1: {cmd1}")
    subprocess.call(cmd1, shell=True)
    
    # Ablation 2: No Negotiation
    cmd2 = f"python scripts/step2_run_pcnrec.py --config {args.config} --run_id {args.run_id} --method_name pcnrec_no_negotiation --no_negotiation"
    if args.max_users:
        cmd2 += f" --max_users {args.max_users}"
        
    print(f"Running Ablation 2: {cmd2}")
    subprocess.call(cmd2, shell=True)

if __name__ == "__main__":
    main()
