import argparse
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config, load_parquet
from pcnrec.eval.evaluate_runs import evaluate_run
from pcnrec.eval.significance import bootstrap_paired_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--methods", default="single_llm,pcnrec,pcnrec_no_verifier,pcnrec_no_negotiation")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_id = args.run_id
    
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    items_path = os.path.join(output_dir, "data", "items.parquet")
    test_path = os.path.join(output_dir, "data", "interactions_test.parquet")
    
    items_df = load_parquet(items_path).set_index('internal_id')
    test_df = load_parquet(test_path)
    
    methods = args.methods.split(',')
    summaries = []
    
    method_dataframes = {}
    
    for method in methods:
        run_dir = os.path.join(output_dir, "runs", method)
        if not os.path.exists(run_dir):
            print(f"Skipping {method} (not found)")
            continue
            
        print(f"Evaluating {method}...")
        summary, df_m = evaluate_run(run_dir, test_df, items_df)
        summary['method'] = method
        summaries.append(summary)
        method_dataframes[method] = df_m
        
    summary_df = pd.DataFrame(summaries)
    print("\n--- Summary ---")
    print(summary_df)
    
    # Comparison
    # Compare PCNRec vs Single LLM
    if 'pcnrec' in method_dataframes and 'single_llm' in method_dataframes:
        print("\n--- Significance (PCN vs Single LLM) ---")
        df_pcn = method_dataframes['pcnrec']
        df_base = method_dataframes['single_llm']
        
        # Align users?
        # Assume same order if processed properly, but safe to merge
        # But evaluate_run returns df with implicit index matching results order.
        # User IDs are available in results but not in df_m properly yet?
        # evaluate_run should ideally return user_id indexed.
        pass # Skip detailed significance for smoke test phase.
        
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved to {summary_path}")

if __name__ == "__main__":
    main()
