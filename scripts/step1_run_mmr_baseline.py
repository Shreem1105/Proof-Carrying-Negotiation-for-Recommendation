import argparse
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

from pcnrec.utils.io import load_config, load_parquet, save_parquet
from pcnrec.utils.logging import setup_logger
from pcnrec.baselines.mmr import run_mmr_for_users

logger = setup_logger("step1_run_mmr_baseline")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_id = args.run_id
    
    output_dir = os.path.join(config['dataset']['output_dir'], run_id)
    cand_path = os.path.join(output_dir, "candidates", "candidates_topk.parquet")
    data_dir = os.path.join(output_dir, "data")
    baseline_dir = os.path.join(output_dir, "baselines")
    
    if not os.path.exists(cand_path):
        logger.error(f"Candidates not found: {cand_path}.")
        sys.exit(1)
        
    logger.info("Loading candidates and items...")
    candidates_df = load_parquet(cand_path)
    items_df = load_parquet(os.path.join(data_dir, "items.parquet"))
    
    lambda_param = config['mmr']['lambda']
    top_n = config['mmr']['top_n']
    
    logger.info("Running MMR reranking...")
    reranked_df = run_mmr_for_users(candidates_df, items_df, lambda_param, top_n)
    
    out_path = os.path.join(baseline_dir, "mmr_topn.parquet")
    logger.info(f"Saving reranked lists to {out_path}")
    save_parquet(reranked_df, out_path)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
